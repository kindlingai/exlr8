"""Check an exlr8 v2 release fileset against FORMAT.md.

Standard library only, so it runs on a host without numpy, torch or safetensors.

    python -m validator.validate ROOT            # structure: headers, records, index tables, sidecar sizes
    python -m validator.validate ROOT --full     # also SHA-256 of every file against its sidecar and set_id

Exit status: 0 when no check fails (warnings allowed), 1 when a check fails, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import array
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field

ALIGN = 65536
CODEBOOKS = {"mcg": "0xCBAC1FED", "mul1": "0x83DCD12D"}
PROJS = ("gate", "up", "down")
DTYPE_SIZE = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
              "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}

# Record formats (FORMAT.md, "Dense, vocab, extra and split files").
TRELLIS, INT4, MXFP8, FP8_BLOCK, FP8_ROW, BF16, FP32 = 1, 2, 3, 4, 5, 6, 7
DATA = {TRELLIS: ("trellis", "suh", "svh"), INT4: ("q4", "s4"), MXFP8: ("fp8", "e8m0"), FP8_BLOCK: ("fp8", "scale"),
        FP8_ROW: ("fp8", "scale"), BF16: ("bf16",), FP32: ("fp32",)}
DATA_SUFFIXES = {s for v in DATA.values() for s in v}

SUPPORT_REQUIRED = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                    "chat_template.jinja", "LICENSE")
SUPPORT_OPTIONAL = {"README.md", "expert-build-summary.json", "processor_config.json", "preprocessor_config.json",
                    "video_preprocessor_config.json", "special_tokens_map.json"}
OPTIONAL_DIRS = {"assets"}
# Hugging Face and git bookkeeping. The validator skips these.
IGNORED = {".gitattributes", ".git", ".cache", ".huggingface"}
# Plain safetensors files with plain tensors, outside the exlr8 encoding.
OPTIONAL_PLAIN = {"dense/vision-bf16.safetensors"}

RE_DENSE = re.compile(r"dense/exlr8-dense-(\d{3})\.safetensors")
RE_EXTRA = re.compile(r"dense/exlr8-extra-([A-Za-z0-9_.-]+)\.safetensors")
RE_SPLIT = re.compile(r"dense/exlr8-split-([A-Za-z0-9_.-]+)\.p(\d+)\.safetensors")
RE_INDEX = re.compile(r"experts/exlr8-layer-(\d{3})\.index\.safetensors")
RE_TRELLIS = re.compile(r"experts/exlr8-layer-(\d{3})\.k(\d)\.safetensors")
RE_MANIFEST = re.compile(r"manifests/exlr8-manifest-([A-Za-z0-9_.-]+)\.json")
RE_SLICE = re.compile(r"index-slices/tp(\d+)/k(\d)/rank(\d+)\.safetensors")
RE_SIDECAR = re.compile(r"([0-9a-f]{64}) (\d+)\s*")

# Report order. Each check prints one line.
CHECKS = ("support files", "layout", "safetensors", "dense records", "vocab", "optional members", "expert indexes",
          "expert trellis files", "expert offsets", "error tables", "manifests", "index slices", "hash sidecars",
          "weight set checksums")


class FormatError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Report


@dataclass
class Check:
    findings: list[tuple[str, str]] = field(default_factory=list)  # (status, message)
    notes: list[str] = field(default_factory=list)
    ran: bool = False

    @property
    def status(self) -> str:
        states = {s for s, _ in self.findings}
        return "fail" if "fail" in states else "warn" if "warn" in states else "pass"


class Report:
    def __init__(self):
        self.checks = {name: Check() for name in CHECKS}

    def note(self, check: str, text: str) -> None:
        self.checks[check].ran = True
        self.checks[check].notes.append(text)

    def warn(self, check: str, text: str) -> None:
        self.checks[check].ran = True
        self.checks[check].findings.append(("warn", text))

    def fail(self, check: str, text: str) -> None:
        self.checks[check].ran = True
        self.checks[check].findings.append(("fail", text))

    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks.values())

    def render(self, out, details: int) -> None:
        counts = {"pass": 0, "warn": 0, "fail": 0}
        for name, c in self.checks.items():
            if not c.ran:
                continue
            counts[c.status] += 1
            head = "; ".join(c.notes)
            nf = sum(1 for s, _ in c.findings if s == "fail")
            nw = len(c.findings) - nf
            tail = ", ".join(x for x in (f"{nf} failed" if nf else "", f"{nw} warnings" if nw else "") if x)
            print(f"{c.status.upper():4}  {name:22} {head}{' -- ' if head and tail else ''}{tail}", file=out)
            shown = sorted(c.findings, key=lambda f: f[0] != "fail")
            limit = len(shown) if details < 0 else details
            for status, text in shown[:limit]:
                print(f"        {status}: {text}", file=out)
            if len(shown) > limit:
                print(f"        ... {len(shown) - limit} more (--details -1 shows all)", file=out)
        total = sum(counts.values())
        print(f"\nsummary: {total} checks, {counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail", file=out)


# ---------------------------------------------------------------------------------------------------------------------
# Safetensors


class SafeFile:
    """A parsed safetensors header. Raises FormatError when the file is not valid safetensors."""

    def __init__(self, path: str):
        self.path = path
        self.size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise FormatError(f"{self.size} bytes, shorter than the 8-byte header length")
            n = struct.unpack("<Q", raw)[0]
            if n > self.size - 8 or n > 100 << 20:
                raise FormatError(f"header length {n} runs past the end of the {self.size}-byte file")
            blob = f.read(n)
        try:
            header = json.loads(blob)
        except ValueError as e:
            raise FormatError(f"header is not JSON: {e}") from None
        if not isinstance(header, dict):
            raise FormatError("header is not a JSON object")
        meta = header.pop("__metadata__", None) or {}
        if not isinstance(meta, dict) or not all(isinstance(v, str) for v in meta.values()):
            raise FormatError("__metadata__ must map strings to strings")
        self.meta: dict[str, str] = meta
        self.start = 8 + n
        self.tensors: dict[str, dict] = {}
        spans = []
        for name, info in header.items():
            try:
                dtype, shape, (a, b) = info["dtype"], info["shape"], info["data_offsets"]
            except (KeyError, TypeError, ValueError):
                raise FormatError(f"{name}: malformed tensor entry") from None
            if dtype not in DTYPE_SIZE:
                raise FormatError(f"{name}: unknown dtype {dtype}")
            if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
                raise FormatError(f"{name}: bad shape {shape}")
            count = 1
            for d in shape:
                count *= d
            if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b):
                raise FormatError(f"{name}: bad data_offsets {info['data_offsets']}")
            if b - a != count * DTYPE_SIZE[dtype]:
                raise FormatError(f"{name}: {b - a} bytes for {dtype} {shape}")
            self.tensors[name] = {"dtype": dtype, "shape": shape, "begin": a, "end": b}
            spans.append((a, b, name))
        spans.sort()
        pos = 0
        for a, b, name in spans:
            if a != pos:
                raise FormatError(f"{name}: data starts at {a}, expected {pos} (gap or overlap)")
            pos = b
        if self.start + pos != self.size:
            raise FormatError(f"tensor data ends at byte {self.start + pos}, file has {self.size} bytes")

    def names(self) -> list[str]:
        return [n for n in self.tensors if not n.startswith("__pad.")]

    def shape(self, name: str) -> tuple[str, list[int]] | None:
        t = self.tensors.get(name)
        return (t["dtype"], t["shape"]) if t else None

    def read(self, name: str) -> bytes:
        t = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(self.start + t["begin"])
            data = f.read(t["end"] - t["begin"])
        if len(data) != t["end"] - t["begin"]:
            raise FormatError(f"{name}: short read")
        return data

    def int64s(self, name: str) -> array.array:
        out = array.array("q")
        out.frombytes(self.read(name))
        if sys.byteorder != "little":
            out.byteswap()
        return out


def exlr8_rules(sf: SafeFile, fmt: str = "exlr8-v2") -> list[str]:
    """The FORMAT.md rules for every exlr8 safetensors file: aligned data start, aligned tensors, U8 pads, format and
    scheme metadata."""
    problems = []
    if sf.start % ALIGN:
        problems.append(f"data starts at byte {sf.start}, not a multiple of {ALIGN}")
    for name, t in sf.tensors.items():
        if name.startswith("__pad."):
            if t["dtype"] != "U8" or len(t["shape"]) != 1:
                problems.append(f"{name}: pad is {t['dtype']} {t['shape']}, expected 1-D U8")
        elif t["begin"] % ALIGN and t["end"] > t["begin"]:
            problems.append(f"{name}: data offset {t['begin']} is not a multiple of {ALIGN}")
    if sf.meta.get("format") != fmt:
        problems.append(f"metadata format is {sf.meta.get('format')!r}, expected {fmt!r}")
    if sf.meta.get("scheme") != "exlr8-e4m3-v2":
        problems.append(f"metadata scheme is {sf.meta.get('scheme')!r}, expected 'exlr8-e4m3-v2'")
    return problems


# ---------------------------------------------------------------------------------------------------------------------
# Geometry


@dataclass
class Geometry:
    hidden: int
    inter: int
    experts: int
    layers: int
    mtp: int
    first_moe: int
    vocab: int | None
    tied: bool

    @property
    def units(self) -> int:
        return self.inter // 128

    @property
    def all_layers(self) -> range:
        return range(self.layers + self.mtp)

    @property
    def moe_layers(self) -> range:
        return range(self.first_moe, self.layers + self.mtp)


# Absent model metadata means GLM-5.3 values (FORMAT.md, "Model metadata").
GLM53 = {"hidden_size": 6144, "moe_intermediate_size": 2048, "n_routed_experts": 256, "num_hidden_layers": 78,
         "first_moe_layer": 3, "num_nextn_predict_layers": 1}


def geometry(root: str, report: Report) -> Geometry:
    cfg = {}
    try:
        with open(os.path.join(root, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        pass  # reported by the support-file check
    text = cfg.get("text_config") or cfg
    meta = {}
    p = os.path.join(root, "dense", "exlr8-dense-000.safetensors")
    if os.path.isfile(p):
        try:
            meta = SafeFile(p).meta
        except (OSError, FormatError):
            pass

    def pick(cfg_key: str, meta_key: str) -> int:
        if cfg_key in text:
            return int(text[cfg_key])
        if meta_key in meta:
            return int(meta[meta_key])
        report.warn("support files", f"{cfg_key} not in config.json; using the GLM-5.3 value {GLM53[meta_key]}")
        return GLM53[meta_key]

    tied = bool(text.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    return Geometry(hidden=pick("hidden_size", "hidden_size"),
                    inter=pick("moe_intermediate_size", "moe_intermediate_size"),
                    experts=pick("n_routed_experts", "n_routed_experts"),
                    layers=pick("num_hidden_layers", "num_hidden_layers"),
                    mtp=int(text.get("num_nextn_predict_layers", 0)),
                    first_moe=pick("first_k_dense_replace", "first_moe_layer"),
                    vocab=int(text["vocab_size"]) if "vocab_size" in text else None,
                    tied=tied)


def meta_vs_geometry(meta: dict[str, str], geo: Geometry) -> list[str]:
    """Mismatches between a file's model metadata and config.json."""
    want = {"hidden_size": geo.hidden, "moe_intermediate_size": geo.inter, "n_routed_experts": geo.experts,
            "routed_experts": geo.experts, "num_hidden_layers": geo.layers, "first_moe_layer": geo.first_moe}
    out = []
    for k, v in want.items():
        if k in meta and str(meta[k]) != str(v):
            out.append(f"metadata {k}={meta[k]}, config.json gives {v}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Fileset


def walk(root: str) -> list[str]:
    """Every regular file under root as a relative POSIX path, following symlinks (HF snapshots and the NAS release
    are symlink trees)."""
    out = []
    for d, dirs, files in os.walk(root, followlinks=True):
        rel_d = os.path.relpath(d, root)
        if rel_d == ".":
            dirs[:] = [x for x in dirs if x not in IGNORED]
        for name in files:
            rel = os.path.normpath(os.path.join(rel_d, name)).replace(os.sep, "/")
            if rel in IGNORED:
                continue
            out.append(rel)
    return sorted(out)


def sidecar_rel(rel: str) -> str:
    """hashes/<path>.sha256, expert files without the experts/ prefix (FORMAT.md, "Release layout")."""
    if rel.startswith("experts/"):
        rel = rel[len("experts/"):]
    return rel + ".sha256"


def is_weight(rel: str) -> bool:
    return rel.endswith(".safetensors") and rel.split("/")[0] in ("dense", "experts", "index-slices")


class Validator:
    def __init__(self, root: str, full: bool, hashes: str | None, jobs: int, progress):
        self.root = os.path.abspath(root)
        self.full = full
        self.hash_root = os.path.abspath(hashes) if hashes else os.path.join(self.root, "hashes")
        self.external_hashes = hashes is not None
        self.jobs = jobs
        self.progress = progress
        self.r = Report()
        self.files: list[str] = []
        self.parsed: dict[str, SafeFile] = {}
        self.indexes: dict[int, SafeFile] = {}
        self.widths: dict[int, tuple[list[int], list[int]]] = {}  # layer -> (routed widths, shared widths)
        self.computed: dict[str, str] = {}  # rel -> sha256 (full mode)
        self.sidecars: dict[str, str] = {}  # rel -> sha256 from a well-formed sidecar whose size matches
        self.owned: dict[str, set[str]] = {}  # rel -> record, checksum and data tensor names
        self.manifests: dict[str, dict] = {}
        self.slice_ids: dict[str, str] = {}  # slice rel -> set_id
        self.slice_index_sha: dict[str, dict] = {}  # slice rel -> {layer: index sha256}

    def path(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def safe(self, rel: str, check: str = "safetensors") -> SafeFile | None:
        if rel in self.parsed:
            return self.parsed[rel]
        try:
            sf = SafeFile(self.path(rel))
        except (OSError, FormatError) as e:
            self.r.fail(check, f"{rel}: {e}")
            return None
        self.parsed[rel] = sf
        return sf

    # -- run ------------------------------------------------------------------------------------------------------

    def run(self) -> Report:
        self.files = walk(self.root)
        self.support()
        self.geo = geometry(self.root, self.r)
        self.layout()
        self.parse_all()
        self.dense()
        self.vocab()
        self.optional()
        self.experts()
        self.manifests_and_slices()
        self.hash_sidecars()
        self.set_ids()
        return self.r

    # -- support files --------------------------------------------------------------------------------------------

    def support(self) -> None:
        check = "support files"
        present = set(self.files)
        for name in SUPPORT_REQUIRED:
            if name not in present:
                self.r.fail(check, f"{name} is missing")
                continue
            p = self.path(name)
            if name.endswith(".json"):
                try:
                    with open(p) as f:
                        json.load(f)
                except (OSError, ValueError) as e:
                    self.r.fail(check, f"{name}: not valid JSON ({e})")
            elif os.path.getsize(p) == 0:
                self.r.fail(check, f"{name} is empty")
        for name in SUPPORT_OPTIONAL & present:
            if name.endswith(".json"):
                try:
                    with open(self.path(name)) as f:
                        json.load(f)
                except (OSError, ValueError) as e:
                    self.r.fail(check, f"{name}: not valid JSON ({e})")
        self.r.note(check, f"{sum(n in present for n in SUPPORT_REQUIRED)}/{len(SUPPORT_REQUIRED)} required present")

    # -- layout ---------------------------------------------------------------------------------------------------

    def layout(self) -> None:
        check = "layout"
        geo = self.geo
        present = set(self.files)
        for L in geo.all_layers:
            rel = f"dense/exlr8-dense-{L:03d}.safetensors"
            if rel not in present:
                self.r.fail(check, f"{rel} is missing (config.json gives layers 0-{geo.all_layers[-1]})")
        if "dense/exlr8-vocab.safetensors" not in present:
            self.r.fail(check, "dense/exlr8-vocab.safetensors is missing")
        for L in geo.moe_layers:
            rel = f"experts/exlr8-layer-{L:03d}.index.safetensors"
            if rel not in present:
                self.r.fail(check, f"{rel} is missing (MoE layers {geo.first_moe}-{geo.moe_layers[-1]})")
        unexpected = []
        for rel in self.files:
            top = rel.split("/")[0]
            if "/" not in rel:
                if rel not in SUPPORT_REQUIRED and rel not in SUPPORT_OPTIONAL:
                    self.r.warn(check, f"{rel}: unexpected top-level file")
                continue
            if top in OPTIONAL_DIRS:
                continue
            if top == "dense":
                m = RE_DENSE.fullmatch(rel)
                if m and int(m.group(1)) not in geo.all_layers:
                    unexpected.append(f"{rel}: layer {int(m.group(1))} is outside 0-{geo.all_layers[-1]}")
                elif not (m or rel == "dense/exlr8-vocab.safetensors" or RE_EXTRA.fullmatch(rel)
                          or RE_SPLIT.fullmatch(rel) or rel in OPTIONAL_PLAIN):
                    unexpected.append(f"{rel}: not a dense, vocab, extra, split or known optional file")
            elif top == "experts":
                m = RE_INDEX.fullmatch(rel) or RE_TRELLIS.fullmatch(rel)
                if not m:
                    unexpected.append(f"{rel}: not an expert index or trellis file")
                elif int(m.group(1)) not in geo.moe_layers:
                    unexpected.append(f"{rel}: layer {int(m.group(1))} is not a MoE layer")
            elif top == "manifests":
                if not RE_MANIFEST.fullmatch(rel):
                    unexpected.append(f"{rel}: not a manifest")
            elif top == "index-slices":
                if not RE_SLICE.fullmatch(rel):
                    unexpected.append(f"{rel}: not an index slice")
            elif top == "hashes":
                pass  # hash_sidecars
            else:
                self.r.warn(check, f"{rel}: unexpected directory {top}/")
        for u in unexpected:
            self.r.fail(check, u)
        self.r.note(check, f"{len(self.files)} files; layers 0-{geo.all_layers[-1]}, MoE {geo.first_moe}-"
                           f"{geo.moe_layers[-1]}, H={geo.hidden} I={geo.inter} E={geo.experts}")

    # -- safetensors ----------------------------------------------------------------------------------------------

    def parse_all(self) -> None:
        check = "safetensors"
        n = 0
        for rel in self.files:
            if not rel.endswith(".safetensors"):
                continue
            sf = self.safe(rel)
            if sf is None:
                continue
            n += 1
            if rel in OPTIONAL_PLAIN or rel.split("/")[0] in OPTIONAL_DIRS:
                continue
            if rel.startswith("index-slices/"):
                # A loader reads a slice whole. The published slices have no alignment and no scheme key.
                if sf.meta.get("format") != "exlr8-v2-index-slice":
                    self.r.fail(check, f"{rel}: metadata format is {sf.meta.get('format')!r}, expected "
                                       "'exlr8-v2-index-slice'")
                continue
            for p in exlr8_rules(sf):
                self.r.fail(check, f"{rel}: {p}")
        self.r.note(check, f"{n} files parsed")

    # -- dense ----------------------------------------------------------------------------------------------------

    def records(self, rel: str, sf: SafeFile, split: bool = False) -> dict[str, tuple[int, ...]]:
        """Check every record in a dense, vocab, extra or split file. Returns {matrix: record}."""
        check = "dense records"
        names = set(sf.names())
        recs = {}
        owned = set()
        for rn in sorted(n for n in names if n.endswith(".record")):
            m = rn[:-len(".record")]
            if sf.shape(rn) != ("I64", [8]):
                self.r.fail(check, f"{rel}: {rn} is {sf.shape(rn)}, expected I64 [8]")
                continue
            rec = struct.unpack("<8q", sf.read(rn))
            recs[m] = rec
            owned.add(rn)
            fmt, out, inp, K, side, chan, rot, solve = rec
            where = f"{rel}: {m}"
            if fmt not in DATA:
                self.r.fail(check, f"{where}: record format {fmt} is not 1-7")
                continue
            if out <= 0 or inp <= 0:
                self.r.fail(check, f"{where}: record shape [{out}, {inp}]")
                continue
            if (fmt == TRELLIS) != (K > 0) or not 0 <= K <= 8:
                self.r.fail(check, f"{where}: K={K} with format {fmt}")
            if side not in (0, 1, 2):
                self.r.fail(check, f"{where}: sharded side {side} is not 0, 1 or 2")
                continue
            if side == 0 and chan != 0:
                self.r.fail(check, f"{where}: replicated but {chan} channels per sharding unit")
            if side != 0 and (chan <= 0 or (out if side == 1 else inp) % chan):
                self.r.fail(check, f"{where}: {chan} channels per sharding unit for side {side} of [{out}, {inp}]")
            if rot < 0 or (rot and fmt != TRELLIS):
                self.r.fail(check, f"{where}: shared rotation {rot} with format {fmt}")
            if solve not in (0, 1):
                self.r.fail(check, f"{where}: solve {solve} is not 0 or 1")
            want = expected_data(fmt, out, inp, K, side)
            if want is None:
                self.r.fail(check, f"{where}: [{out}, {inp}] does not fit format {fmt} (dimension not divisible)")
                continue
            for suffix, (dtype, shape) in want.items():
                tn = f"{m}.{suffix}"
                owned.add(tn)
                got = sf.shape(tn)
                if got is None:
                    self.r.fail(check, f"{where}: data tensor {suffix} is missing")
                elif got[0] != dtype:
                    self.r.fail(check, f"{where}: {suffix} is {got[0]}, expected {dtype}")
                elif split:
                    if got[1][1:] != shape[1:]:
                        self.r.fail(check, f"{where}: {suffix} is {got[1]}, expected [rows]+{shape[1:]}")
                elif got[1] != shape:
                    self.r.fail(check, f"{where}: {suffix} is {got[1]}, expected {shape}")
            blocks = -(-(inp if side == 2 else out) // 128)
            cs = sf.shape(f"{m}.checksum")
            owned.add(f"{m}.checksum")
            if cs is None:
                if not split:
                    self.r.fail(check, f"{where}: checksum is missing")
            elif not split and cs != ("U64", [blocks]):
                self.r.fail(check, f"{where}: checksum is {cs}, expected U64 [{blocks}]")
        for n in sorted(names - owned):
            base, _, suffix = n.rpartition(".")
            if base in recs and suffix in DATA_SUFFIXES:
                self.r.fail(check, f"{rel}: {n} is not a data tensor of {base}'s format {recs[base][0]}")
            elif suffix == "checksum" and base not in recs:
                self.r.fail(check, f"{rel}: {n} has no {base}.record")
        self.owned[rel] = owned
        return recs

    def dense(self) -> None:
        check = "dense records"
        geo = self.geo
        nrec = nfiles = 0
        for rel in self.files:
            m = RE_DENSE.fullmatch(rel)
            if not m:
                continue
            sf = self.parsed.get(rel)
            if sf is None:
                continue
            nfiles += 1
            L = int(m.group(1))
            if sf.meta.get("layer") != str(L):
                self.r.fail(check, f"{rel}: metadata layer is {sf.meta.get('layer')!r}, expected {L}")
            if sf.meta.get("contents", "dense") != "dense":
                self.r.fail(check, f"{rel}: metadata contents is {sf.meta.get('contents')!r}")
            for p in meta_vs_geometry(sf.meta, geo):
                self.r.fail(check, f"{rel}: {p}")
            recs = self.records(rel, sf)
            nrec += len(recs)
            self.source_maps(rel, sf, recs)
            if not recs:
                self.r.fail(check, f"{rel}: no matrices")
        for rel in self.files:
            if RE_EXTRA.fullmatch(rel) and rel in self.parsed:
                nrec += len(self.records(rel, self.parsed[rel]))
        self.splits()
        self.r.note(check, f"{nrec} matrices in {nfiles} layer files")

    def source_maps(self, rel: str, sf: SafeFile, recs: dict) -> None:
        """source_names / source_shapes metadata, when present, must cover the stored matrices and agree with them."""
        check = "dense records"
        for key in ("source_names", "source_shapes"):
            if key not in sf.meta:
                continue
            try:
                value = json.loads(sf.meta[key])
                assert isinstance(value, dict)
            except (ValueError, AssertionError):
                self.r.fail(check, f"{rel}: metadata {key} is not a JSON object")
                continue
            if key == "source_names":
                for m in recs:
                    if m not in value:
                        self.r.fail(check, f"{rel}: {m} has no entry in source_names")
            else:
                for m, rec in recs.items():
                    if m in value and list(value[m]) != [rec[1], rec[2]]:
                        self.r.fail(check, f"{rel}: {m} record is [{rec[1]}, {rec[2]}], source_shapes {value[m]}")
                for n in sf.names():
                    if n in value and n not in self.owned.get(rel, ()) and list(value[n]) != sf.shape(n)[1]:
                        self.r.fail(check, f"{rel}: {n} is {sf.shape(n)[1]}, source_shapes {value[n]}")

    def splits(self) -> None:
        check = "dense records"
        groups: dict[str, dict[int, str]] = {}
        for rel in self.files:
            m = RE_SPLIT.fullmatch(rel)
            if m:
                groups.setdefault(m.group(1), {})[int(m.group(2))] = rel
        for name, parts in sorted(groups.items()):
            first = self.parsed.get(parts[min(parts)])
            if first is None:
                continue
            try:
                count = int(first.meta.get("parts", ""))
            except ValueError:
                self.r.fail(check, f"split {name}: metadata parts is {first.meta.get('parts')!r}")
                continue
            if sorted(parts) != list(range(count)):
                self.r.fail(check, f"split {name}: parts {sorted(parts)}, metadata says {count}")
            ranges, rec0 = [], None
            totals: dict[str, int] = {}
            for p in sorted(parts):
                rel = parts[p]
                sf = self.parsed.get(rel)
                if sf is None:
                    continue
                recs = self.records(rel, sf, split=True)
                if len(recs) != 1:
                    self.r.fail(check, f"{rel}: {len(recs)} records, a split part holds one")
                    continue
                (m, rec), = recs.items()
                rec0 = rec0 or (m, rec)
                if (m, rec) != rec0:
                    self.r.fail(check, f"{rel}: record {m} {rec} differs from part 0's {rec0}")
                try:
                    r0, r1 = (int(x) for x in sf.meta["rows"].split("-"))
                except (KeyError, ValueError):
                    self.r.fail(check, f"{rel}: metadata rows is {sf.meta.get('rows')!r}, expected r0-r1")
                    continue
                ranges.append((r0, r1, rel))
                for suffix in DATA.get(rec[0], ()):
                    got = sf.shape(f"{m}.{suffix}")
                    if got and len(got[1]) >= 2:
                        totals[suffix] = totals.get(suffix, 0) + got[1][0]
            pos = 0
            for r0, r1, rel in sorted(ranges):
                if r0 != pos or r1 <= r0:
                    self.r.fail(check, f"{rel}: rows {r0}-{r1}, expected to start at {pos}")
                pos = r1
            if rec0:
                m, rec = rec0
                want = expected_data(rec[0], rec[1], rec[2], rec[3], rec[4]) or {}
                if ranges and pos != want.get(DATA[rec[0]][0], ("", [pos]))[1][0]:
                    self.r.fail(check, f"split {name}: rows cover 0-{pos} of {want[DATA[rec[0]][0]][1][0]}")
                for suffix, total in totals.items():
                    if suffix in want and total != want[suffix][1][0]:
                        self.r.fail(check, f"split {name}: {suffix} parts add to {total} rows, expected "
                                           f"{want[suffix][1][0]}")

    def vocab(self) -> None:
        check = "vocab"
        rel = "dense/exlr8-vocab.safetensors"
        sf = self.parsed.get(rel)
        if sf is None:
            if rel in self.files:
                self.r.fail(check, f"{rel} did not parse")
            else:
                self.r.fail(check, f"{rel} is missing")
            return
        if sf.meta.get("contents", "vocab") != "vocab":
            self.r.fail(check, f"{rel}: metadata contents is {sf.meta.get('contents')!r}")
        recs = self.records(rel, sf)
        need = ["embed_tokens"] + ([] if self.geo.tied else ["lm_head"])
        for m in need:
            if m not in recs:
                self.r.fail(check, f"{rel}: {m} is missing")
            elif recs[m][2] != self.geo.hidden or (self.geo.vocab and recs[m][1] != self.geo.vocab):
                self.r.fail(check, f"{rel}: {m} is [{recs[m][1]}, {recs[m][2]}], config.json gives "
                                   f"[{self.geo.vocab}, {self.geo.hidden}]")
        norm = sf.shape("norm.weight")
        if norm is None:
            self.r.fail(check, f"{rel}: norm.weight is missing")
        elif norm[1] != [self.geo.hidden]:
            self.r.fail(check, f"{rel}: norm.weight is {norm[1]}, expected [{self.geo.hidden}]")
        self.r.note(check, ", ".join(sorted(recs)) + " + norm.weight")

    def optional(self) -> None:
        check = "optional members"
        found = [rel for rel in self.files if rel in OPTIONAL_PLAIN]
        for rel in found:
            sf = self.parsed.get(rel)
            if sf is None:
                continue  # parse failure reported under safetensors
            names = sf.names()
            if not names:
                self.r.fail(check, f"{rel}: no tensors")
            if "vision" in rel:
                bad = [n for n in names if not n.startswith("model.visual.")]
                if bad:
                    self.r.fail(check, f"{rel}: {len(bad)} tensors outside model.visual.*, e.g. {bad[0]}")
            if any(n.endswith(".record") for n in names):
                self.r.fail(check, f"{rel}: holds exlr8 records; a plain member must hold plain tensors")
            dtypes = sorted({sf.tensors[n]["dtype"] for n in names})
            self.r.note(check, f"{rel}: {len(names)} tensors, {'/'.join(dtypes)}")
        if not found:
            self.r.note(check, "none")

    # -- experts --------------------------------------------------------------------------------------------------

    def experts(self) -> None:
        geo = self.geo
        H, I, E, U = geo.hidden, geo.inter, geo.experts, geo.units
        no_errors = []
        n_units = 0
        for L in geo.moe_layers:
            rel = f"experts/exlr8-layer-{L:03d}.index.safetensors"
            sf = self.parsed.get(rel)
            if sf is None:
                continue
            check = "expert indexes"
            fail = lambda msg: self.r.fail(check, f"{rel}: {msg}")  # noqa: E731
            if sf.meta.get("layer") != str(L):
                fail(f"metadata layer is {sf.meta.get('layer')!r}")
            if sf.meta.get("contents", "index") != "index":
                fail(f"metadata contents is {sf.meta.get('contents')!r}")
            for p in meta_vs_geometry(sf.meta, geo):
                fail(p)
            if sf.meta.get("experts") != f"0-{E - 1}":
                fail(f"metadata experts is {sf.meta.get('experts')!r}, expected '0-{E - 1}'")
            try:
                widths = parse_widths(sf.meta.get("widths", ""))
                shared = parse_widths(sf.meta.get("shared_widths", ""))
            except ValueError as e:
                fail(str(e))
                continue
            if not widths:
                fail("no routed widths")
            stored = sorted(set(widths) | set(shared))
            self.widths[L] = (widths, shared)
            self.indexes[L] = sf
            want_files = [f"exlr8-layer-{L:03d}.k{K}.safetensors" for K in stored]
            if sf.meta.get("trellis_files", "").split(",") != want_files:
                fail(f"metadata trellis_files is {sf.meta.get('trellis_files')!r}, expected {','.join(want_files)}")
            want = {"signs.in": ("I8", [H]), "signs.out": ("I8", [H])}
            errs = {}
            for pre, n, ks in (("", E, widths), ("shared.", 1, shared)):
                for K in ks:
                    p = f"{pre}k{K}."
                    want.update({p + "index": ("I64", [n, 3, U, 2]), p + "checksum": ("U64", [n, 3, U]),
                                 p + "gate.su_block": ("F16", [n, H // 128]), p + "up.su_block": ("F16", [n, H // 128]),
                                 p + "down.sv_block": ("F16", [n, H // 128]), p + "gate.svh": ("F16", [n, I]),
                                 p + "up.svh": ("F16", [n, I]), p + "down.suh": ("F16", [n, I])})
                    errs.update({p + "err_gate": ("F32", [n, U]), p + "err_up": ("F32", [n, U]),
                                 p + "err_down": ("F32", [n])})
                if ks:
                    errs[pre + "heldout_tokens"] = ("I32", [n])
            names = set(sf.names())
            for name, spec in want.items():
                got = sf.shape(name)
                if got is None:
                    fail(f"{name} is missing")
                elif list(got) != list(spec):
                    fail(f"{name} is {got[0]} {got[1]}, expected {spec[0]} {spec[1]}")
            for name in sorted(names - set(want) - set(errs)):
                fail(f"unexpected tensor {name}")
            present = [n for n in errs if n in names]
            if not present:
                no_errors.append(L)
            elif len(present) != len(errs):
                self.r.fail("error tables", f"{rel}: partial error tables, missing "
                                            f"{', '.join(sorted(set(errs) - set(present)))}")
            for name in present:
                if list(sf.shape(name)) != list(errs[name]):
                    self.r.fail("error tables", f"{rel}: {name} is {sf.shape(name)}, expected {errs[name]}")
            n_units += self.trellis(L, sf, widths, shared)
        if no_errors:
            self.r.warn("error tables", f"{len(no_errors)} of {len(geo.moe_layers)} index files have no per-unit error "
                                        f"tables (err_gate/err_up/err_down): layers {ranges_text(no_errors)}")
        else:
            self.r.note("error tables", f"present in all {len(self.indexes)} index files")
        counts: dict[str, int] = {}
        for w, _ in self.widths.values():
            counts[",".join(map(str, w))] = counts.get(",".join(map(str, w)), 0) + 1
        self.r.note("expert indexes", f"{len(self.indexes)} index files; routed widths "
                    + ", ".join(f"K{k} x{n}" for k, n in sorted(counts.items())))
        self.r.note("expert offsets", f"{n_units} unit offsets checked")
        trellis_rels = [f for f in self.files if RE_TRELLIS.fullmatch(f)]
        self.r.note("expert trellis files", f"{len(trellis_rels)} files")
        for rel in trellis_rels:
            L, K = (int(x) for x in RE_TRELLIS.fullmatch(rel).groups())
            if L in self.widths and K not in set(self.widths[L][0]) | set(self.widths[L][1]):
                self.r.fail("expert trellis files", f"{rel}: K{K} is not in the index's widths or shared_widths")

    def trellis(self, L: int, index: SafeFile, widths: list[int], shared: list[int]) -> int:
        """Check each kK file of a layer and every unit offset the index gives for it. Returns units checked."""
        check = "expert trellis files"
        geo = self.geo
        E, U, H = geo.experts, geo.units, geo.hidden
        checked = 0
        for K in sorted(set(widths) | set(shared)):
            rel = f"experts/exlr8-layer-{L:03d}.k{K}.safetensors"
            if rel not in self.files:
                self.r.fail(check, f"{rel} is missing (index stores K{K})")
                continue
            sf = self.parsed.get(rel)
            if sf is None:
                continue
            fail = lambda msg: self.r.fail(check, f"{rel}: {msg}")  # noqa: E731
            for key, val in (("layer", str(L)), ("k", str(K)),
                             ("index_file", f"exlr8-layer-{L:03d}.index.safetensors")):
                if sf.meta.get(key) != val:
                    fail(f"metadata {key} is {sf.meta.get(key)!r}, expected {val!r}")
            for key, val in (("gate_up_order", "grouped"), ("down_order", "item128")):
                if key in sf.meta and sf.meta[key] != val:
                    fail(f"metadata {key} is {sf.meta[key]!r}, expected {val!r}")
            pad = sf.meta.get("pad_halves", index.meta.get("pad_halves", "false"))
            if pad not in ("true", "false"):
                fail(f"metadata pad_halves is {pad!r}")
                continue
            B = H * 16 * K
            S = -(-B // ALIGN) * ALIGN if pad == "true" else B
            # FORMAT.md "Codebook": mcg keeps the bare names, any other codebook suffixes them and names itself in
            # the file's metadata, as its first word followed somewhere by its multiplier.
            found = {(n.split("trellis.", 1)[1].split(".", 1) + ["mcg"])[1] for n in sf.names() if "trellis." in n}
            if len(found) > 1:
                fail(f"trellis tensors mix codebooks {sorted(found)}")
                continue
            cb = found.pop() if found else "mcg"
            if cb not in CODEBOOKS:
                fail(f"unknown codebook {cb!r}")
                continue
            named = sf.meta.get("codebook")
            if (named is not None or cb != "mcg") and (
                    (named or "").split(" ", 1)[0] != cb or CODEBOOKS[cb].lower() not in (named or "").lower()):
                fail(f"tensor names select {cb} but metadata codebook is {named!r}")
            sfx = "" if cb == "mcg" else f".{cb}"
            want = {}
            if K in widths:
                want.update({f"trellis.{p}{sfx}": ("U8", [E * U * S]) for p in PROJS})
            if K in shared:
                want.update({f"shared.trellis.{p}{sfx}": ("U8", [U * S]) for p in PROJS})
            ok = True
            for name, spec in want.items():
                got = sf.shape(name)
                if got is None:
                    fail(f"{name} is missing")
                    ok = False
                elif list(got) != list(spec):
                    fail(f"{name} is {got[0]} {got[1]}, expected {spec[0]} {spec[1]}")
                    ok = False
            for name in sorted(set(sf.names()) - set(want)):
                fail(f"unexpected tensor {name}")
            if not ok:
                continue
            for pre, n, present in (("", E, K in widths), ("shared.", 1, K in shared)):
                if not present or index.shape(f"{pre}k{K}.index") != ("I64", [n, 3, U, 2]):
                    continue
                checked += self.offsets(L, K, pre, n, index, sf, S, B, sfx)
        return checked

    def offsets(self, L: int, K: int, pre: str, n: int, index: SafeFile, sf: SafeFile, S: int, B: int,
                sfx: str = "") -> int:
        """index[e, p, u] = (absolute byte offset of unit u in the kK file, B); unit u of expert e at (U e + u) S."""
        U = self.geo.units
        table = index.int64s(f"{pre}k{K}.index")
        bad = 0
        first = None
        for e in range(n):
            for p, proj in enumerate(PROJS):
                base = sf.start + sf.tensors[f"{pre}trellis.{proj}{sfx}"]["begin"]
                for u in range(U):
                    i = ((e * 3 + p) * U + u) * 2
                    off, nb = table[i], table[i + 1]
                    want = base + (U * e + u) * S
                    if off != want or nb != B:
                        bad += 1
                        if first is None:
                            first = f"expert {e} {proj} unit {u}: ({off}, {nb}), expected ({want}, {B})"
        if bad:
            who = "shared expert" if pre else "routed experts"
            self.r.fail("expert offsets", f"layer {L} K{K} {who}: {bad} of {n * 3 * U} index entries wrong; first "
                                          f"{first}")
        return n * 3 * U

    # -- manifests and index slices -------------------------------------------------------------------------------

    def manifests_and_slices(self) -> None:
        geo = self.geo
        rels = [f for f in self.files if RE_MANIFEST.fullmatch(f)]
        check = "manifests"
        dense_needed = [f"dense/exlr8-dense-{L:03d}.safetensors" for L in geo.all_layers] + \
                       ["dense/exlr8-vocab.safetensors"]
        for rel in rels:
            name = RE_MANIFEST.fullmatch(rel).group(1)
            fail = lambda msg: self.r.fail(check, f"{rel}: {msg}")  # noqa: E731
            try:
                with open(self.path(rel)) as f:
                    m = json.load(f)
                assert isinstance(m, dict)
            except (OSError, ValueError, AssertionError) as e:
                fail(f"not a JSON object ({e})")
                continue
            self.manifests[rel] = m
            if m.get("format") != "exlr8-v2-manifest":
                fail(f"format is {m.get('format')!r}, expected 'exlr8-v2-manifest'")
            if m.get("name") != name:
                fail(f"name is {m.get('name')!r}, the file name says {name!r}")
            layers = m.get("layers")
            if not isinstance(layers, dict):
                fail("no layers object")
                continue
            keys = {int(k) for k in layers if str(k).isdigit()}
            if set(map(str, keys)) != set(layers):
                fail(f"non-numeric layer keys {sorted(set(layers) - set(map(str, keys)))}")
            for L in sorted(set(geo.moe_layers) - keys):
                fail(f"layer {L} is missing")
            for L in sorted(keys - set(geo.moe_layers)):
                fail(f"layer {L} is not a MoE layer")
            needed = set(dense_needed)
            for L in sorted(keys & set(geo.moe_layers)):
                needed.add(f"experts/exlr8-layer-{L:03d}.index.safetensors")
                try:
                    used, used_shared = manifest_layer_widths(layers[str(L)], geo.experts, geo.units)
                except ValueError as e:
                    fail(f"layer {L}: {e}")
                    continue
                w, ws = self.widths.get(L, (None, None))
                for K in sorted(used | used_shared):
                    needed.add(f"experts/exlr8-layer-{L:03d}.k{K}.safetensors")
                if w is not None:
                    if used - set(w):
                        fail(f"layer {L}: routed widths {sorted(used - set(w))} are not stored (index widths {w})")
                    if used_shared - set(ws):
                        fail(f"layer {L}: shared widths {sorted(used_shared - set(ws))} are not stored "
                             f"(index shared_widths {ws})")
            missing = sorted(needed - set(self.files))
            if missing:
                fail(f"{len(missing)} referenced weight files are missing: {', '.join(missing[:5])}"
                     + (" ..." if len(missing) > 5 else ""))
            sid = m.get("set_id")
            if not (isinstance(sid, str) and re.fullmatch(r"[0-9a-f]{64}", sid)):
                self.r.fail("weight set checksums", f"{rel}: set_id is {sid!r}, expected a sha256 hex digest")
        self.r.note(check, f"{len(rels)} manifests" if rels else "none shipped")
        self.slices()

    def slices(self) -> None:
        check = "index slices"
        geo = self.geo
        E, U, H = geo.experts, geo.units, geo.hidden
        rels = [f for f in self.files if RE_SLICE.fullmatch(f)]
        groups: dict[tuple[int, int], set[int]] = {}
        for rel in rels:
            N, K, R = (int(x) for x in RE_SLICE.fullmatch(rel).groups())
            groups.setdefault((N, K), set()).add(R)
            sf = self.parsed.get(rel)
            if sf is None:
                continue
            fail = lambda msg: self.r.fail(check, f"{rel}: {msg}")  # noqa: E731
            for key, val in (("tp", N), ("rank", R), ("k", K)):
                if sf.meta.get(key) != str(val):
                    fail(f"metadata {key} is {sf.meta.get(key)!r}, expected {val}")
            sid = sf.meta.get("set_id")
            if not (isinstance(sid, str) and re.fullmatch(r"[0-9a-f]{64}", sid)):
                self.r.fail("weight set checksums", f"{rel}: set_id is {sid!r}, expected a sha256 hex digest")
            else:
                self.slice_ids[rel] = sid
            names = set(sf.names())
            seen = set()
            for L in geo.moe_layers:
                w, ws = self.widths.get(L, ([], []))
                owned = sum(1 for e in range(E) if K in w for u in range(U) if (U * e + u + L) % N == R) + \
                    sum(1 for u in range(U) if K in ws and (U * E + u + L) % N == R)
                p = f"L{L:03d}."
                if not (K in w or K in ws):
                    if any(n.startswith(p) for n in names):
                        fail(f"has layer {L}, which stores no K{K}")
                    continue
                want = {p + "units": ("I16", [owned, 2]), p + "offset": ("I64", [owned, 3]),
                        p + "checksum": ("U64", [owned, 3]), p + "chan": ("F16", [owned, 3, 128]),
                        p + "block": ("F16", [E + (K in ws), 3, H // 128]), p + "signs": ("I8", [2, H])}
                for name, spec in want.items():
                    seen.add(name)
                    got = sf.shape(name)
                    if got is None:
                        fail(f"{name} is missing")
                    elif list(got) != list(spec):
                        fail(f"{name} is {got[0]} {got[1]}, expected {spec[0]} {spec[1]}")
            for name in sorted(names - seen):
                fail(f"unexpected tensor {name}")
            try:
                idx = json.loads(sf.meta.get("index_sha256", "{}"))
                assert isinstance(idx, dict)
            except (ValueError, AssertionError):
                fail("metadata index_sha256 is not a JSON object")
                idx = {}
            self.slice_index_sha[rel] = idx
        for (N, K), ranks in sorted(groups.items()):
            if ranks != set(range(N)):
                self.r.fail(check, f"index-slices/tp{N}/k{K}: ranks {sorted(ranks)}, expected 0-{N - 1}")
        self.r.note(check, f"{len(rels)} slices in {len(groups)} (tp, k) groups" if rels else "none shipped")

    # -- hashes ---------------------------------------------------------------------------------------------------

    def hash_files(self, rels: list[str]) -> None:
        """SHA-256 of each file, streamed, a few files at a time."""
        total = sum(os.path.getsize(self.path(r)) for r in rels)
        done = [0, 0]
        t0 = time.monotonic()

        def one(rel: str) -> tuple[str, str]:
            h = hashlib.sha256()
            with open(self.path(rel), "rb") as f:
                while True:
                    b = f.read(16 << 20)
                    if not b:
                        break
                    h.update(b)
            return rel, h.hexdigest()

        with concurrent.futures.ThreadPoolExecutor(self.jobs) as pool:
            for rel, digest in pool.map(one, rels):
                self.computed[rel] = digest
                done[0] += 1
                done[1] += os.path.getsize(self.path(rel))
                if self.progress:
                    rate = done[1] / max(time.monotonic() - t0, 1e-6) / 1e9
                    print(f"[hash] {done[0]}/{len(rels)} {done[1] / 1e9:.1f}/{total / 1e9:.1f} GB "
                          f"{rate:.2f} GB/s {rel}", file=self.progress, flush=True)

    def hash_sidecars(self) -> None:
        check = "hash sidecars"
        weights = [f for f in self.files if is_weight(f)]
        if self.full:
            self.hash_files(weights)
        missing = []
        verified = 0
        for rel in weights:
            side = os.path.join(self.hash_root, sidecar_rel(rel))
            shown = os.path.relpath(side, self.root) if not self.external_hashes else side
            if not os.path.isfile(side):
                missing.append(rel)
                continue
            with open(side) as f:
                text = f.read()
            m = RE_SIDECAR.fullmatch(text)
            if not m:
                self.r.fail(check, f"{shown}: expected '<sha256> <bytes>', got {text[:80]!r}")
                continue
            digest, size = m.group(1), int(m.group(2))
            actual = os.path.getsize(self.path(rel))
            if size != actual:
                self.r.fail(check, f"{rel}: sidecar says {size} bytes, file has {actual}")
                continue
            self.sidecars[rel] = digest
            if rel in self.computed:
                if self.computed[rel] != digest:
                    self.r.fail(check, f"{rel}: sha256 {self.computed[rel]} does not match sidecar {digest}")
                else:
                    verified += 1
        for rel in missing:
            self.r.warn(check, f"{rel}: no sha256 sidecar at {sidecar_rel(rel)} under {self.hash_root}")
        # Sidecars with no file behind them.
        if os.path.isdir(self.hash_root):
            for d, _, files in os.walk(self.hash_root, followlinks=True):
                for name in files:
                    side = os.path.relpath(os.path.join(d, name), self.hash_root).replace(os.sep, "/")
                    if not side.endswith(".sha256"):
                        self.r.fail(check, f"hashes/{side}: not a .sha256 sidecar")
                        continue
                    target = side[:-len(".sha256")]
                    if target.startswith("exlr8-layer-"):
                        target = "experts/" + target
                    if target not in self.files:
                        self.r.fail(check, f"hashes/{side}: {target} is not in the fileset")
        where = self.hash_root if self.external_hashes else "hashes/"
        note = f"{len(weights) - len(missing)}/{len(weights)} weight files have a sidecar in {where}"
        note += f"; {verified} sha256 verified" if self.full else "; sizes checked, sha256 not computed (fast mode)"
        self.r.note(check, note)

    def digest(self, rel: str) -> str | None:
        return self.computed.get(rel) or (None if self.full else self.sidecars.get(rel))

    def set_ids(self) -> None:
        """set_id = sha256 of "<path> <bytes> <sha256>\\n" over experts/* and dense/*, sorted by path."""
        check = "weight set checksums"
        claims = {rel: m.get("set_id") for rel, m in self.manifests.items()}
        claims.update(self.slice_ids)
        claims = {k: v for k, v in claims.items() if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)}
        members = [f for f in self.files if f.split("/")[0] in ("dense", "experts")]
        # Slices also name each index file's sha256.
        for rel, idx in self.slice_index_sha.items():
            for layer, want in idx.items():
                try:
                    irel = f"experts/exlr8-layer-{int(layer):03d}.index.safetensors"
                except ValueError:
                    self.r.fail(check, f"{rel}: index_sha256 key {layer!r} is not a layer")
                    continue
                got = self.digest(irel)
                if irel not in self.files:
                    self.r.fail(check, f"{rel}: index_sha256 names {irel}, which is missing")
                elif got and got != want:
                    self.r.fail(check, f"{rel}: index_sha256 for layer {layer} is {want}, {irel} hashes to {got}")
        if not self.manifests and not self.slice_ids:
            self.r.warn(check, "no manifests or index slices: the release declares no set_id, so per-file sidecars "
                               "are its only checksums")
            return
        if not claims:
            return
        unknown = [f for f in members if self.digest(f) is None]
        if unknown:
            msg = f"cannot compute set_id: {len(unknown)} files have no hash"
            if self.full:
                self.r.fail(check, msg)
            else:
                self.r.warn(check, msg + " in a sidecar; run --full to hash them")
            return
        sid = set_id(self.root, members, self.digest)
        without = [f for f in members if f not in OPTIONAL_PLAIN]
        sid_core = set_id(self.root, without, self.digest) if len(without) != len(members) else sid
        src = "computed hashes" if self.full else "sidecar hashes"
        for rel, want in sorted(claims.items()):
            if want == sid:
                continue
            if want == sid_core:
                self.r.warn(check, f"{rel}: set_id matches only without the optional "
                                   f"{', '.join(sorted(set(members) - set(without)))}; it predates that member")
            else:
                self.r.fail(check, f"{rel}: set_id {want} does not match {sid} ({src})")
        self.r.note(check, f"{len(claims)} set_id values checked against {src} of {len(members)} files")


def set_id(root: str, members: list[str], digest) -> str:
    h = hashlib.sha256()
    for rel in sorted(members):
        h.update(f"{rel} {os.path.getsize(os.path.join(root, rel))} {digest(rel)}\n".encode())
    return h.hexdigest()


def expected_data(fmt: int, out: int, inp: int, K: int, side: int) -> dict[str, tuple[str, list[int]]] | None:
    """Data tensors of one dense record (FORMAT.md table). None when the shape does not fit the format."""
    if fmt == TRELLIS:
        if out % 128 or inp % 128:
            return None
        entries, nbytes = (inp // 128, 16 * K * out) if side == 2 else (out // 128, 16 * K * inp)
        return {"trellis": ("U8", [entries, nbytes]), "suh": ("F16", [inp]), "svh": ("F16", [out])}
    if fmt == INT4:
        if side == 2:
            if inp % 128:
                return None
            return {"q4": ("U8", [inp // 128, out, 64]), "s4": ("F16", [inp // 128, out, 4])}
        if inp % 32:
            return None
        return {"q4": ("U8", [out, inp // 2]), "s4": ("F16", [out, inp // 32])}
    if fmt == MXFP8:
        if inp % 32:
            return None
        return {"fp8": ("U8", [out, inp]), "e8m0": ("U8", [out, inp // 32])}
    if fmt == FP8_BLOCK:
        return {"fp8": ("U8", [out, inp]), "scale": ("F32", [-(-out // 128), -(-inp // 128)])}
    if fmt == FP8_ROW:
        return {"fp8": ("U8", [out, inp]), "scale": ("F32", [out])}
    if fmt == BF16:
        return {"bf16": ("BF16", [out, inp])}
    return {"fp32": ("F32", [out, inp])}


def parse_widths(text: str) -> list[int]:
    if not text.strip():
        return []
    try:
        ks = [int(x) for x in text.split(",")]
    except ValueError:
        raise ValueError(f"widths {text!r} is not a comma list of integers") from None
    if any(not 1 <= k <= 8 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"widths {text!r}: each K must be 1-8 and listed once")
    return ks


def manifest_layer_widths(entry, E: int, U: int) -> tuple[set[int], set[int]]:
    """Validate one manifest layer's shapes; return (routed widths used, shared widths used)."""
    if not isinstance(entry, dict):
        raise ValueError("not an object")
    used, used_shared = set(), set()
    for proj in ("gate", "up"):
        rows = entry.get(proj)
        if not (isinstance(rows, list) and len(rows) == E and all(isinstance(r, list) and len(r) == U for r in rows)):
            raise ValueError(f"{proj} is not [{E}][{U}]")
        used.update(int(k) for r in rows for k in r)
    down = entry.get("down")
    if not (isinstance(down, list) and len(down) == E):
        raise ValueError(f"down is not [{E}]")
    used.update(int(k) for k in down)
    sh = entry.get("shared")
    if not isinstance(sh, dict):
        raise ValueError("shared is missing")
    for proj in ("gate", "up"):
        if not (isinstance(sh.get(proj), list) and len(sh[proj]) == U):
            raise ValueError(f"shared.{proj} is not [{U}]")
        used_shared.update(int(k) for k in sh[proj])
    if not isinstance(sh.get("down"), int):
        raise ValueError("shared.down is not an integer")
    used_shared.add(sh["down"])
    return used, used_shared


def ranges_text(xs: list[int]) -> str:
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{xs[i]}-{xs[j]}" if j > i else str(xs[i]))
        i = j + 1
    return ",".join(out)


def validate(root: str, full: bool = False, hashes: str | None = None, jobs: int = 2, progress=None) -> Report:
    return Validator(root, full, hashes, jobs, progress).run()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m validator.validate", description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="release directory: a local copy or a Hugging Face snapshot directory")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="structure only, no hashing (default)")
    mode.add_argument("--full", action="store_true", help="also hash every weight file and check sidecars and set_id")
    ap.add_argument("--hashes", help="read sha256 sidecars from this directory instead of ROOT/hashes")
    ap.add_argument("--jobs", type=int, default=2, help="files hashed at once in --full mode (default 2)")
    ap.add_argument("--details", type=int, default=10, help="findings shown per check; -1 shows all (default 10)")
    ap.add_argument("--progress", action="store_true", help="print hashing progress to stderr")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.root):
        ap.error(f"{args.root} is not a directory")
    if args.hashes and not os.path.isdir(args.hashes):
        ap.error(f"{args.hashes} is not a directory")
    print(f"exlr8 fileset {os.path.abspath(args.root)} ({'full' if args.full else 'fast'} mode)")
    report = validate(args.root, args.full, args.hashes, max(1, args.jobs), sys.stderr if args.progress else None)
    report.render(sys.stdout, args.details)
    return 1 if report.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
