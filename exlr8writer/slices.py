"""Index slices and the weight set ID of an EXLR8 release.

An index slice holds everything one tensor-parallel rank needs to fetch and load its expert halves at one width,
without reading the layer index files. See FORMAT.md, ``index-slices/tpN/kK/rankR.safetensors``.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .native import unit_bytes as native_unit_bytes

SLICE_FORMAT = "exlr8-v2-index-slice"
SHARDING_RULE = "({units} * expert + half + layer) mod tp"
SET_ID_RULE = ('sha256 of the lines "<path> <size> <sha256>" (one per weight file: experts/*.safetensors and '
               'dense/*.safetensors, sorted by path, each ending in a newline)')
# FORMAT.md "Model metadata": absent means the GLM-5.3 value.
MODEL_KEYS = ("model", "moe_intermediate_size", "num_hidden_layers", "first_moe_layer", "tensor_prefix",
              "expert_names")
GLM53 = {"hidden_size": 6144, "n_routed_experts": 256, "moe_intermediate_size": 2048}
ALIGN = 1 << 16
PROJECTIONS = ("gate", "up", "down")
CHAN = ("gate.svh", "up.svh", "down.suh")
BLOCK = ("gate.su_block", "up.su_block", "down.sv_block")
_DTYPES = {"U8": "u1", "I8": "i1", "I16": "<i2", "I32": "<i4", "I64": "<i8", "U64": "<u8", "F16": "<f2", "F32": "<f4"}
_NAMES = {np.dtype(v): k for k, v in _DTYPES.items()}


def _index_name(layer: int) -> str:
    return f"exlr8-layer-{layer:03d}.index.safetensors"


class _Index:
    """Read-only view of one layer index file."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            self.raw = f.read()
        self.meta: dict[str, str] = header.pop("__metadata__", None) or {}
        self.tensors = header

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def get(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        lo, hi = t["data_offsets"]
        return np.frombuffer(self.raw[lo:hi], _DTYPES[t["dtype"]]).reshape(t["shape"])


def _store(width: int | str) -> str:
    """The store name of a width: ``kK`` for trellis width K, else a native store's own name."""
    return f"k{width}" if isinstance(width, int) else width


def _unit_bytes(width: int | str, hidden: int) -> int:
    return hidden * 128 * width // 8 if isinstance(width, int) else native_unit_bytes(width, hidden)


def _layer_tensors(idx: _Index, layer: int, width: int | str, tp: int, hidden: int, routed: int,
                   units: int) -> dict[int, dict[str, np.ndarray]]:
    """{rank: {name: array}} for one layer at one width or native store and TP size tp."""
    s = _store(width)
    has = [p for p in ("", "shared.") if f"{p}{s}.index" in idx]

    def cat(name: str) -> np.ndarray:
        return np.concatenate([idx.get(f"{p}{s}.{name}") for p in has])

    # Expert slots: the routed experts, then the shared expert as expert E.
    slots = np.concatenate([np.arange(routed)] * ("" in has) + [np.array([routed])] * ("shared." in has))
    index = cat("index")                                                           # [slots, 3, U, 2]
    check = cat("checksum")                                                        # [slots, 3, U]
    b = _unit_bytes(width, hidden)
    # The tensors follow one another from 64 KiB, each starting on a 64 KiB boundary.
    want, pos, row = np.empty(index.shape[:3], np.int64), ALIGN, 0
    for p in has:
        n = routed if p == "" else 1
        for pi in range(3):
            want[row:row + n, pi] = pos + (np.arange(n)[:, None] * units + np.arange(units)) * b
            pos += -(-(n * units * b) // ALIGN) * ALIGN
        row += n
    if not (index[..., 0] == want).all() or not (index[..., 1] == b).all():
        raise ValueError(f"layer {layer} {s}: unit offsets do not follow the published layout")
    owner = (units * slots[:, None] + np.arange(units)[None, :] + layer) % tp
    prefix = f"L{layer:03d}."
    trellis = isinstance(width, int)
    if trellis:
        chan = np.stack([cat(c) for c in CHAN], 1)                                 # [slots, 3, intermediate]
        block = np.stack([cat(c) for c in BLOCK], 1)                               # [slots, 3, hidden / 128]
        signs = np.stack([idx.get("signs.in"), idx.get("signs.out")]).astype(np.int8)
    out = {}
    for rank in range(tp):
        e, h = np.nonzero(owner == rank)                                           # sorted by expert, then unit
        t = {prefix + "units": np.stack([slots[e], h], 1).astype(np.int16),
             prefix + "offset": index[e, :, h, 0].astype(np.int64),
             prefix + "checksum": check[e, :, h].astype(np.uint64)}
        if trellis:
            t.update({prefix + "chan": np.stack([chan[e[:, None], p, h[:, None] * 128 + np.arange(128)]
                                                 for p in range(3)], 1),
                      prefix + "block": block,
                      prefix + "signs": signs})
        out[rank] = t
    return out


def _write_packed(path: Path, tensors: Mapping[str, np.ndarray], meta: Mapping[str, object]) -> tuple[int, str]:
    """Packed safetensors: tensors in order of decreasing item size (each naturally aligned), header padded to 8.

    Index slices are small and read whole, so they do not use the 64 KiB layout of the weight files.
    """
    order = sorted(tensors, key=lambda n: (-tensors[n].dtype.itemsize, n))
    header: dict[str, object] = {}
    offset = 0
    for name in order:
        a = tensors[name]
        header[name] = {"dtype": _NAMES[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
    header["__metadata__"] = {str(k): str(v) for k, v in meta.items()}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    digest = hashlib.sha256()
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        for chunk in (struct.pack("<Q", len(encoded)), encoded,
                      *(np.ascontiguousarray(tensors[n]).tobytes() for n in order)):
            f.write(chunk)
            digest.update(chunk)
    os.replace(tmp, path)
    return 8 + len(encoded) + offset, digest.hexdigest()


def release_set_id(root: str | os.PathLike[str]) -> dict[str, object]:
    """The weight set ID of the release under ``root`` and the sha256 of each index file.

    Reads every weight file once. Returns ``{"set_id": hex, "index_sha256": {file name: hex}}``.
    """
    root = Path(root)
    files = sorted([*root.glob("experts/*.safetensors"), *root.glob("dense/*.safetensors")])
    if not files:
        raise ValueError(f"{root}: no weight files under experts/ or dense/")
    lines, indexes = [], {}
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        digest = h.hexdigest()
        lines.append(f"{p.relative_to(root).as_posix()} {p.stat().st_size} {digest}\n")
        if p.name.endswith(".index.safetensors"):
            indexes[p.name] = digest
    return {"set_id": hashlib.sha256("".join(lines).encode()).hexdigest(), "index_sha256": indexes}


def write_index_slices(
    experts: str | os.PathLike[str],
    out: str | os.PathLike[str],
    tp_sizes: Sequence[int],
    *,
    widths: Sequence[int | str] = (2, 3, 4),
    layers: Sequence[int] | None = None,
    set_id: Mapping[str, object] | None = None,
    source: str = "",
    hashes: str | os.PathLike[str] | None = None,
) -> list[dict[str, object]]:
    """Write ``out/tp<N>/k<K>/rank<r>.safetensors`` for every TP size, width and rank from the index files in
    ``experts``. A native store in ``widths`` (by name) writes ``out/tp<N>/<store>/rank<r>.safetensors``.

    ``layers`` defaults to every layer with an index file. ``set_id`` is :func:`release_set_id`'s result (or the
    same keys from elsewhere); without it the slices carry no set ID. ``source`` names the release and revision.
    A SHA256 sidecar for each slice goes to ``hashes/tp<N>/k<K>/rank<r>.safetensors.sha256`` (default:
    ``out/hashes``); a release keeps them under ``hashes/index-slices/``.

    Each layer index is read once per width. Memory holds one width's slices for every TP size at a time.
    Raises ``ValueError`` when an index's offsets do not follow the published layout or the layers mix geometries.
    """
    experts, out = Path(experts), Path(out)
    hash_root = Path(hashes) if hashes is not None else out / "hashes"
    if layers is None:
        layers = sorted(int(p.name[12:15]) for p in experts.glob("exlr8-layer-*.index.safetensors"))
    if not layers:
        raise ValueError(f"{experts}: no index files")
    records = []
    for k in widths:
        s = _store(k)
        acc: dict[tuple[int, int], dict[str, np.ndarray]] = {(n, r): {} for n in tp_sizes for r in range(n)}
        have: list[int] = []
        routed_by_layer: dict[int, int] = {}
        geometry = model = None
        for layer in layers:
            idx = _Index(experts / _index_name(layer))
            if f"{s}.index" not in idx and f"shared.{s}.index" not in idx:
                continue
            g = (int(idx.meta.get("hidden_size", GLM53["hidden_size"])),
                 int(idx.meta.get("moe_intermediate_size", GLM53["moe_intermediate_size"])) // 128)
            m = {key: idx.meta[key] for key in MODEL_KEYS if key in idx.meta}
            if geometry is not None and (g, m) != (geometry, model):
                raise ValueError(f"layer {layer}: model geometry differs from the other layers")
            geometry, model = g, m
            routed_by_layer[layer] = int(idx.meta.get("n_routed_experts",
                                                      idx.meta.get("routed_experts", GLM53["n_routed_experts"])))
            have.append(layer)
            for n in tp_sizes:
                for rank, tensors in _layer_tensors(idx, layer, k, n, g[0], routed_by_layer[layer], g[1]).items():
                    acc[(n, rank)].update(tensors)
        if not have:
            continue
        hidden, units = geometry
        routed = routed_by_layer[have[0]]
        # Layers whose routed expert count differs from the first layer's (DeepSeek-V4.1's MTP stages).
        odd = {str(L): E for L, E in routed_by_layer.items() if E != routed}
        glm53 = not model and (hidden, routed) == (GLM53["hidden_size"], GLM53["n_routed_experts"])
        for (n, rank), tensors in acc.items():
            meta: dict[str, object] = {"format": SLICE_FORMAT, "version": 1, "tp": n, "rank": rank,
                                       ("k" if isinstance(k, int) else "store"): k,
                                       "layers": ",".join(map(str, have))}
            if not glm53:
                meta.update(hidden_size=hidden, n_routed_experts=routed)
            if odd:
                meta.update(layer_experts=json.dumps(odd, separators=(",", ":")))
            meta.update({"half_bytes" if isinstance(k, int) else "unit_bytes": _unit_bytes(k, hidden)},
                        rule=SHARDING_RULE.format(units=units), files=f"experts/exlr8-layer-LLL.{s}.safetensors",
                        source=source)
            meta.update(model)
            if set_id is not None:
                sha = set_id["index_sha256"]
                meta.update(set_id=set_id["set_id"], set_id_rule=SET_ID_RULE, index_sha256=json.dumps(
                    {str(layer): sha[_index_name(layer)] for layer in have}, separators=(",", ":")))
            rel = Path(f"tp{n}") / s / f"rank{rank}.safetensors"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            size, digest = _write_packed(out / rel, tensors, meta)
            (hash_root / rel).parent.mkdir(parents=True, exist_ok=True)
            (hash_root / (str(rel) + ".sha256")).write_text(f"{digest} {size}\n")
            records.append({"path": rel.as_posix(), "bytes": size, "sha256": digest})
    return records
