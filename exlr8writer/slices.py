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

SLICE_FORMAT = "exlr8-v2-index-slice"
SHARDING_RULE = "(16 * expert + half + layer) mod tp"
SET_ID_RULE = ('sha256 of the lines "<path> <size> <sha256>" (one per weight file: experts/*.safetensors and '
               'dense/*.safetensors, sorted by path, each ending in a newline)')
# FORMAT.md "Model metadata": absent means the GLM-5.3 value.
MODEL_KEYS = ("model", "moe_intermediate_size", "num_hidden_layers", "first_moe_layer", "tensor_prefix")
GLM53 = {"hidden_size": 6144, "n_routed_experts": 256}
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


def _layer_tensors(idx: _Index, layer: int, k: int, tp: int, hidden: int, routed: int) -> dict[int, dict[str, np.ndarray]]:
    """{rank: {name: array}} for one layer at width k and TP size tp."""
    shared = f"shared.k{k}.index" in idx

    def cat(name: str) -> np.ndarray:
        parts = [idx.get(f"k{k}.{name}")] + ([idx.get(f"shared.k{k}.{name}")] if shared else [])
        return np.concatenate(parts)

    experts = routed + (1 if shared else 0)
    index = cat("index")                                                           # [E, 3, 16, 2]
    check = cat("checksum")                                                        # [E, 3, 16]
    chan = np.stack([cat(c) for c in CHAN], 1)                                     # [E, 3, intermediate]
    block = np.stack([cat(b) for b in BLOCK], 1)                                   # [E, 3, hidden / 128]
    signs = np.stack([idx.get("signs.in"), idx.get("signs.out")]).astype(np.int8)
    half_bytes = hidden * 128 * k // 8
    e_, p_, h_ = np.meshgrid(np.arange(experts), np.arange(3), np.arange(16), indexing="ij")
    want = np.where(e_ < routed, 65536 + ((p_ * routed + e_) * 16 + h_) * half_bytes,
                    65536 + (3 * routed * 16 + p_ * 16 + h_) * half_bytes)
    if not (index[..., 0] == want).all() or not (index[..., 1] == half_bytes).all():
        raise ValueError(f"layer {layer} K{k}: half offsets do not follow the published layout")
    owner = (16 * np.arange(experts)[:, None] + np.arange(16)[None, :] + layer) % tp
    prefix = f"L{layer:03d}."
    out = {}
    for rank in range(tp):
        e, h = np.nonzero(owner == rank)                                           # sorted by expert, then half
        out[rank] = {prefix + "units": np.stack([e, h], 1).astype(np.int16),
                     prefix + "offset": index[e, :, h, 0].astype(np.int64),
                     prefix + "checksum": check[e, :, h].astype(np.uint64),
                     prefix + "chan": np.stack([chan[e[:, None], p, h[:, None] * 128 + np.arange(128)]
                                                for p in range(3)], 1),
                     prefix + "block": block,
                     prefix + "signs": signs}
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
    widths: Sequence[int] = (2, 3, 4),
    layers: Sequence[int] | None = None,
    set_id: Mapping[str, object] | None = None,
    source: str = "",
    hashes: str | os.PathLike[str] | None = None,
) -> list[dict[str, object]]:
    """Write ``out/tp<N>/k<K>/rank<r>.safetensors`` for every TP size, width and rank from the index files in
    ``experts``.

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
        acc: dict[tuple[int, int], dict[str, np.ndarray]] = {(n, r): {} for n in tp_sizes for r in range(n)}
        have: list[int] = []
        geometry = model = None
        for layer in layers:
            idx = _Index(experts / _index_name(layer))
            if f"k{k}.index" not in idx:
                continue
            g = (int(idx.meta.get("hidden_size", GLM53["hidden_size"])),
                 int(idx.meta.get("n_routed_experts", idx.meta.get("routed_experts", GLM53["n_routed_experts"]))))
            m = {key: idx.meta[key] for key in MODEL_KEYS if key in idx.meta}
            if geometry is not None and (g, m) != (geometry, model):
                raise ValueError(f"layer {layer}: model geometry differs from the other layers")
            geometry, model = g, m
            have.append(layer)
            for n in tp_sizes:
                for rank, tensors in _layer_tensors(idx, layer, k, n, *g).items():
                    acc[(n, rank)].update(tensors)
        if not have:
            continue
        hidden, routed = geometry
        glm53 = not model and (hidden, routed) == (GLM53["hidden_size"], GLM53["n_routed_experts"])
        for (n, rank), tensors in acc.items():
            meta: dict[str, object] = {"format": SLICE_FORMAT, "version": 1, "tp": n, "rank": rank, "k": k,
                                       "layers": ",".join(map(str, have))}
            if not glm53:
                meta.update(hidden_size=hidden, n_routed_experts=routed)
            meta.update(half_bytes=hidden * 128 * k // 8, rule=SHARDING_RULE,
                        files=f"experts/exlr8-layer-LLL.k{k}.safetensors", source=source)
            meta.update(model)
            if set_id is not None:
                sha = set_id["index_sha256"]
                meta.update(set_id=set_id["set_id"], set_id_rule=SET_ID_RULE, index_sha256=json.dumps(
                    {str(layer): sha[_index_name(layer)] for layer in have}, separators=(",", ":")))
            rel = Path(f"tp{n}") / f"k{k}" / f"rank{rank}.safetensors"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            size, digest = _write_packed(out / rel, tensors, meta)
            (hash_root / rel).parent.mkdir(parents=True, exist_ok=True)
            (hash_root / (str(rel) + ".sha256")).write_text(f"{digest} {size}\n")
            records.append({"path": rel.as_posix(), "bytes": size, "sha256": digest})
    return records
