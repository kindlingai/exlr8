"""Native stores: a source checkpoint's own quantized expert bytes in EXLR8 units.

See FORMAT.md, "Native stores". A unit's bytes are the source codes for its 128 intermediate channels, then their
E8M0 scales, so the source tensors come back byte for byte by concatenating the units.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import xxhash

from .api import DTYPES, PROJECTIONS, _SafetensorsWriter, _tensor_shape, _write_arrays

# Per store: (weights per code byte, the E8M0 scale block along (out, in) of the HF weight).
STORES = {"mxfp4": (2, (1, 32)), "fp8b32": (1, (32, 32))}
NATIVE_SCHEME = "exlr8-native-v2"

# (weight, scale) of one projection in HF orientation as uint8: gate/up [I, H / per_byte] and down
# [H, I / per_byte] codes, with E8M0 scales of the store's granularity.
Projection = tuple[np.ndarray, np.ndarray]
Source = Callable[[int, str], Projection]


def unit_bytes(store: str, hidden: int) -> int:
    """B for one unit of 128 intermediate channels."""
    per_byte, (so, si) = STORES[store]
    return hidden * 128 // per_byte + hidden * 128 // (so * si)


def cut_units(store: str, projection: str, weight: np.ndarray, scale: np.ndarray, hidden: int,
              intermediate: int) -> list[bytes]:
    """The U units of one projection, each codes then scales, row-major in HF orientation.

    Raises ValueError when the arrays do not have the store's shapes.
    """
    per_byte, (so, si) = STORES[store]
    out, inp = (hidden, intermediate) if projection == "down" else (intermediate, hidden)
    w, s = np.ascontiguousarray(weight).view(np.uint8), np.ascontiguousarray(scale).view(np.uint8)
    if w.shape != (out, inp // per_byte) or s.shape != (out // so, inp // si):
        raise ValueError(f"{store} {projection}: weight {w.shape} scale {s.shape}, expected "
                         f"{(out, inp // per_byte)} {(out // so, inp // si)}")
    units = []
    for u in range(intermediate // 128):
        if projection == "down":
            codes = w[:, u * 128 // per_byte:(u + 1) * 128 // per_byte]
            scales = s[:, u * 128 // si:(u + 1) * 128 // si]
        else:
            codes, scales = w[u * 128:(u + 1) * 128], s[u * 128 // so:(u + 1) * 128 // so]
        units.append(np.ascontiguousarray(codes).tobytes() + np.ascontiguousarray(scales).tobytes())
    return units


def _tensor(prefix: str, store: str, projection: str) -> str:
    return f"{prefix}{store}.{projection}"


def _file_name(layer: int, store: str) -> str:
    return f"exlr8-layer-{layer:03d}.{store}.safetensors"


def write_native_layer(
    layer_id: int,
    out: str | os.PathLike[str],
    *,
    hidden_size: int,
    intermediate_size: int,
    routed_experts: int,
    routed: tuple[str, Source] | None,
    shared: tuple[str, Callable[[str], Projection]] | None,
    metadata: Mapping[str, str],
    hashes: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Write one MoE layer's native store files and its index file.

    ``routed`` is (store, source) with ``source(expert, projection)`` returning that projection's (weight, scale);
    ``shared`` is (store, source) with ``source(projection)``. The sources are called once per projection, in
    file order, so they can read from disk on demand. A layer that has both puts them in separate files when the
    stores differ. ``metadata`` adds the model metadata keys of FORMAT.md to every file.

    Raises ValueError when a store is unknown or a source returns the wrong shapes.
    """
    if routed is None and shared is None:
        raise ValueError("nothing to write")
    for store, _ in filter(None, (routed, shared)):
        if store not in STORES:
            raise ValueError(f"unknown native store {store!r}, expected one of {sorted(STORES)}")
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    units = intermediate_size // 128
    # (store, tensor prefix, expert count, unit source(expert slot, projection))
    groups = []
    if routed:
        groups.append((routed[0], "", routed_experts, routed[1]))
    if shared:
        groups.append((shared[0], "shared.", 1, lambda e, p, src=shared[1]: src(p)))
    base_meta = {"format": "exlr8-v2", "scheme": NATIVE_SCHEME, "layer": str(layer_id),
                 "hidden_size": str(hidden_size), "moe_intermediate_size": str(intermediate_size),
                 "n_routed_experts": str(routed_experts), "writer_commit": "exlr8writer-0.1.0", **metadata}
    arrays: list[tuple[str, np.ndarray]] = []
    records = []
    for store in dict.fromkeys(g[0] for g in groups):
        mine = [g for g in groups if g[0] == store]
        b = unit_bytes(store, hidden_size)
        tensors = [(_tensor(prefix, store, p), "U8", (n * units * b,)) for _, prefix, n, _ in mine for p in PROJECTIONS]
        writer = _SafetensorsWriter(out_path / _file_name(layer_id, store), tensors,
                                    {**base_meta, "contents": "native", "store": store, "unit_bytes": str(b),
                                     "index_file": f"exlr8-layer-{layer_id:03d}.index.safetensors"})
        for _, prefix, n, source in mine:
            index = np.zeros((n, 3, units, 2), np.int64)
            sums = np.zeros((n, 3, units), np.uint64)
            for pi, projection in enumerate(PROJECTIONS):
                name = _tensor(prefix, store, projection)
                start = writer.data_start + next(o for t, o, _ in writer.layout if t == name)
                writer.begin(name)
                for e in range(n):
                    for u, data in enumerate(cut_units(store, projection, *source(e, projection),
                                                       hidden_size, intermediate_size)):
                        writer.write(data)
                        index[e, pi, u] = (start + (e * units + u) * b, b)
                        sums[e, pi, u] = xxhash.xxh64_intdigest(data)
            arrays += [(f"{prefix}{store}.index", index), (f"{prefix}{store}.checksum", sums)]
        size, digest = writer.close()
        records.append({"path": _file_name(layer_id, store), "bytes": size, "sha256": digest})

    index_meta = {**base_meta, "contents": "index", "widths": "", "shared_widths": "",
                  "stores": routed[0] if routed else "", "shared_stores": shared[0] if shared else "",
                  "experts": f"0-{routed_experts - 1}", "store_files": ",".join(r["path"] for r in records),
                  "sharding": f"unit u of expert e at TP=N goes to rank ({units}e + u + L) mod N; the shared "
                              "expert is expert E",
                  "checksum": "xxhash64 seed 0 per (expert, projection, unit) over the unit's bytes"}
    defs = [(n, DTYPES[a.dtype], _tensor_shape(a)) for n, a in arrays]
    index_name = f"exlr8-layer-{layer_id:03d}.index.safetensors"
    size, digest = _write_arrays(_SafetensorsWriter(out_path / index_name, defs, index_meta), arrays)
    records.append({"path": index_name, "bytes": size, "sha256": digest})
    hash_path = Path(hashes) if hashes is not None else out_path / "hashes"
    hash_path.mkdir(parents=True, exist_ok=True)
    for r in records:
        (hash_path / (str(r["path"]) + ".sha256")).write_text(f"{r['sha256']} {r['bytes']}\n")
    return {"layer": layer_id, "files": records, "bytes": sum(int(r["bytes"]) for r in records)}


__all__ = ["STORES", "NATIVE_SCHEME", "unit_bytes", "cut_units", "write_native_layer"]
