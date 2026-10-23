"""Provider based EXLR8 fileset writer.

The package owns EXLR8 serialization. Callers own model loading and quantization and
provide a :class:`LayerData` through the small ``get_layer`` hook.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import xxhash

ALIGN = 1 << 16
PROJECTIONS = ("gate", "up", "down")
SIDES = {
    "gate": ("svh", "su_block"),
    "up": ("svh", "su_block"),
    "down": ("suh", "sv_block"),
}
DTYPES = {
    np.dtype("uint8"): "U8", np.dtype("int8"): "I8", np.dtype("int16"): "I16",
    np.dtype("float16"): "F16", np.dtype("int32"): "I32", np.dtype("int64"): "I64", np.dtype("uint64"): "U64",
    np.dtype("float32"): "F32",
}


@dataclass(frozen=True)
class ModelMetadata:
    """Dimensions and format provenance needed to write one model layer."""

    routed_experts: int
    hidden_size: int
    intermediate_size: int
    scheme: str = "exlr8-e4m3-v2"
    hessian: str = "unspecified"
    hessian_rule: str = "unspecified"
    codebook: str = "mcg 0xCBAC1FED, E4M3 rounding"
    encoder_commit: str = "unspecified"
    exllamav3_version: str = "unspecified"
    sign_seed: str = "unspecified"
    rotation: str = "unspecified"
    shrink: str = "unspecified"
    error_reference: str = "unspecified"
    max_heldout: str = "0"
    min_heldout: str = "0"
    extra: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectionData:
    """One already quantized projection at one width.

    ``tiles`` is int16 trellis data in the codec's native tile order. Gate/up
    shapes are ``[hidden/16, intermediate/16, 16*K]``; down is
    ``[intermediate/16, hidden/16, 16*K]``. Side arrays are fp16. ``error`` is
    optional held-out error data: one value per 128 intermediate channels for
    gate/up, one scalar for down.
    """

    tiles: np.ndarray
    intermediate_scale: np.ndarray
    hidden_block_scale: np.ndarray
    error: np.ndarray | None = None


@dataclass(frozen=True)
class EncodedExpert:
    """Quantized values for a routed expert or shared expert."""

    expert_id: int
    widths: Mapping[int, Mapping[str, ProjectionData]]
    heldout_tokens: int | None = None


@dataclass(frozen=True)
class LayerData:
    """Layer payload supplied by a model-specific provider hook."""

    layer_id: int
    metadata: ModelMetadata
    signs_in: np.ndarray
    signs_out: np.ndarray
    experts: Sequence[EncodedExpert]
    shared_expert: EncodedExpert | None = None


class LayerProvider(Protocol):
    """Implement this hook to connect EXLR8 writing to any model workflow."""

    def get_layer(self, layer_id: int) -> LayerData:
        """Return quantized tensors and metadata for ``layer_id``."""


class _SafetensorsWriter:
    """Streaming writer for the aligned safetensors layout used by EXLR8."""

    def __init__(self, path: Path, tensors: list[tuple[str, str, tuple[int, ...]]], metadata: dict[str, str]):
        self.path = path
        header: dict[str, object] = {"__metadata__": metadata}
        self.layout: list[tuple[str, int, int]] = []
        offset = pad_number = 0
        for name, dtype, shape in tensors:
            if offset % ALIGN:
                gap = ALIGN - offset % ALIGN
                pad_name = f"__pad.{pad_number}"
                header[pad_name] = {"dtype": "U8", "shape": [gap], "data_offsets": [offset, offset + gap]}
                self.layout.append((pad_name, offset, gap))
                offset += gap
                pad_number += 1
            nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype({
                "U8": "u1", "I8": "i1", "I16": "<i2", "F16": "<f2", "I32": "<i4", "I64": "<i8", "U64": "<u8", "F32": "<f4",
            }[dtype]).itemsize
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
            self.layout.append((name, offset, nbytes))
            offset += nbytes
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        padded = ((8 + len(encoded) + ALIGN - 1) // ALIGN) * ALIGN
        self.data_start = padded
        header_size = padded - 8
        self.data_size = offset
        self.fd = os.open(str(path) + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        self.sha = hashlib.sha256()
        self.pos = self.next = 0
        self.current: tuple[str, int] | None = None
        self._write(header_size.to_bytes(8, "little") + encoded + b" " * (header_size - len(encoded)))

    def _write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            n = os.write(self.fd, view)
            if n <= 0:
                raise OSError("short safetensors write")
            view = view[n:]
        self.sha.update(data)
        self.pos += len(data)

    def tensor(self, name: str, data: bytes) -> None:
        self.begin(name)
        self.write(data)

    def begin(self, name: str) -> None:
        if self.current is not None:
            if self.pos != self.current[1]:
                raise RuntimeError(f"{self.path}: incomplete tensor {self.current[0]}")
            self.current = None
        while self.next < len(self.layout):
            entry, offset, nbytes = self.layout[self.next]
            self.next += 1
            if entry == name:
                self.current = (name, self.data_start + offset + nbytes)
                return
            if not entry.startswith("__pad."):
                raise RuntimeError(f"{self.path}: expected tensor {entry} before {name}")
            self._write(bytes(nbytes))
        raise RuntimeError(f"{self.path}: unexpected tensor {name}")

    def write(self, data: bytes) -> None:
        if self.current is None:
            raise RuntimeError(f"{self.path}: begin a tensor before writing bytes")
        if self.pos + len(data) > self.current[1]:
            raise ValueError(f"{self.path}: writing beyond tensor {self.current[0]}")
        self._write(data)

    def close(self) -> tuple[int, str]:
        if self.current is not None and self.pos != self.current[1]:
            raise RuntimeError(f"{self.path}: incomplete tensor {self.current[0]}")
        self.current = None
        while self.next < len(self.layout) and self.layout[self.next][0].startswith("__pad."):
            _, _, nbytes = self.layout[self.next]
            self.next += 1
            self._write(bytes(nbytes))
        if self.next != len(self.layout) or self.pos != self.data_start + self.data_size:
            os.close(self.fd)
            raise RuntimeError(f"{self.path}: incomplete file ({self.pos}, expected {self.data_start + self.data_size})")
        os.fsync(self.fd)
        os.close(self.fd)
        os.replace(str(self.path) + ".tmp", self.path)
        return self.pos, self.sha.hexdigest()


def _raw(array: np.ndarray, dtype: str | None = None) -> bytes:
    a = np.ascontiguousarray(array)
    if dtype is not None:
        a = a.astype({"F16": "<f2", "I8": "i1", "I16": "<i2", "F32": "<f4", "I32": "<i4", "I64": "<i8", "U64": "<u8"}[dtype], copy=False)
    return a.tobytes()


def _tile_halves(projection: str, tiles: np.ndarray, hidden_size: int, intermediate_size: int) -> list[bytes]:
    k_words = tiles.shape[-1]
    tile_bytes = k_words * 2
    units = intermediate_size // 128
    if projection == "down":
        a = np.ascontiguousarray(tiles).view(np.uint8)
        hidden_tiles = hidden_size // 16
        split = a.reshape(units, 8, hidden_tiles // 8, 8, tile_bytes).transpose(0, 2, 1, 3, 4)
    else:
        a = np.ascontiguousarray(tiles).view(np.uint8)
        hidden_tiles = hidden_size // 16
        split = a.reshape(hidden_tiles, units, 8, tile_bytes).transpose(1, 0, 2, 3)
    return [np.ascontiguousarray(split[h]).tobytes() for h in range(units)]


def _checksum(trellis: bytes, intermediate: np.ndarray, hidden_blocks: np.ndarray, half: int) -> int:
    digest = xxhash.xxh64(seed=0)
    digest.update(trellis)
    digest.update(_raw(intermediate[half * 128:(half + 1) * 128], "F16"))
    digest.update(_raw(hidden_blocks, "F16"))
    return digest.intdigest()


def _tensor_shape(array: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in array.shape)


def _validate_layer(layer: LayerData) -> None:
    meta = layer.metadata
    if meta.routed_experts < 1 or len(layer.experts) != meta.routed_experts:
        raise ValueError(f"layer {layer.layer_id}: expected {meta.routed_experts} routed experts, got {len(layer.experts)}")
    if meta.hidden_size % 128 or meta.intermediate_size % 128:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    if [e.expert_id for e in layer.experts] != list(range(meta.routed_experts)):
        raise ValueError("routed experts must be supplied once in ascending 0..E-1 order")
    if np.asarray(layer.signs_in).shape != (meta.hidden_size,) or np.asarray(layer.signs_out).shape != (meta.hidden_size,):
        raise ValueError("sign vector lengths must match hidden_size")
    if not np.isin(layer.signs_in, (-1, 1)).all() or not np.isin(layer.signs_out, (-1, 1)).all():
        raise ValueError("sign vectors must contain only -1 and +1")
    for expert in (*layer.experts, *((layer.shared_expert,) if layer.shared_expert else ())):
        for width, projections in expert.widths.items():
            if width not in (2, 3, 4) or set(projections) != set(PROJECTIONS):
                raise ValueError(f"expert {expert.expert_id}: width {width} must supply gate, up and down")
            for projection, data in projections.items():
                expected = ((meta.intermediate_size // 16, meta.hidden_size // 16, 16 * width) if projection == "down"
                            else (meta.hidden_size // 16, meta.intermediate_size // 16, 16 * width))
                if data.tiles.dtype != np.int16 or tuple(data.tiles.shape) != expected:
                    raise ValueError(f"expert {expert.expert_id} {projection} K{width}: tiles are {data.tiles.dtype} {data.tiles.shape}, expected int16 {expected}")
                if data.intermediate_scale.dtype != np.float16 or data.intermediate_scale.shape != (meta.intermediate_size,):
                    raise ValueError(f"expert {expert.expert_id} {projection}: intermediate scale must be fp16 [{meta.intermediate_size}]")
                if data.hidden_block_scale.dtype != np.float16 or data.hidden_block_scale.shape != (meta.hidden_size // 128,):
                    raise ValueError(f"expert {expert.expert_id} {projection}: hidden block scale must be fp16 [{meta.hidden_size // 128}]")


def _metadata(layer: LayerData, contents: str, **extra: str) -> dict[str, str]:
    model = layer.metadata
    values = {
        "format": "exlr8-v2", "contents": contents, "scheme": model.scheme,
        "layer": str(layer.layer_id), "routed_experts": str(model.routed_experts),
        "n_routed_experts": str(model.routed_experts),
        "hidden_size": str(model.hidden_size), "moe_intermediate_size": str(model.intermediate_size),
        "hessian": model.hessian, "hessian_rule": model.hessian_rule, "codebook": model.codebook,
        "encoder_commit": model.encoder_commit, "exllamav3_version": model.exllamav3_version,
        "sign_seed": model.sign_seed, "rotation": model.rotation, "shrink": model.shrink,
        "error_reference": model.error_reference, "max_heldout": model.max_heldout,
        "min_heldout": model.min_heldout,
        "gate_up_order": "grouped", "down_order": "item128",
        "pad_halves": "false", "writer_commit": "exlr8writer-0.1.0",
    }
    values.update({str(k): str(v) for k, v in model.extra.items()})
    values.update(extra)
    return values


def _file_names(layer_id: int, width: int) -> tuple[str, str]:
    return f"exlr8-layer-{layer_id:03d}.k{width}.safetensors", f"exlr8-layer-{layer_id:03d}.index.safetensors"


def _write_width(layer: LayerData, width: int, out: Path, pad_halves: bool) -> tuple[int, dict[str, object]]:
    meta = layer.metadata
    trellis_name, _ = _file_names(layer.layer_id, width)
    per_expert = [e.widths[width] for e in layer.experts] if all(width in e.widths for e in layer.experts) else []
    shared = layer.shared_expert.widths[width] if layer.shared_expert and width in layer.shared_expert.widths else None
    half_bytes = meta.hidden_size * 128 * width // 8
    units = meta.intermediate_size // 128
    stride = ((half_bytes + ALIGN - 1) // ALIGN) * ALIGN if pad_halves else half_bytes
    trellis_tensors: list[tuple[str, str, tuple[int, ...]]] = []
    for prefix, present in (("", per_expert), ("shared.", [shared] if shared else [])):
        if not present:
            continue
        for projection in PROJECTIONS:
            trellis_tensors.append((f"{prefix}trellis.{projection}", "U8", (len(present) * units * stride,)))
    trellis_meta = _metadata(layer, "trellis", k=str(width), index_file=_file_names(layer.layer_id, width)[1], pad_halves=str(pad_halves).lower())
    writer = _SafetensorsWriter(out / trellis_name, trellis_tensors, trellis_meta)
    for prefix, experts in (("", per_expert), ("shared.", [shared] if shared else [])):
        if not experts:
            continue
        for pi, projection in enumerate(PROJECTIONS):
            name = f"{prefix}trellis.{projection}"
            pos = 0
            writer.begin(name)
            for expert in experts:
                halves = _tile_halves(projection, expert[projection].tiles, meta.hidden_size, meta.intermediate_size)
                for half, data in enumerate(halves):
                    if len(data) != half_bytes:
                        raise ValueError(f"{projection} half has {len(data)} bytes, expected {half_bytes}")
                    writer.write(data)
                    if stride > half_bytes:
                        writer.write(bytes(stride - half_bytes))
                    pos += stride
            expected = len(experts) * units * stride
            if pos != expected:
                raise RuntimeError(f"{name}: emitted {pos} bytes, expected {expected}")
    file_size, file_hash = writer.close()
    return file_size, {"path": trellis_name, "bytes": file_size, "sha256": file_hash}


def write_layer(
    provider: LayerProvider,
    layer_id: int,
    out: str | os.PathLike[str],
    *,
    widths: Sequence[int] | None = None,
    shared_widths: Sequence[int] | None = None,
    pad_halves: bool = False,
    include_errors: bool = True,
    expert_limit: int | None = None,
    hashes: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Write a complete layer through a caller supplied tensor provider.

    ``provider.get_layer(layer_id)`` is the integration hook. It can read from
    safetensors, an object store, an in-memory quantizer, or another workflow.
    The returned ``LayerData`` contains encoded trellis tiles and side vectors;
    this package handles EXLR8 file layout, indexing, checksums and hashes.

    ``expert_limit`` writes only experts ``0..N-1`` and marks the index partial;
    it is intended for small integrations and format checks.
    """
    layer = provider.get_layer(layer_id)
    if layer.layer_id != layer_id:
        raise ValueError(f"provider returned layer {layer.layer_id} for request {layer_id}")
    _validate_layer(layer)
    if expert_limit is not None:
        if not 1 <= expert_limit <= len(layer.experts):
            raise ValueError(f"expert_limit must be between 1 and {len(layer.experts)}")
        layer = replace(layer, experts=layer.experts[:expert_limit])
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    if len({tuple(sorted(e.widths)) for e in layer.experts}) > 1:
        raise ValueError("all routed experts must provide the same width set")
    routed_widths = sorted(set.intersection(*(set(e.widths) for e in layer.experts))) if layer.experts else []
    if widths is not None:
        requested = set(widths)
        if requested - set(routed_widths):
            raise ValueError(f"requested routed widths are not present: {sorted(requested - set(routed_widths))}")
        routed_widths = [k for k in routed_widths if k in requested]
    shared_available = sorted(layer.shared_expert.widths) if layer.shared_expert else []
    if shared_widths is not None:
        requested = set(shared_widths)
        if requested - set(shared_available):
            raise ValueError(f"requested shared widths are not present: {sorted(requested - set(shared_available))}")
        shared_available = [k for k in shared_available if k in requested]
    chosen = sorted(set(routed_widths) | set(shared_available))
    if not chosen:
        raise ValueError("no routed or shared widths selected")
    records = []
    for width in chosen:
        if width not in routed_widths and width not in shared_available:
            continue
        file_size, file_record = _write_width(layer, width, out_path, pad_halves)
        records.append(file_record)

    # Build the index tensors from the same input payloads, following FORMAT.md.
    arrays: list[tuple[str, np.ndarray]] = [("signs.in", np.asarray(layer.signs_in, dtype=np.int8)),
                                             ("signs.out", np.asarray(layer.signs_out, dtype=np.int8))]
    # Index details are prepared by re-reading the written trellis files below.
    index_widths = [k for k in chosen if k in routed_widths]
    index_shared = [k for k in chosen if k in shared_available]
    for prefix, source_experts, width_list in (("", layer.experts, index_widths),
                                                ("shared.", [layer.shared_expert] if layer.shared_expert else [], index_shared)):
        for width in width_list:
            indices, sums = _rebuild_index(layer, width, source_experts, out_path, prefix, pad_halves)
            arrays.extend([(f"{prefix}k{width}.index", indices), (f"{prefix}k{width}.checksum", sums)])
            for projection in PROJECTIONS:
                inter_name, hidden_name = SIDES[projection]
                payload = source_experts
                arr_inter = np.stack([e.widths[width][projection].intermediate_scale for e in payload])
                arr_block = np.stack([e.widths[width][projection].hidden_block_scale for e in payload])
                arrays.extend([(f"{prefix}k{width}.{projection}.{inter_name}", arr_inter),
                               (f"{prefix}k{width}.{projection}.{hidden_name}", arr_block)])
    has_errors = False
    if include_errors:
        error_arrays: list[tuple[str, np.ndarray]] = []
        groups = (("", layer.experts, index_widths),
                  ("shared.", [layer.shared_expert] if layer.shared_expert else [], index_shared))
        for prefix, source_experts, group_widths in groups:
            if not source_experts or not group_widths:
                continue
            if any(e.heldout_tokens is None for e in source_experts):
                error_arrays = []
                break
            for width in group_widths:
                for projection in PROJECTIONS:
                    values = [e.widths[width][projection].error for e in source_experts]
                    if any(v is None for v in values):
                        error_arrays = []
                        break
                    arr = np.stack([np.asarray(v).reshape(-1) for v in values])
                    if projection == "down":
                        arr = arr[:, 0]
                    error_arrays.append((f"{prefix}k{width}.err_{projection}", arr.astype(np.float32)))
                if not error_arrays:
                    break
            if not error_arrays:
                break
            error_arrays.append((f"{prefix}heldout_tokens", np.asarray([e.heldout_tokens for e in source_experts], dtype=np.int32)))
        if error_arrays:
            has_errors = True
            arrays.extend(error_arrays)
    tensor_defs = [(name, DTYPES[np.asarray(arr).dtype], _tensor_shape(np.asarray(arr))) for name, arr in arrays]
    _, index_name = _file_names(layer_id, chosen[0])
    meta = _metadata(layer, "index", widths=",".join(map(str, index_widths)),
                     shared_widths=",".join(map(str, index_shared)),
                     experts=f"0-{len(layer.experts) - 1}" + (" (partial: test subset)" if len(layer.experts) != layer.metadata.routed_experts else ""),
                     trellis_files=",".join(_file_names(layer_id, k)[0] for k in chosen),
                     pad_halves=str(pad_halves).lower(),
                     sharding="half h of expert e at TP=N goes to rank (16e + h + L) mod N; shared expert follows routed experts",
                     checksum="xxhash64 seed 0 per (expert, projection, half) over the half's trellis bytes (no padding), then its 128 intermediate-side fp16 values (gate/up svh, down suh), then the projection's hidden-side fp16 block scales (gate/up su_block, down sv_block)",
                     error_inputs="provider supplied held-out errors" if has_errors else "none: no error tables")
    writer = _SafetensorsWriter(out_path / index_name, tensor_defs, meta)
    index_size, index_hash = _write_arrays(writer, arrays)
    records.append({"path": index_name, "bytes": index_size, "sha256": index_hash})
    total_bytes = sum(int(r["bytes"]) for r in records)
    hash_path = Path(hashes) if hashes is not None else out_path / "hashes"
    hash_path.mkdir(parents=True, exist_ok=True)
    for record in records:
        (hash_path / (str(record["path"]) + ".sha256")).write_text(f"{record['sha256']} {record['bytes']}\n")
    return {"layer": layer_id, "experts": len(layer.experts), "partial": len(layer.experts) != layer.metadata.routed_experts,
            "widths": index_widths,
            "shared_widths": index_shared, "files": records, "bytes": total_bytes}


def _write_arrays(writer: _SafetensorsWriter, arrays: list[tuple[str, np.ndarray]]) -> tuple[int, str]:
    for name, array in arrays:
        dtype = DTYPES.get(np.asarray(array).dtype)
        if dtype is None:
            raise TypeError(f"unsupported safetensors dtype for {name}: {np.asarray(array).dtype}")
        writer.tensor(name, _raw(np.asarray(array), dtype))
    return writer.close()


def _rebuild_index(layer: LayerData, width: int, experts: Sequence[EncodedExpert], out: Path,
                   prefix: str, pad_halves: bool) -> tuple[np.ndarray, np.ndarray]:
    """Recompute offsets and checksums from the completed trellis file layout."""
    name, _ = _file_names(layer.layer_id, width)
    with open(out / name, "rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len))
    data_start = 8 + header_len
    stride = layer.metadata.hidden_size * 128 * width // 8
    if pad_halves:
        stride = ((stride + ALIGN - 1) // ALIGN) * ALIGN
    units = layer.metadata.intermediate_size // 128
    indices = np.zeros((len(experts), 3, units, 2), dtype=np.int64)
    checksums = np.zeros((len(experts), 3, units), dtype=np.uint64)
    for pi, projection in enumerate(PROJECTIONS):
        info = header[f"{prefix}trellis.{projection}"]
        start = data_start + info["data_offsets"][0]
        for ei, expert in enumerate(experts):
            item = expert.widths[width][projection]
            halves = _tile_halves(projection, item.tiles, layer.metadata.hidden_size, layer.metadata.intermediate_size)
            for half, payload in enumerate(halves):
                indices[ei, pi, half] = (start + (ei * units + half) * stride, len(payload))
                checksums[ei, pi, half] = _checksum(payload, item.intermediate_scale, item.hidden_block_scale, half)
    return indices, checksums
