# Reference loader for exlr8 v2 checkpoints (GLM-5.3 exlr8 K2/K3/K4).
#
# This is a derivative of exllamav3 by turboderp (https://github.com/turboderp-org/exllamav3). The trellis
# codebook, the bit packing, the tile order and the 128x128 Hadamard rotation reproduce exllamav3 0.0.43.
# MIT licence: Copyright (c) 2025 Turboderp. See LICENSE in this directory.
#
# Pure PyTorch, CPU or GPU, written for clarity and checking rather than speed.
"""Decode exlr8 v2 weights into HF tensors (GLM-5.3, or the model the files' metadata names).

    python -m loader.exlr8 DIR --manifest home --layer 40 [--out layer40.safetensors] [--verify] [--raw]

DIR is a copy of the release (experts/, dense/, manifests/). --layer takes a layer number or "vocab". --raw
returns native stores and FP8 32x32 matrices as the source checkpoint's own tensors, byte for byte, in place of
decoded values. Split matrices (dense/exlr8-split-*) are listed, not decoded.
"""
import argparse
import json
import math
import os
import struct

import torch

GLM53 = {"hidden_size": 6144, "moe_intermediate_size": 2048, "n_routed_experts": 256, "tensor_prefix": "model.layers.",
         "expert_names": {**{p: f"mlp.experts.{{e}}.{p}_proj" for p in ("gate", "up", "down")},
                          **{f"shared.{p}": f"mlp.shared_experts.{p}_proj" for p in ("gate", "up", "down")}}}
PROJS = ("gate", "up", "down")
# Native stores (FORMAT.md "Native stores"): weights per code byte, and the E8M0 scale block (out, in).
NATIVE = {"mxfp4": (2, (1, 32)), "fp8b32": (1, (32, 32))}
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
MCG = 0xCBAC1FED
MUL1 = 0x83DCD12D
# mul1's scale and offset, exactly the fp16 bit patterns the kernel holds: v = fp16(1024 + byte_sum) * C + D.
C_MUL1 = 0.00676727294921875    # fp16 0x1EEE
D_MUL1 = -10.3828125            # fp16 0xC931
TRELLIS, INT4, MXFP8, FP8_BLOCK, FP8_ROW, BF16, FP32, FP8_B32 = 1, 2, 3, 4, 5, 6, 7, 8     # record field 0
IN = 2                                  # record field 4: 0 replicated, 1 output side, 2 input side
DATA = {TRELLIS: ("trellis", "suh", "svh"), INT4: ("q4", "s4"), MXFP8: ("fp8", "e8m0"), FP8_BLOCK: ("fp8", "scale"),
        FP8_ROW: ("fp8", "scale"), BF16: ("bf16",), FP32: ("fp32",), FP8_B32: ("fp8", "e8m0")}     # in checksum order
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
          "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "U64": torch.int64}
ST_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int64: "I64",
            torch.int32: "I32", torch.int8: "I8", torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3",
            torch.float8_e8m0fnu: "F8_E8M0"}


# ---------------------------------------------------------------- files

class SafeFile:
    """Reads named tensors or raw byte ranges from a safetensors file. U64 is returned as int64."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.meta = self.header.pop("__metadata__", None) or {}
        self.start = 8 + n

    def names(self):
        return [k for k in self.header if not k.startswith("__pad.")]

    def read(self, offset, length):
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        if len(data) != length:
            raise IOError(f"{self.path}: short read at {offset}")
        return torch.frombuffer(bytearray(data), dtype=torch.uint8)

    def get(self, name):
        info = self.header[name]
        a, b = info["data_offsets"]
        return self.read(self.start + a, b - a).view(DTYPES[info["dtype"]]).reshape(info["shape"])


def save_safetensors(tensors, path):
    """A minimal safetensors writer (no metadata)."""
    header, offset = {}, 0
    for name, t in tensors.items():
        n = t.numel() * t.element_size()
        header[name] = {"dtype": ST_NAMES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for t in tensors.values():
            f.write(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())


# ---------------------------------------------------------------- trellis codec

def tile_perm():
    """perm[j] = row * 16 + col of the tile element at stream position j (exllamav3's tensor-core order).
    Rows are the input side, columns the output side."""
    perm = []
    for t in range(32):
        r0, c0 = 2 * (t % 4), t // 4
        for r, c in ((r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                     (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)):
            perm.append(16 * r + c)
    return torch.tensor(perm)


def codebook(states, name="mcg"):
    """16-bit trellis states -> fp32 weights: the file's codebook, then E4M3 rounding (round to nearest even,
    saturating). The rounding is part of the format. The codebook comes from the trellis tensor name: bare for
    mcg, suffixed otherwise."""
    if name == "mcg":
        x = ((states & 0xFFFF) * MCG) & 0xFFFFFFFF
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60

        def fp16(v):
            return ((v ^ 0x8000) - 0x8000).to(torch.int16).view(torch.float16)
        v = (fp16(x & 0xFFFF) + fp16(x >> 16)).float()
    elif name == "mul1":
        x = ((states & 0xFFFF) * MUL1) & 0xFFFFFFFF
        byte_sum = sum((x >> shift) & 0xFF for shift in (0, 8, 16, 24))
        # fp16 0x6400 is 1024.0 and the binade's ulp is 1, so 1024 + byte_sum is exact. The kernel's fused
        # multiply-add rounds once: double here, then fp16, gives the same bits.
        v = ((1024 + byte_sum).double() * C_MUL1 + D_MUL1).half().float()
    else:
        raise ValueError(f"unknown codebook {name!r}")
    # IMPORTANT: round to E4M3 here, even when the output is BF16 or FP32. The encoder rounded every codebook value
    # to E4M3 inside its Viterbi search, so the stored states are the best choice for the rounded values, not the
    # fp16 ones. Skipping the rounding decodes different weights (+2.3% error in the README's measurement).
    return v.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def file_codebook(f):
    """A trellis file's codebook, from its tensor names: `trellis.gate` is mcg, `trellis.gate.mul1` is mul1.

    Raises ValueError if the file mixes codebooks, which no writer produces.
    """
    found = {n.rsplit(".", 1)[1] for n in f.names() if "trellis." in n} - set(PROJS)
    if len(found) > 1:
        raise ValueError(f"trellis file mixes codebooks {sorted(found)}")
    return found.pop() if found else "mcg"


def decode_tiles(raw, K, cb="mcg"):
    """Packed tiles, uint8 [n, 32K] -> fp32 [n, 16, 16], (input, output) within each tile."""
    n = raw.shape[0]
    b = raw.reshape(n, 16 * K, 2).long()
    words = (b[..., 0] | (b[..., 1] << 8)).reshape(n, 8 * K, 2).flip(-1).reshape(n, 16 * K)
    bits = (words.unsqueeze(-1) >> torch.arange(15, -1, -1, device=raw.device)) & 1        # MSB first
    vals = (bits.reshape(n, 256, K) << torch.arange(K - 1, -1, -1, device=raw.device)).sum(-1)
    states = torch.zeros_like(vals)
    for j in range(-(-16 // K)):                     # tail-biting: the 16 bits that end with value i
        states |= torch.roll(vals, j, dims=1) << (j * K)
    out = torch.empty(n, 256, device=raw.device)
    out[:, tile_perm().to(raw.device)] = codebook(states, cb)
    return out.view(n, 16, 16)


def tiles_out_major(raw, K, rows, cols, cb="mcg"):
    """A run of tiles stored [k16][c = 0..7] (gate/up halves, dense entries on the output side) -> (rows, 128)."""
    return decode_tiles(raw.view(-1, 32 * K), K, cb).view(rows // 16, cols // 16, 16, 16) \
        .permute(0, 2, 1, 3).reshape(rows, cols)


def tiles_in_major(raw, K, cols, cb="mcg"):
    """A run of tiles stored [b][r = 0..7][c = 0..7] (down halves, dense entries on the input side) -> (128, cols)."""
    return decode_tiles(raw.view(-1, 32 * K), K, cb).view(cols // 128, 8, 8, 16, 16) \
        .permute(1, 3, 0, 2, 4).reshape(128, cols)


def hadamard(n=128, device="cpu"):
    """Sylvester Hadamard matrix of order n, scaled by 1/sqrt(n)."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h * (1 / math.sqrt(n))).to(device)


def reconstruct(T, suh, svh):
    """W (in, out) = diag(suh) Hb T Hb diag(svh), Hb the block-diagonal 128-block Hadamard. The operations and
    their order are exllamav3's, so the result matches it value for value on the same device."""
    k, n = T.shape
    had = hadamard(128, T.device)
    x = (had @ T.view(-1, 128, n)).view(k, n) * suh.float().unsqueeze(1)
    return (x.view(k, -1, 128) @ had).view(k, n) * svh.float().unsqueeze(0)


# ---------------------------------------------------------------- experts

def e8m0(s):
    """E8M0 bytes -> fp32 powers of two."""
    e = s.view(torch.uint8).int() - 127
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)


def model_meta(meta):
    """The model metadata keys of a file, GLM-5.3's where absent."""
    m = {k: meta.get(k, v) for k, v in GLM53.items()}
    for k in ("hidden_size", "moe_intermediate_size", "n_routed_experts"):
        m[k] = int(m[k])
    if isinstance(m["expert_names"], str):
        m["expert_names"] = json.loads(m["expert_names"])
    return m


def _widths(text):
    return [int(k) for k in text.split(",") if k]


def _stores(text):
    return [k for k in (text or "").split(",") if k]


def split_native(raw, store, p, H):
    """A native unit's bytes -> (codes, scales) uint8, HF orientation (out, in)."""
    per_byte, (so, si) = NATIVE[store]
    (cr, cc), (sr, sc) = (((H, 128 // per_byte), (H // so, 128 // si)) if p == "down"
                          else ((128, H // per_byte), (128 // so, H // si)))
    return raw[:cr * cc].view(cr, cc), raw[cr * cc:].view(sr, sc)


def decode_native(codes, scales, store):
    """(codes, E8M0 scales) of an HF-orientation (out, in) block -> fp32 (out, in)."""
    _, (so, si) = NATIVE[store]
    if store == "mxfp4":
        c = codes.long()
        w = E2M1.to(codes.device)[torch.stack([c & 15, c >> 4], -1)].reshape(c.shape[0], -1)
    else:
        w = codes.view(torch.float8_e4m3fn).float()
    return w * e8m0(scales).repeat_interleave(so, 0).repeat_interleave(si, 1)


class ExpertLayer:
    """One MoE layer's index file and its per-width trellis and native store files."""

    def __init__(self, root, L, device="cpu"):
        self.L, self.device = L, device
        base = os.path.join(root, "experts", f"exlr8-layer-{L:03d}")
        self.index = SafeFile(base + ".index.safetensors")
        self.t = {n: self.index.get(n) for n in self.index.names()}
        self.m = model_meta(self.index.meta)
        self.H, self.U = self.m["hidden_size"], self.m["moe_intermediate_size"] // 128
        self.E = self.m["n_routed_experts"]
        meta = self.index.meta
        self.widths = _widths(meta["widths"]) + _stores(meta.get("stores"))
        self.shared_widths = _widths(meta["shared_widths"]) + _stores(meta.get("shared_stores"))
        self.files = {}
        self.base = base

    def trellis_file(self, K):
        """(file, codebook) for width K."""
        if K not in self.files:
            f = SafeFile(f"{self.base}.k{K}.safetensors")
            self.files[K] = (f, file_codebook(f))
        return self.files[K]

    def native_raw(self, e, p, h, store, verify=False):
        """Unit h of projection p of expert e in a native store: its bytes, uint8."""
        pre, j = ("shared.", 0) if e == "shared" else ("", e)
        k = f"{pre}{store}."
        off, length = (int(v) for v in self.t[k + "index"][j, PROJS.index(p), h])
        if store not in self.files:
            self.files[store] = (SafeFile(f"{self.base}.{store}.safetensors"), None)
        f = self.files[store][0]
        if f"{pre}{store}.{p}" not in f.header:
            raise ValueError(f"{f.path}: no tensor {pre}{store}.{p}")
        raw = f.read(off, length)
        if verify:
            want = int(self.t[k + "checksum"][j, PROJS.index(p), h]) & (2 ** 64 - 1)
            if xxh64(raw) != want:
                raise ValueError(f"layer {self.L} expert {e} {p} unit {h} {store}: checksum mismatch")
        return raw

    def half(self, e, p, h, K, verify=False):
        """Unit h of projection p of expert e (e = "shared" for the shared expert) at width K, or in a native
        store, as (in, out) fp32: gate/up (H, 128) = output channels 128h.., down (128, H) = input channels
        128h.."""
        pre, j = ("shared.", 0) if e == "shared" else ("", e)
        if K not in (self.shared_widths if e == "shared" else self.widths):
            raise ValueError(f"layer {self.L} expert {e} {p}: {K} is not stored")
        if isinstance(K, str):
            codes, scales = split_native(self.native_raw(e, p, h, K, verify).to(self.device), K, p, self.H)
            return decode_native(codes, scales, K).T
        k = f"{pre}k{K}."
        off, length = (int(v) for v in self.t[k + "index"][j, PROJS.index(p), h])
        f, cb = self.trellis_file(K)
        raw = f.read(off, length)
        c = slice(128 * h, 128 * h + 128)
        if p == "down":
            chan, block = self.t[k + "down.suh"][j, c], self.t[k + "down.sv_block"][j]
        else:
            chan, block = self.t[k + f"{p}.svh"][j, c], self.t[k + f"{p}.su_block"][j]
        if verify:
            want = int(self.t[k + "checksum"][j, PROJS.index(p), h]) & (2 ** 64 - 1)
            got = xxh64(raw, chan, block)
            if got != want:
                raise ValueError(f"layer {self.L} expert {e} {p} half {h} K{K}: checksum {got:#x}, expected {want:#x}")
        raw, chan, block = raw.to(self.device), chan.to(self.device), block.float().to(self.device)
        if p == "down":
            T = tiles_in_major(raw, K, self.H, cb)
            svh = self.t["signs.out"].to(self.device).float() * block.repeat_interleave(128)
            return reconstruct(T, chan, svh)
        T = tiles_out_major(raw, K, self.H, 128, cb)
        suh = self.t["signs.in"].to(self.device).float() * block.repeat_interleave(128)
        return reconstruct(T, suh, chan)

    def projection(self, e, p, widths, verify=False):
        """The whole projection as an HF weight (out, in) fp32. widths: U per-unit widths for gate/up, one for down;
        a width is K or a native store's name."""
        ks = widths if isinstance(widths, list) else [widths] * self.U
        parts = [self.half(e, p, h, ks[h], verify) for h in range(self.U)]
        return torch.cat(parts, dim=0 if p == "down" else 1).T

    def native_projection(self, e, p, store, verify=False):
        """The source checkpoint's (weight, scale) of one projection in a native store, byte for byte."""
        units = [split_native(self.native_raw(e, p, h, store, verify), store, p, self.H) for h in range(self.U)]
        axis = 1 if p == "down" else 0
        codes, scales = (torch.cat([u[i] for u in units], axis) for i in (0, 1))
        return (codes.view(torch.int8) if store == "mxfp4" else codes.view(torch.float8_e4m3fn),
                scales.view(torch.float8_e8m0fnu))


# ---------------------------------------------------------------- dense and vocabulary

def decode_dense(rec, t, device="cpu"):
    """One dense matrix as an HF weight (out, in) fp32. rec: the 8-field record, t: {suffix: tensor} as stored."""
    fmt, out, inp, K, side = (int(x) for x in rec[:5])
    t = {k: v.to(device) for k, v in t.items()}
    if fmt == TRELLIS:
        ent = t["trellis"]
        if side == IN:      # entry g = input rows 128g.., tiles as down halves
            T = torch.cat([tiles_in_major(ent[g], K, out) for g in range(ent.shape[0])], 0)
        else:               # entry g = output columns 128g.., tiles as gate/up halves
            T = torch.cat([tiles_out_major(ent[g], K, inp, 128) for g in range(ent.shape[0])], 1)
        return reconstruct(T, t["suh"], t["svh"]).T
    if fmt == INT4:
        q4, s4 = t["q4"], t["s4"]
        if side == IN:      # stored [in/128, out, 64] and [in/128, out, 4]
            q4, s4 = q4.transpose(0, 1).reshape(out, -1), s4.transpose(0, 1).reshape(out, -1)
        n = torch.stack([(q4 & 15).int() - 8, (q4 >> 4).int() - 8], -1).reshape(out, inp).float()
        return n * s4.float().repeat_interleave(32, 1)
    w = t["fp8"].view(torch.float8_e4m3fn).float() if "fp8" in t else None
    if fmt == MXFP8:
        return w * e8m0(t["e8m0"]).repeat_interleave(32, 1)
    if fmt == FP8_B32:
        q, s = fp8_b32_rows(t, side)
        return q.view(torch.float8_e4m3fn).float() * \
            e8m0(s).repeat_interleave(32, 0)[:out].repeat_interleave(32, 1)[:, :inp]
    if fmt == FP8_BLOCK:
        return w * t["scale"].repeat_interleave(128, 0)[:out].repeat_interleave(128, 1)[:, :inp]
    if fmt == FP8_ROW:
        return w * t["scale"].unsqueeze(1)
    return t["bf16" if fmt == BF16 else "fp32"].float()


def fp8_b32_rows(t, side):
    """Format 8's (fp8, e8m0) as row-major [out, in] and [out/32, in/32], undoing side 2's 128-input entries."""
    q, s = t["fp8"], t["e8m0"]
    if side == IN:      # stored [in/128, out, 128] and [in/128, out/32, 4]
        q, s = q.permute(1, 0, 2).reshape(q.shape[1], -1), s.permute(1, 0, 2).reshape(s.shape[1], -1)
    return q, s


def dense_checksums(rec, t):
    """M.checksum recomputed from the stored tensors: one xxhash64 per 128-channel block on the sharded side."""
    fmt, out, inp, K, side = (int(x) for x in rec[:5])
    sums = []
    for b in range(-(-(inp if side == IN else out) // 128)):
        r = slice(128 * b, 128 * b + 128)
        if fmt == TRELLIS:
            own, other = ("suh", "svh") if side == IN else ("svh", "suh")
            parts = [t["trellis"][b], t[own][r], t[other]]
        elif fmt in (INT4, FP8_B32) and side == IN:
            parts = [t[k][b] for k in DATA[fmt]]
        elif fmt == FP8_B32:
            parts = [t["fp8"][r], t["e8m0"][4 * b:4 * b + 4]]
        elif fmt == FP8_BLOCK:
            parts = [t["fp8"][r], t["scale"][b]]
        else:
            parts = [t[k][r] for k in DATA[fmt]]
        sums.append(xxh64(*parts))
    return sums


def load_matrices(f, prefix, dtype, device, verify, raw=False):
    """Every tensor of a dense or vocabulary file: matrices decoded (by their records), vectors as stored. With raw,
    FP8 32x32 block matrices come back as the source's (weight, scale), MXFP8 as (weight, scale) too, and BF16 as
    stored."""
    names = f.names()
    mats = [n[:-len(".record")] for n in names if n.endswith(".record")]
    out, owned = {}, set()
    for m in mats:
        rec = f.get(m + ".record")
        t = {s: f.get(f"{m}.{s}") for s in DATA[int(rec[0])]}
        if verify:
            want = [int(v) & (2 ** 64 - 1) for v in f.get(m + ".checksum")]
            if dense_checksums(rec, t) != want:
                raise ValueError(f"{f.path}: {m} checksum mismatch")
        fmt, side = int(rec[0]), int(rec[4])
        if raw and fmt in (FP8_B32, MXFP8):
            q, s = fp8_b32_rows(t, side) if fmt == FP8_B32 else (t["fp8"], t["e8m0"])
            out[prefix(m) + ".weight"] = q.view(torch.float8_e4m3fn)
            out[prefix(m) + ".scale"] = s.view(torch.float8_e8m0fnu)
        elif raw and fmt == BF16:
            out[prefix(m) + ".weight"] = t["bf16"]
        else:
            out[prefix(m) + ".weight"] = decode_dense(rec, t, device).to(dtype).cpu()
        owned.update(f"{m}.{s}" for s in ("record", "checksum") + DATA[int(rec[0])])
    for n in names:
        if n not in owned:
            out[prefix(n)] = f.get(n)
    return out


# ---------------------------------------------------------------- checksums

def xxh64(*parts):
    """xxhash64, seed 0, over the bytes of the given tensors in order. Needs the xxhash package."""
    import xxhash
    h = xxhash.xxh64(seed=0)
    for p in parts:
        h.update(p.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.intdigest()


# ---------------------------------------------------------------- layers

def load_manifest(root, name):
    path = name if os.path.isfile(name) else os.path.join(root, "manifests", f"exlr8-manifest-{name}.json")
    with open(path) as f:
        return json.load(f)


def load_layer(root, manifest, L, dtype=torch.bfloat16, device="cpu", verify=False, raw=False):
    """Every tensor of layer L under its HF names; experts at the manifest's widths. With raw, experts in a native
    store come back as the source's (weight, scale)."""
    f = SafeFile(os.path.join(root, "dense", f"exlr8-dense-{L:03d}.safetensors"))
    pre = f.meta.get("hf_prefix") or f"{f.meta.get('tensor_prefix', GLM53['tensor_prefix'])}{L}."
    out = load_matrices(f, lambda n: pre + n, dtype, device, verify, raw)
    m = manifest["layers"].get(str(L))
    if m is None:                    # a dense MLP layer: no experts
        return out
    x = ExpertLayer(root, L, device)
    names = x.m["expert_names"]

    def put(e, p, width, name):
        store = width if isinstance(width, str) else width[0] if len(set(width)) == 1 else None
        if raw and isinstance(store, str):
            out[name + ".weight"], out[name + ".scale"] = x.native_projection(e, p, store, verify)
        else:
            out[name + ".weight"] = x.projection(e, p, width, verify).to(dtype).cpu()
    for e in range(x.E):
        for p in PROJS:
            put(e, p, m[p][e], pre + names[p].format(e=e))
    for p in PROJS:
        put("shared", p, m["shared"][p], pre + names["shared." + p])
    return out


def load_vocab(root, dtype=torch.bfloat16, device="cpu", verify=False, raw=False):
    f = SafeFile(os.path.join(root, "dense", "exlr8-vocab.safetensors"))
    names = {"embed_tokens": "model.embed_tokens", "lm_head": "lm_head", "norm.weight": "model.norm.weight"}
    if "hf_names" in f.meta:
        names = json.loads(f.meta["hf_names"])
    return load_matrices(f, lambda n: names.get(n, n), dtype, device, verify, raw)


def main():
    ap = argparse.ArgumentParser(description="Decode one exlr8 v2 layer into HF tensors.")
    ap.add_argument("dir", help="the release directory (experts/, dense/, manifests/)")
    ap.add_argument("--manifest", default="home",
                    help="home, down1, up1, bit225, bit243, bit260, bit325, bit338, or a manifest file")
    ap.add_argument("--layer", required=True, help="layer number, or vocab")
    ap.add_argument("--out", help="write the tensors to this safetensors file")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verify", action="store_true", help="check every checksum (needs xxhash)")
    ap.add_argument("--raw", action="store_true", help="native and FP8 32x32 weights as the source's own tensors")
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    if a.layer == "vocab":
        tensors = load_vocab(a.dir, dtype, a.device, a.verify, a.raw)
    else:
        tensors = load_layer(a.dir, load_manifest(a.dir, a.manifest), int(a.layer), dtype, a.device, a.verify,
                             a.raw)
        splits = sorted(n for n in os.listdir(os.path.join(a.dir, "dense"))
                        if n.startswith(f"exlr8-split-{int(a.layer):03d}."))
        for n in splits:
            print(f"split file, not decoded: dense/{n}")
    for name, t in tensors.items():
        print(f"{name} {str(t.dtype).replace('torch.', '')} {list(t.shape)}")
    if a.out:
        save_safetensors(tensors, a.out)
        print(f"wrote {len(tensors)} tensors to {a.out}")


if __name__ == "__main__":
    main()
