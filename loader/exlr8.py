# Reference loader for exlr8 v2 and v4 checkpoints (FORMAT.md).
#
# This is a derivative of exllamav3 by turboderp (https://github.com/turboderp-org/exllamav3). The trellis
# codebook, the bit packing, the tile order and the 128x128 Hadamard rotation reproduce exllamav3 0.0.43.
# MIT licence: Copyright (c) 2025 Turboderp. See LICENSE in this directory.
#
# Pure PyTorch, CPU or GPU, written for clarity and checking rather than speed.
"""Decode exlr8 v2 and v4 weights into HF tensors (GLM-5.3, or the model the files' metadata names).

    python -m loader.exlr8 DIR --manifest home --layer 40 [--out layer40.safetensors] [--verify] [--raw]

DIR is a copy of the release (experts/, dense/, manifests/). --layer takes a layer number or "vocab". --raw
returns native stores and FP8 32x32 matrices as the source checkpoint's own tensors, byte for byte, in place of
decoded values. Split matrices (dense/exlr8-split-*) are listed, not decoded.
"""
import argparse
import json
import math
import os
import re
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
MCG2 = 0xBCA14C51               # pmcgx's second multiplier
MUL1 = 0x83DCD12D
PMUL1X = (0x13755EC1, 0xF8D9B559)   # pmul1x's two multipliers
# mul1's scale and offset, exactly the fp16 bit patterns the kernel holds: v = fp16(1024 + byte_sum) * C + D.
C_MUL1 = 0.00676727294921875    # fp16 0x1EEE
D_MUL1 = -10.3828125            # fp16 0xC931
# The trellis codebooks (FORMAT.md "Codebook"): name -> (state bits, multipliers its `codebook` metadata must name).
# Each has an -f16 form with the same hash and no E4M3 rounding.
CODEBOOKS = {"mcg": (16, ("0xCBAC1FED",)), "mul1": (16, ("0x83DCD12D",)), "pmcgx": (20, ("0xCBAC1FED", "0xBCA14C51")),
             "pmul1x": (20, ("0x13755EC1", "0xF8D9B559"))}
CODEBOOKS.update({f"{k}-f16": v for k, v in list(CODEBOOKS.items())})
# The byte-sum codebooks: their -f16 forms decode to the exact integer (byte sum - 510) * C_MUL1, not the fp16 v.
BYTE_SUM = ("mul1", "pmul1x")
FORMATS = ("exlr8-v2", "exlr8-v4")
MIXING = ("none", "width", "expert")
TRELLIS, INT4, MXFP8, FP8_BLOCK, FP8_ROW, BF16, FP32, FP8_B32, ROWS, INT8 = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10  # field 0
IN = 2                                  # record field 4: 0 replicated, 1 output side, 2 input side
DATA = {TRELLIS: ("trellis", "suh", "svh"), INT4: ("q4", "s4"), MXFP8: ("fp8", "e8m0"), FP8_BLOCK: ("fp8", "scale"),
        FP8_ROW: ("fp8", "scale"), BF16: ("bf16",), FP32: ("fp32",), FP8_B32: ("fp8", "e8m0"),
        ROWS: ("rows", "svh", "suh"), INT8: ("q8", "s8")}     # in checksum order
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


def is_pair(cb):
    """pmcgx, pmul1x and their -f16 forms: one 20-bit state per pair of weights, 2K new bits per state."""
    return cb.removesuffix("-f16") in ("pmcgx", "pmul1x")


def check_width(K, cb):
    """Raises ValueError unless codebook cb has width K (FORMAT.md "Widths")."""
    top = 6 if cb.endswith("-f16") else 5
    low, step = (0.5, 0.5) if is_pair(cb) else (1, 1)
    if not (low <= K <= top and (K / step) == int(K / step)):
        raise ValueError(f"codebook {cb} has no width K{K}")


def width_name(K):
    """k4, or k4p5 for K = 4.5 (FORMAT.md "Symbols")."""
    return f"k{int(K)}" + ("p5" if K != int(K) else "")


def _fp16(v):
    """16-bit patterns in an int64 tensor -> float16"""
    return ((v ^ 0x8000) - 0x8000).to(torch.int16).view(torch.float16)


def _mask(x):
    """mcg's lop3: two fp16 halves in [0.125, 2) from a 32-bit product"""
    return (x & 0x8FFF8FFF) ^ 0x3B603B60


def mul1_byte_sum(states):
    """mul1's hash: the sum of the four bytes of state * 0x83DCD12D, 0 to 1020."""
    x = ((states & 0xFFFF) * MUL1) & 0xFFFFFFFF
    return sum((x >> shift) & 0xFF for shift in (0, 8, 16, 24))


def pmul1x_byte_sums(states):
    """pmul1x's hash: [..., 2], the byte sums of weights 2p and 2p + 1, each 0 to 1020. x1 = w * c1 and x2 = w * c2;
    weight 2p sums x1's two low bytes and x2's two high bytes, weight 2p + 1 the other four."""
    w = states & 0xFFFFF
    x1, x2 = (w * PMUL1X[0]) & 0xFFFFFFFF, (w * PMUL1X[1]) & 0xFFFFFFFF
    b1 = [(x1 >> shift) & 0xFF for shift in (0, 8, 16, 24)]
    b2 = [(x2 >> shift) & 0xFF for shift in (0, 8, 16, 24)]
    return torch.stack([b1[0] + b1[1] + b2[2] + b2[3], b1[2] + b1[3] + b2[0] + b2[1]], -1)


def byte_sum(states, cb):
    """The byte sum of a byte-sum codebook (mul1, pmul1x, either form): [...] for mul1, [..., 2] for pmul1x."""
    return mul1_byte_sum(states) if cb.removesuffix("-f16") == "mul1" else pmul1x_byte_sums(states)


def codebook_f16(states, cb="mcg"):
    """Trellis states -> the codebook's fp16 values, before any rounding. mcg and mul1 take 16-bit states and give one
    value each; pmcgx and pmul1x take 20-bit pair states and give [..., 2], the values of weights 2p and 2p + 1. The
    -f16 forms hash the same way; mul1-f16 and pmul1x-f16 then take a different value (see codebook)."""
    name = cb.removesuffix("-f16")
    if name == "mcg":
        x = _mask(((states & 0xFFFF) * MCG) & 0xFFFFFFFF)
        return _fp16(x & 0xFFFF) + _fp16(x >> 16)
    if name in BYTE_SUM:
        # fp16 0x6400 is 1024.0 and the binade's ulp is 1, so 1024 + byte_sum is exact. The kernel's fused
        # multiply-add rounds once: double here, then fp16, gives the same bits.
        return ((1024 + byte_sum(states, name)).double() * C_MUL1 + D_MUL1).half()
    if name == "pmcgx":
        w = states & 0xFFFFF
        x1, x2 = _mask((w * MCG) & 0xFFFFFFFF), _mask((w * MCG2) & 0xFFFFFFFF)
        # the halves cross: weight 2p is x1's low half plus x2's high half
        return torch.stack([_fp16(x1 & 0xFFFF) + _fp16(x2 >> 16), _fp16(x1 >> 16) + _fp16(x2 & 0xFFFF)], -1)
    raise ValueError(f"unknown codebook {cb!r}")


def codebook(states, cb="mcg"):
    """Trellis states -> fp32 weights: the codebook's fp16 values, then, for every codebook but the -f16 forms, E4M3
    rounding (round to nearest even, saturating). mul1-f16 and pmul1x-f16 are the exact (byte_sum - 510) * C_MUL1."""
    if cb.endswith("-f16") and cb.removesuffix("-f16") in BYTE_SUM:
        # Exact in fp32 (9 by 11 significant bits), and unlike mul1's fp16 multiply-add it lets an fp16 engine keep
        # the integer byte_sum - 510 and apply C_MUL1 with its scales.
        return (byte_sum(states, cb) - 510).float() * C_MUL1
    v = codebook_f16(states, cb).float()
    if cb.endswith("-f16"):
        return v
    # IMPORTANT: round to E4M3 here, even when the output is BF16 or FP32. The encoder rounded every codebook value
    # to E4M3 inside its Viterbi search, so the stored states are the best choice for the rounded values, not the
    # fp16 ones. Skipping the rounding decodes different weights (+2.3% error in the README's measurement).
    return v.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def tile_bits(raw, K):
    """Packed tiles, uint8 [n, 32K] -> each tile's 256K stream bits, int64 [n, 256K]: int16 words, the two words of
    each pair swapped, bits MSB first."""
    n, nw = raw.shape[0], int(16 * K)
    b = raw.reshape(n, nw, 2).long()
    words = (b[..., 0] | (b[..., 1] << 8)).reshape(n, nw // 2, 2).flip(-1).reshape(n, nw)
    return ((words.unsqueeze(-1) >> torch.arange(15, -1, -1, device=raw.device)) & 1).reshape(n, 16 * nw)


def ring_states(bits, ends, width):
    """The width-bit states whose last bit is at stream bit ends - 1, reading the stream as a ring (tail-biting)."""
    idx = (ends.unsqueeze(1) - width + torch.arange(width, device=bits.device)) % bits.shape[1]
    return (bits[:, idx] << torch.arange(width - 1, -1, -1, device=bits.device)).sum(-1)


def decode_tiles(raw, K, cb="mcg"):
    """Packed tiles, uint8 [n, 32K] -> fp32 [n, 16, 16], (input, output) within each tile. K may be a half-integer
    for a pair codebook."""
    n, dev = raw.shape[0], raw.device
    bits = tile_bits(raw, K)
    if is_pair(cb):         # pair p: the 20 bits ending at bit 2K(p + 1)
        vals = codebook(ring_states(bits, (torch.arange(128, device=dev) + 1) * int(2 * K), 20), cb).reshape(n, 256)
    else:                   # value i: the 16 bits ending at bit K(i + 1)
        vals = codebook(ring_states(bits, (torch.arange(256, device=dev) + 1) * int(K), 16), cb)
    out = torch.empty(n, 256, device=dev)
    out[:, tile_perm().to(dev)] = vals
    return out.view(n, 16, 16)


def tiles_out_major(raw, K, rows, cols, cb="mcg"):
    """A run of tiles stored [k16][c = 0..7] (gate/up halves, dense entries on the output side) -> (rows, 128)."""
    return decode_tiles(raw.view(-1, int(32 * K)), K, cb).view(rows // 16, cols // 16, 16, 16) \
        .permute(0, 2, 1, 3).reshape(rows, cols)


def tiles_in_major(raw, K, cols, cb="mcg"):
    """A run of tiles stored [b][r = 0..7][c = 0..7] (down halves, dense entries on the input side) -> (128, cols)."""
    return decode_tiles(raw.view(-1, int(32 * K)), K, cb).view(cols // 128, 8, 8, 16, 16) \
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


def parse_width(text):
    """A width as written in metadata: 4 or 4.5."""
    k = float(text)
    return int(k) if k == int(k) else k


def _widths(text):
    return [parse_width(k) for k in text.split(",") if k]


def check_codebook_text(name, text):
    """Raises ValueError unless name is a codebook this loader decodes and text (its `codebook` metadata) starts
    with it and names its multipliers."""
    if name not in CODEBOOKS:
        raise ValueError(f"codebook {name!r} is not one of {sorted(CODEBOOKS)}")
    first = text.replace(",", " ").split()[0] if text.strip() else ""
    missing = [m for m in CODEBOOKS[name][1] if m.lower() not in text.lower()]
    if first != name or missing:
        raise ValueError(f"codebook metadata {text!r} does not name {name} and its multipliers")


def file_codebook(f):
    """A trellis file's codebook, from its tensor names: `trellis.gate` is mcg, `trellis.gate.<codebook>` names it,
    and `trellis.gate.mix` marks a mixed file ("mix"). Checks the `codebook` metadata against the names."""
    found = set()
    for n in f.names():
        parts = n.removeprefix("shared.").split(".")
        if parts[0] == "trellis" and parts[1] in PROJS:
            found.add(parts[2] if len(parts) > 2 else "mcg")
    if len(found) != 1:
        raise ValueError(f"{f.path}: trellis tensors name codebooks {sorted(found)}")
    cb = found.pop()
    text = f.meta.get("codebook")
    if cb == "mix":
        if text != "mix":
            raise ValueError(f"{f.path}: .mix tensors, but metadata codebook {text!r}")
    elif cb in CODEBOOKS:
        if text is not None or cb != "mcg":     # mcg files written before the other codebooks carry none
            check_codebook_text(cb, text or "")
    else:
        raise ValueError(f"{f.path}: codebook {cb!r} is not one of {sorted(CODEBOOKS)}")
    return cb


def mixed_codebooks(f):
    """A mixed file's codebook names in id order, from its `codebooks` metadata."""
    names = []
    for entry in f.meta["codebooks"].split(";"):
        entry = entry.strip()
        name = entry.replace(",", " ").split()[0]
        check_codebook_text(name, entry)
        names.append(name)
    return names


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
        check_format(self.index)
        self.widths = _widths(meta["widths"]) + _stores(meta.get("stores"))
        self.shared_widths = _widths(meta["shared_widths"]) + _stores(meta.get("shared_stores"))
        self.files = {}
        self.base = base

    def trellis_file(self, K):
        """(file, codebook, mixed) for width K: mixed is None, or for a mixed file {"names": codebook by id,
        "": routed ids [E], "shared.": the shared expert's id [1]}."""
        if K not in self.files:
            f = SafeFile(f"{self.base}.{width_name(K)}.safetensors")
            v4 = check_format(f) == "exlr8-v4"
            cb = file_codebook(f)
            mixed = None
            if cb == "mix":
                mixed = {"names": mixed_codebooks(f)}
                for pre in ("", "shared."):
                    if f"{pre}trellis.codebook" in f.header:
                        ids = f.get(f"{pre}trellis.codebook")
                        if int(ids.max()) >= len(mixed["names"]):
                            raise ValueError(f"{f.path}: {pre}trellis.codebook has an id with no codebook")
                        mixed[pre] = ids
            if not v4 and (mixed or cb.endswith("-f16") or K not in (2, 3, 4)):
                raise ValueError(f"{f.path}: a v4 feature (width K{K}, codebook {cb}) in an exlr8-v2 file")
            self.files[K] = (f, cb, mixed)
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
        k = f"{pre}{width_name(K)}."
        off, length = (int(v) for v in self.t[k + "index"][j, PROJS.index(p), h])
        f, cb, mixed = self.trellis_file(K)
        name = f"{pre}trellis.{p}" + ("" if cb == "mcg" else "." + cb)
        a, b = (f.start + x for x in f.header[name]["data_offsets"])
        if not (a <= off and off + length <= b and length == int(16 * K * self.H)):
            raise ValueError(f"{f.path}: layer {self.L} expert {e} {p} unit {h} lies outside {name}")
        extra = []
        if mixed:           # the expert's own codebook; its id joins the checksum
            cid = int(mixed[pre][j])
            cb, extra = mixed["names"][cid], [torch.tensor([cid], dtype=torch.uint8)]
        check_width(K, cb)
        raw = f.read(off, length)
        c = slice(128 * h, 128 * h + 128)
        if p == "down":
            chan, block = self.t[k + "down.suh"][j, c], self.t[k + "down.sv_block"][j]
        else:
            chan, block = self.t[k + f"{p}.svh"][j, c], self.t[k + f"{p}.su_block"][j]
        if verify:
            want = int(self.t[k + "checksum"][j, PROJS.index(p), h]) & (2 ** 64 - 1)
            got = xxh64(raw, chan, block, *extra)
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

def decode_dense(rec, t, device="cpu", cb="mcg"):
    """One dense matrix as an HF weight (out, in) fp32. rec: the 8-field record, t: {suffix: tensor} as stored, with
    a trellis matrix's data under `trellis` whatever its codebook cb."""
    fmt, out, inp, K, side = (int(x) for x in rec[:5])
    if fmt not in DATA:
        raise ValueError(f"dense format {fmt} is not one of {sorted(DATA)}")
    K = K if K >= 0 else -K / 2      # a half width is written as -2K
    t = {k: v.to(device) for k, v in t.items()}
    if fmt == ROWS:         # tile t of row r: values 256t.. in stream order, rotated along the row only
        check_width(K, cb)
        w, perm, had = torch.empty(out, inp, device=device), tile_perm().to(device), hadamard(128, device)
        suh, svh = t["suh"].float().unsqueeze(0), t["svh"].float().unsqueeze(1)
        for r in range(0, out, 4096):    # bounds decode_tiles' scratch, tens of bytes a value
            rows = t["rows"][r:r + 4096]
            n = rows.shape[0] * inp // 256
            v = decode_tiles(rows.reshape(n, int(32 * K)), K, cb).view(n, 256)[:, perm]
            w[r:r + rows.shape[0]] = (v.reshape(-1, 128) @ had).view(-1, inp) * suh * svh[r:r + rows.shape[0]]
        return w
    if fmt == TRELLIS:
        check_width(K, cb)
        ent = t["trellis"]
        if side == IN:      # entry g = input rows 128g.., tiles as down halves
            T = torch.cat([tiles_in_major(ent[g], K, out, cb) for g in range(ent.shape[0])], 0)
        else:               # entry g = output columns 128g.., tiles as gate/up halves
            T = torch.cat([tiles_out_major(ent[g], K, inp, 128, cb) for g in range(ent.shape[0])], 1)
        return reconstruct(T, t["suh"], t["svh"]).T
    if fmt == INT4:
        q4, s4 = t["q4"], t["s4"]
        if side == IN:      # stored [in/128, out, 64] and [in/128, out, 4]
            q4, s4 = q4.transpose(0, 1).reshape(out, -1), s4.transpose(0, 1).reshape(out, -1)
        n = torch.stack([(q4 & 15).int() - 8, (q4 >> 4).int() - 8], -1).reshape(out, inp).float()
        return n * s4.float().repeat_interleave(32, 1)
    if fmt == INT8:         # K holds the group size G
        G = int(K)
        if G not in (16, 32, 64, 128) or inp % (128 if side == IN else G):
            raise ValueError(f"int8 group size {G} with {inp} inputs (side {side})")
        q8, s8 = t["q8"], t["s8"]
        if side == IN:      # stored [in/128, out, 128] and [in/128, out, 128/G]
            q8, s8 = q8.transpose(0, 1).reshape(out, inp), s8.transpose(0, 1).reshape(out, inp // G)
        if (q8 == -128).any():
            raise ValueError("int8 code -128 is not a code")
        s = s8.float()
        if not torch.isfinite(s).all() or (s < 0).any():
            raise ValueError("int8 scales must be finite and >= 0")
        return q8.float() * s.repeat_interleave(G, 1)
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
        elif fmt == ROWS:
            parts = [t["rows"][r], t["svh"][r], t["suh"]]
        elif fmt in (INT4, FP8_B32, INT8) and side == IN:
            parts = [t[k][b] for k in DATA[fmt]]
        elif fmt == FP8_B32:
            parts = [t["fp8"][r], t["e8m0"][4 * b:4 * b + 4]]
        elif fmt == FP8_BLOCK:
            parts = [t["fp8"][r], t["scale"][b]]
        else:
            parts = [t[k][r] for k in DATA[fmt]]
        sums.append(xxh64(*parts))
    return sums


def dense_matrices(f, widths=None):
    """{stored name: (base, codebook, {data suffix: tensor name})} for every matrix of a dense or vocabulary file,
    and the set of every tensor name a matrix owns, unpicked widths included. base prefixes the matrix's tensors: `M`,
    or `M.kK` for a matrix stored at several widths (FORMAT.md "Dense widths"), K picked by widths, {stored name: K}
    from the manifest's `dense` object."""
    names = set(f.names())
    stored = {}
    for n in names:
        if not n.endswith(".record"):
            continue
        base = n[:-len(".record")]
        m, _, w = base.rpartition(".")
        if m and re.fullmatch(r"k\d+(p5)?", w) and f"{m}.record" not in names:
            stored.setdefault(m, {})[w] = base
        else:
            stored[base] = {None: base}
    widths = widths or {}
    extra = set(widths) - {m for m, ws in stored.items() if None not in ws}
    if extra:
        raise ValueError(f"{f.path}: the manifest picks widths for {sorted(extra)}, which have one width")
    texts = [x.strip() for x in f.meta.get("codebooks", "").split(";") if x.strip()]
    listed = {x.replace(",", " ").split()[0]: x for x in texts}

    def tensors(base):
        """base's codebook and {data suffix: tensor name}, checked against the `codebooks` metadata."""
        fmt = int(f.get(base + ".record")[0])
        codes = "rows" if fmt == ROWS else "trellis"
        found = [n[len(f"{base}.{codes}."):] for n in names if n.startswith(f"{base}.{codes}.")]
        cb = found[0] if found else "mcg"
        if found and (len(found) > 1 or cb not in listed):
            raise ValueError(f"{f.path}: {base} names codebooks {found}, metadata codebooks lists {sorted(listed)}")
        if found:
            check_codebook_text(cb, listed[cb])
        return cb, {s: f"{base}.{s}.{cb}" if s == codes and found else f"{base}.{s}" for s in DATA[fmt]}

    mats, owned = {}, set()
    for m, ws in stored.items():
        for base in ws.values():
            owned.update([base + ".record", base + ".checksum", *tensors(base)[1].values()])
        if None in ws:
            base = ws[None]
        elif m not in widths:
            raise ValueError(f"{f.path}: {m} is stored at {sorted(ws)} and the manifest picks none")
        elif width_name(widths[m]) not in ws:
            raise ValueError(f"{f.path}: the manifest picks {width_name(widths[m])} for {m}, stored at {sorted(ws)}")
        else:
            base = ws[width_name(widths[m])]
        mats[m] = (base, *tensors(base))
    return mats, owned


def load_matrices(f, prefix, dtype, device, verify, raw=False, widths=None):
    """Every tensor of a dense or vocabulary file: matrices decoded (by their records), vectors as stored. widths:
    the manifest's `dense` entry for this file. With raw, FP8 32x32 block matrices come back as the source's
    (weight, scale), MXFP8 as (weight, scale) too, and BF16 as stored."""
    mats, owned = dense_matrices(f, widths)
    out = {}
    for m, (base, cb, data) in mats.items():
        rec = f.get(base + ".record")
        fmt, side = int(rec[0]), int(rec[4])
        t = {s: f.get(n) for s, n in data.items()}
        if verify:
            want = [int(v) & (2 ** 64 - 1) for v in f.get(base + ".checksum")]
            if dense_checksums(rec, t) != want:
                raise ValueError(f"{f.path}: {base} checksum mismatch")
        if raw and fmt in (FP8_B32, MXFP8):
            q, s = fp8_b32_rows(t, side) if fmt == FP8_B32 else (t["fp8"], t["e8m0"])
            out[prefix(m) + ".weight"] = q.view(torch.float8_e4m3fn)
            out[prefix(m) + ".scale"] = s.view(torch.float8_e8m0fnu)
        elif raw and fmt == BF16:
            out[prefix(m) + ".weight"] = t["bf16"]
        else:
            out[prefix(m) + ".weight"] = decode_dense(rec, t, device, cb).to(dtype).cpu()
    for n in f.names():
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

def check_format(f):
    """A safetensors file's `format`, refused unless this loader reads it."""
    fmt = f.meta.get("format")
    if fmt not in FORMATS:
        raise ValueError(f"{f.path}: format {fmt!r} is not one of {FORMATS}")
    return fmt


def check_requires(manifest):
    """Refuses a v4 manifest whose `requires` names a width, codebook, store or mixing level this loader does not
    decode, from that object alone (FORMAT.md "requires"). A v2 manifest has no `requires`."""
    fmt = manifest.get("format")
    if fmt == "exlr8-v2-manifest":
        return
    if fmt != "exlr8-v4-manifest":
        raise ValueError(f"manifest format {fmt!r} is not exlr8-v2-manifest or exlr8-v4-manifest")
    req = manifest.get("requires")
    if req is None:
        raise ValueError("a v4 manifest without requires")
    for item in req["trellis"]:
        wname, cb = item.split()
        if cb not in CODEBOOKS:
            raise ValueError(f"requires {item!r}: codebook {cb} is not one of {sorted(CODEBOOKS)}")
        K = parse_width(wname.removeprefix("k").replace("p5", ".5"))
        if width_name(K) != wname:
            raise ValueError(f"requires {item!r}: {wname} is not a width name")
        check_width(K, cb)
    for store in req["stores"]:
        if store not in NATIVE:
            raise ValueError(f"requires store {store!r}, not one of {sorted(NATIVE)}")
    if req["mixing"] not in MIXING:
        raise ValueError(f"requires mixing {req['mixing']!r}, not one of {MIXING}")


def load_manifest(root, name):
    path = name if os.path.isfile(name) else os.path.join(root, "manifests", f"exlr8-manifest-{name}.json")
    with open(path) as f:
        m = json.load(f)
    check_requires(m)
    return m


def load_layer(root, manifest, L, dtype=torch.bfloat16, device="cpu", verify=False, raw=False):
    """Every tensor of layer L under its HF names; experts at the manifest's widths. With raw, experts in a native
    store come back as the source's (weight, scale)."""
    f = SafeFile(os.path.join(root, "dense", f"exlr8-dense-{L:03d}.safetensors"))
    pre = f.meta.get("hf_prefix") or f"{f.meta.get('tensor_prefix', GLM53['tensor_prefix'])}{L}."
    widths = manifest.get("dense", {}).get(str(L))
    out = load_matrices(f, lambda n: pre + n, dtype, device, verify, raw, widths)
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
    ap = argparse.ArgumentParser(description="Decode one exlr8 v2 or v4 layer into HF tensors.")
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
