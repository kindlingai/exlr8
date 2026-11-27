# Tests for the reference loader against FORMAT.md's test values.
#
#     python -m pytest loader/test_exlr8.py        (needs torch, xxhash and pytest)
#
# The codebook values, table hashes and tile hashes are FORMAT.md "Test values"; gr8's own implementation
# (engine/loader-rs) produces the same ones.
import hashlib
import json
import os
import struct
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import exlr8 as x    # noqa: E402

# state -> ([v as fp16 hex], [v rounded to E4M3, as fp16 hex]); pmcgx states give two values
STATES = {
    "mcg": [[0, ["3f60"], ["3f80"]], [1, ["304e"], ["3080"]], [4660, ["3e0f"], ["3e00"]], [32767, ["4172"], ["4180"]],
            [32768, ["b915"], ["b900"]], [48879, ["c293"], ["c280"]], [65535, ["3acd"], ["3b00"]]],
    "mul1": [[0, ["c2e8"], ["c300"]], [1, ["3921"], ["3900"]], [4660, ["b89e"], ["b880"]], [32767, ["3fca"], ["4000"]],
             [32768, ["baf2"], ["bb00"]], [48879, ["3124"], ["3100"]], [65535, ["b936"], ["b900"]]],
    "pmcgx": [[0, ["3f60", "3f60"], ["3f80", "3f80"]], [1, ["b268", "34cb"], ["b280", "3500"]],
              [74565, ["3e6d", "c0b4"], ["3e80", "c080"]], [524287, ["c120", "1e00"], ["c100", "1e00"]],
              [524288, ["3ea4", "395c"], ["3e80", "3980"]], [782069, ["bd2e", "4070"], ["bd00", "4080"]],
              [1048575, ["b9b0", "ac60"], ["b980", "ac80"]]],
    "pmul1x": [[0, ["c2e8", "c2e8"], ["c300", "c300"]], [1, ["3e8b", "b9a5"], ["3e80", "b980"]],
               [74565, ["385f", "2c37"], ["3880", "2c00"]], [524287, ["3af8", "3990"], ["3b00", "3980"]],
               [524288, ["bb98", "bef0"], ["bb80", "bf00"]], [782069, ["ae2e", "9770"], ["ae00", "9800"]],
               [1048575, ["ad50", "3974"], ["ad80", "3980"]]],
}
# mul1-f16: state -> the integer byte_sum - 510; the weight is that times C_MUL1
MUL1_F16 = {0: -510, 1: 95, 4660: -85, 32767: 288, 32768: -128, 48879: 24, 65535: -96}
# sha256 of the table of those integers, as little-endian i16, over every state
MUL1_F16_TABLE = "85abd370a7531193db8b9df21a70a0a4dd870464ea74d7f33fff5db3213b929b"
# pmul1x-f16: state -> the integers byte_sum - 510 of weights 2p and 2p + 1, and the sha256 of their table
PMUL1X_F16 = {0: [-510, -510], 1: [242, -104], 74565: [81, 10], 524287: [129, 103], 524288: [-140, -256],
              782069: [-14, 0], 1048575: [-12, 101]}
PMUL1X_F16_TABLE = "1371509153e88d27e56616f2aadff81a2c3e3c5043f8fd1829606198cc9cd9a8"
# sha256 of the fp16 table and of the E4M3 table (one byte per value)
TABLES = {
    "mcg": ("95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9",
            "852259a151b9d7c4d8cc6b7537ba5f548e78681488e926953c04e94b275591b9"),
    "mul1": ("bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8",
             "ee3283e85586daec879a03d2cf93b7f33e8f3400b1c2b55e9eb4563f152a45cc"),
    "pmcgx": ("c2f33d5515f86e4c296867d9ca4f6871f8c1441248b24410bebc13e49da442a5",
              "7f66c49ed68c726cc5b602dc7df7788e06662000893b5fa52d4b042a834be07e"),
    "pmul1x": ("c7f10379ba9c11052b774bdac2a53e67c21f26fa57dc2a9e9bd301129c289388",
               "b6973dd205d4bc1b87469240f73cac8c05898ac7e12fd37abcb7e0411928d300"),
}
# first 16 hex digits of the sha256 of the decoded tile with byte j = (37j + 11) mod 256
TILES = {
    ("mcg", 1): "0f55b1ebabddf3dc", ("mcg", 2): "8dde4d941e508fa7", ("mcg", 3): "640c2065964c409d",
    ("mcg", 4): "43b9bff37b64f579", ("mcg", 5): "cc48fcc66caf3a41",
    ("mul1", 1): "24848da1282fbd70", ("mul1", 2): "5828c7200d079dad", ("mul1", 3): "eaa28e21af3b58e2",
    ("mul1", 4): "e24c0798dc46e799", ("mul1", 5): "3d1043cb44bdf76d",
    ("pmcgx", 0.5): "dd6bc537b173f3e8", ("pmcgx", 1): "f0c40ae8fe7f23d1", ("pmcgx", 1.5): "36a1897a31ccd5d4",
    ("pmcgx", 2): "d5b6cc4ed594469f", ("pmcgx", 2.5): "1428ff10cbc4044b", ("pmcgx", 3): "d4005af0b8591dc1",
    ("pmcgx", 3.5): "b9898c3e54ae2d7e", ("pmcgx", 4): "e407108ab19f224c", ("pmcgx", 4.5): "143277248d13e27c",
    ("pmcgx", 5): "c037dde67a1f5de4",
    ("mcg-f16", 6): "626bfe33614027e1", ("mul1-f16", 6): "8194763e86479b2e",
    ("pmcgx-f16", 5.5): "8d06db3bf149c785", ("pmcgx-f16", 6): "b888e32d22699015",
    ("pmul1x", 0.5): "2d5246bb48deff89", ("pmul1x", 1): "8025036245972646", ("pmul1x", 1.5): "37f9be48867ea152",
    ("pmul1x", 2): "98da956a07adbc79", ("pmul1x", 2.5): "f4946c25b34b2fe0", ("pmul1x", 3): "7a72d1dd5f2c8478",
    ("pmul1x", 3.5): "90a0efed5e77528e", ("pmul1x", 4): "79d9ba536e499ff0", ("pmul1x", 4.5): "62c17e5462268df4",
    ("pmul1x", 5): "8829e7991f56a46c",
    ("pmul1x-f16", 5.5): "77a36b43b71e4c6b", ("pmul1x-f16", 6): "aebf5b38d2a71e33",
}
WIDTHS = {"mcg": [1, 2, 3, 4, 5], "pmcgx": [k / 2 for k in range(1, 11)],
          "mcg-f16": [1, 2, 3, 4, 5, 6], "mul1-f16": [1, 2, 3, 4, 5, 6], "pmcgx-f16": [k / 2 for k in range(1, 13)],
          "pmul1x": [k / 2 for k in range(1, 11)], "pmul1x-f16": [k / 2 for k in range(1, 13)]}


def hex16(t):
    return [f"{int(v) & 0xFFFF:04x}" for v in t.reshape(-1).view(torch.int16)]


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- codebooks

@pytest.mark.parametrize("cb", sorted(STATES))
def test_codebook_states(cb):
    for state, v, rounded in STATES[cb]:
        s = torch.tensor([state])
        assert hex16(x.codebook_f16(s, cb)) == v
        assert hex16(x.codebook(s, cb).half()) == rounded
        if cb not in x.BYTE_SUM:
            assert hex16(x.codebook(s, cb + "-f16").half()) == v


def test_mul1_f16_states():
    for state, n in MUL1_F16.items():
        assert x.codebook(torch.tensor([state]), "mul1-f16").item() == n * x.C_MUL1


def test_mul1_f16_table():
    states = torch.arange(1 << 16)
    n = x.mul1_byte_sum(states) - 510
    assert sha(n.to(torch.int16).numpy().tobytes()) == MUL1_F16_TABLE
    # exact in fp32, and within half an fp16 ulp plus mul1's 0.27 c offset of mul1's fp16 value
    v = x.codebook(states, "mul1-f16")
    assert torch.equal(v.double(), n.double() * x.C_MUL1)
    assert (v - x.codebook_f16(states, "mul1").float()).abs().max() < 0.003


@pytest.mark.parametrize("cb", sorted(TABLES))
def test_codebook_tables(cb):
    states = torch.arange(1 << x.CODEBOOKS[cb][0])
    f16 = x.codebook_f16(states, cb).reshape(-1).contiguous()
    e4 = x.codebook(states, cb).to(torch.float8_e4m3fn).reshape(-1).contiguous()
    assert sha(f16.view(torch.int16).numpy().tobytes()) == TABLES[cb][0]
    assert sha(e4.view(torch.uint8).numpy().tobytes()) == TABLES[cb][1]
    if cb not in x.BYTE_SUM:
        assert torch.equal(x.codebook(states, cb + "-f16").reshape(-1), f16.float())


def test_pmul1x_f16_states():
    for state, n in PMUL1X_F16.items():
        assert x.codebook(torch.tensor([state]), "pmul1x-f16").reshape(-1).tolist() == [k * x.C_MUL1 for k in n]


def test_pmul1x_f16_table():
    states = torch.arange(1 << 20)
    n = x.pmul1x_byte_sums(states) - 510
    assert sha(n.reshape(-1).to(torch.int16).numpy().tobytes()) == PMUL1X_F16_TABLE
    assert torch.equal(x.codebook(states, "pmul1x-f16").double(), n.double() * x.C_MUL1)


def test_pmul1x_gpu_form():
    """The decode FORMAT.md suggests for pmul1x: PRMT 0x7610 / 0x5432 picks each weight's four bytes from x1 and x2,
    and a dp4a accumulating into 0x6400 gives fp16(1024 + byte sum). Emulated on integers, every state."""
    w = torch.arange(1 << 20)
    x1, x2 = (w * x.PMUL1X[0]) & 0xFFFFFFFF, (w * x.PMUL1X[1]) & 0xFFFFFFFF

    def prmt(a, b, sel):
        src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
        return sum(src[(sel >> (4 * i)) & 7] << (8 * i) for i in range(4))

    def dp4a_6400(v):
        return sum((v >> (8 * i)) & 0xFF for i in range(4)) + 0x6400
    got = torch.stack([x._fp16(dp4a_6400(prmt(x1, x2, sel))) for sel in (0x7610, 0x5432)], -1).float() - 1534
    assert torch.equal(got, (x.pmul1x_byte_sums(w) - 510).float())


# ---------------------------------------------------------------- tiles

def tile_bytes(K):
    n = int(32 * K)
    return torch.tensor([(j * 37 + 11) % 256 for j in range(n)], dtype=torch.uint8).reshape(1, n)


@pytest.mark.parametrize("cb,K", sorted(TILES))
def test_tile_vectors(cb, K):
    v = x.decode_tiles(tile_bytes(K), K, cb).reshape(-1)
    assert sha(v.numpy().tobytes())[:16] == TILES[(cb, K)]


def spec_decode(raw, K, cb):
    """One tile decoded one bit at a time, as FORMAT.md "Decode" words it: the slow, plain form of decode_tiles."""
    words = [raw[2 * i] | (raw[2 * i + 1] << 8) for i in range(len(raw) // 2)]
    words = [words[i ^ 1] for i in range(len(words))]                  # the two words of each pair swapped
    bits = [(w >> (15 - b)) & 1 for w in words for b in range(16)]     # MSB first
    n = len(bits)
    assert n == 256 * K

    def state(end, width):          # the width bits whose last bit is end - 1, wrapping
        s = 0
        for i in range(end - width, end):
            s = (s << 1) | bits[i % n]
        return s
    if x.is_pair(cb):
        vals = []
        for p in range(128):
            vals += x.codebook(torch.tensor([state(int(2 * K) * (p + 1), 20)]), cb).reshape(-1).tolist()
    else:
        vals = [x.codebook(torch.tensor([state(int(K) * (i + 1), 16)]), cb).item() for i in range(256)]
    out = [0.0] * 256
    for j, pos in enumerate(x.tile_perm().tolist()):
        out[pos] = vals[j]
    return torch.tensor(out).view(16, 16)


@pytest.mark.parametrize("cb", sorted(WIDTHS))
def test_tiles_match_the_spec_wording(cb):
    g = torch.Generator().manual_seed(7)
    for K in WIDTHS[cb]:
        raw = torch.randint(0, 256, (2, int(32 * K)), dtype=torch.uint8, generator=g)
        got = x.decode_tiles(raw, K, cb)
        for t in range(2):
            assert torch.equal(got[t], spec_decode(raw[t].tolist(), K, cb)), (cb, K)


# ---------------------------------------------------------------- widths and manifests

def test_widths():
    assert [x.width_name(k) for k in (0.5, 1, 4, 4.5)] == ["k0p5", "k1", "k4", "k4p5"]
    for cb, ks in WIDTHS.items():
        for k in ks:
            x.check_width(k, cb)
    for k, cb in ((4.5, "mcg"), (0.5, "mul1"), (6, "mcg"), (6, "pmcgx"), (6.5, "pmcgx-f16"), (0, "pmcgx")):
        with pytest.raises(ValueError):
            x.check_width(k, cb)


def requires(**kw):
    return {"format": "exlr8-v4-manifest", "requires": {"trellis": [], "stores": [], "mixing": "none", **kw}}


def test_requires():
    x.check_requires({"format": "exlr8-v2-manifest"})
    x.check_requires(requires(trellis=["k2 mul1", "k4p5 pmcgx", "k6 mcg-f16", "k0p5 pmcgx"], stores=["mxfp4"],
                              mixing="expert"))
    for bad in (requires(trellis=["k4p5 mcg"]), requires(trellis=["k6 mul1"]), requires(trellis=["k4 nope"]),
                requires(trellis=["k4.5 pmcgx"]), requires(stores=["int3"]), requires(mixing="some"),
                {"format": "exlr8-v4-manifest"}, {"format": "exlr8-v3-manifest"}):
        with pytest.raises(ValueError):
            x.check_requires(bad)


# ---------------------------------------------------------------- a mixed width file, end to end

ST_DTYPES = {torch.int64: "I64", torch.int8: "I8", torch.uint8: "U8", torch.float16: "F16",
             torch.bfloat16: "BF16", torch.float32: "F32"}


def write_st(path, tensors, meta):
    """A safetensors file with metadata; U64 tensors are passed as ("U64", int64 tensor)."""
    header, blobs, off = {"__metadata__": meta}, [], 0
    for name, t in tensors.items():
        dt, t = t if isinstance(t, tuple) else (ST_DTYPES[t.dtype], t)
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return 8 + len(raw), header


H, I, E = 128, 128, 2
MIXED = ["pmcgx", "pmcgx-f16"]
IDS = [1, 0]            # expert 0 decodes with pmcgx-f16, expert 1 with pmcgx


def build(root, K, wfile_meta, ids=IDS, checksum_ids=True):
    """Layer 3 with one width K: a mixed file whose experts use MIXED[IDS[e]]."""
    os.makedirs(os.path.join(root, "experts"), exist_ok=True)
    base = os.path.join(root, "experts", "exlr8-layer-003")
    g = torch.Generator().manual_seed(3)
    B = int(16 * K * H)
    trel = {f"trellis.{p}.mix": torch.randint(0, 256, (E * B,), dtype=torch.uint8, generator=g) for p in x.PROJS}
    start, hdr = write_st(f"{base}.{x.width_name(K)}.safetensors",
                          {**trel, "trellis.codebook": torch.tensor(ids, dtype=torch.uint8)}, wfile_meta)
    w = x.width_name(K)
    side = {f"{w}.gate.su_block": torch.rand(E, H // 128, generator=g).half() + 0.5,
            f"{w}.up.su_block": torch.rand(E, H // 128, generator=g).half() + 0.5,
            f"{w}.down.sv_block": torch.rand(E, H // 128, generator=g).half() + 0.5,
            f"{w}.gate.svh": torch.randn(E, I, generator=g).half(), f"{w}.up.svh": torch.randn(E, I, generator=g).half(),
            f"{w}.down.suh": torch.randn(E, I, generator=g).half()}
    index = torch.zeros(E, 3, 1, 2, dtype=torch.int64)
    checks = torch.zeros(E, 3, 1, dtype=torch.int64)
    for e in range(E):
        for pi, p in enumerate(x.PROJS):
            off = start + hdr[f"trellis.{p}.mix"]["data_offsets"][0] + e * B
            index[e, pi, 0] = torch.tensor([off, B])
            raw = trel[f"trellis.{p}.mix"][e * B:(e + 1) * B]
            chan = side[f"{w}.down.suh"][e] if p == "down" else side[f"{w}.{p}.svh"][e]
            block = side[f"{w}.down.sv_block" if p == "down" else f"{w}.{p}.su_block"][e]
            extra = [torch.tensor([ids[e]], dtype=torch.uint8)] if checksum_ids else []
            c = x.xxh64(raw, chan, block, *extra)
            checks[e, pi, 0] = c - 2 ** 64 if c >= 2 ** 63 else c        # stored as U64, held here as int64
    signs = {"signs.in": (torch.randint(0, 2, (H,), generator=g) * 2 - 1).to(torch.int8),
             "signs.out": (torch.randint(0, 2, (H,), generator=g) * 2 - 1).to(torch.int8)}
    write_st(f"{base}.index.safetensors", {**signs, **side, f"{w}.index": index, f"{w}.checksum": ("U64", checks)},
             {"format": "exlr8-v4", "widths": str(K), "shared_widths": "", "hidden_size": str(H),
              "moe_intermediate_size": str(I), "n_routed_experts": str(E)})
    return trel, side, signs


MIX_META = {"format": "exlr8-v4", "codebook": "mix", "k": "4.5",
            "codebooks": "pmcgx V2 L20 0xCBAC1FED 0xBCA14C51, E4M3 rounding; "
                         "pmcgx-f16 V2 L20 0xCBAC1FED 0xBCA14C51, fp16"}


def test_mixed_file(tmp_path):
    K = 4.5
    trel, side, signs = build(str(tmp_path), K, MIX_META)
    layer = x.ExpertLayer(str(tmp_path), 3)
    w, B = x.width_name(K), int(16 * K * H)
    for e in range(E):
        cb = MIXED[IDS[e]]
        for p in x.PROJS:
            got = layer.half(e, p, 0, K, verify=True)
            raw = trel[f"trellis.{p}.mix"][e * B:(e + 1) * B]
            if p == "down":
                T = x.tiles_in_major(raw, K, H, cb)
                svh = signs["signs.out"].float() * side[f"{w}.down.sv_block"][e].float().repeat_interleave(128)
                want = x.reconstruct(T, side[f"{w}.down.suh"][e], svh)
            else:
                T = x.tiles_out_major(raw, K, H, 128, cb)
                suh = signs["signs.in"].float() * side[f"{w}.{p}.su_block"][e].float().repeat_interleave(128)
                want = x.reconstruct(T, suh, side[f"{w}.{p}.svh"][e])
            assert torch.equal(got, want), (e, p)
    # the other expert's codebook gives other weights: the id really selects the decode
    assert not torch.equal(x.decode_tiles(trel["trellis.gate.mix"][:144].view(1, 144), K, "pmcgx"),
                           x.decode_tiles(trel["trellis.gate.mix"][:144].view(1, 144), K, "pmcgx-f16"))


def test_mixed_checksum_needs_the_id(tmp_path):
    build(str(tmp_path), 4.5, MIX_META, checksum_ids=False)
    with pytest.raises(ValueError, match="checksum"):
        x.ExpertLayer(str(tmp_path), 3).half(0, "gate", 0, 4.5, verify=True)


@pytest.mark.parametrize("meta", [
    {**MIX_META, "format": "exlr8-v2"},                          # a v4 feature under v2 strings
    {**MIX_META, "codebook": "pmcgx"},                           # .mix names, single-codebook metadata
    {**MIX_META, "codebooks": "pmcgx V2 L20 0xCBAC1FED, E4M3"},  # pmcgx without its second multiplier
    {**MIX_META, "codebooks": "mcg 0xCBAC1FED, E4M3 rounding; pmcgx-f16 V2 L20 0xCBAC1FED 0xBCA14C51, fp16"},
])
def test_mixed_file_refused(tmp_path, meta):
    """The last case gives expert 1 mcg at K4.5, which only a pair codebook has."""
    build(str(tmp_path), 4.5, meta)
    with pytest.raises(ValueError):
        layer = x.ExpertLayer(str(tmp_path), 3)
        for e in range(E):
            layer.half(e, "gate", 0, 4.5)


PMCGX_TEXT = "pmcgx V2 L20 0xCBAC1FED 0xBCA14C51, E4M3 rounding"


def dense_trellis(base, K, cb, g):
    """A 256 x 128 trellis matrix on the output side, its tensors under base."""
    out, inp = 256, 128
    ent = torch.randint(0, 256, (out // 128, int(16 * K * inp)), dtype=torch.uint8, generator=g)
    suh, svh = torch.randn(inp, generator=g).half(), torch.randn(out, generator=g).half()
    sums = [x.xxh64(ent[b], svh[128 * b:128 * b + 128], suh) for b in range(out // 128)]
    name = f"{base}.trellis" + ("" if cb == "mcg" else f".{cb}")
    return {f"{base}.record": torch.tensor([x.TRELLIS, out, inp, K, 1, 128, 0, 1]),
            f"{base}.checksum": ("U64", torch.tensor([c - 2 ** 64 if c >= 2 ** 63 else c for c in sums])),
            name: ent, f"{base}.suh": suh, f"{base}.svh": svh}


def dense_file(root, meta):
    """Layer 0 with q_proj at K2 (mcg) and K3 (pmcgx), o_proj at K4 (pmcgx) alone, and a q_proj.bias vector."""
    os.makedirs(os.path.join(root, "dense"), exist_ok=True)
    g = torch.Generator().manual_seed(5)
    t = {**dense_trellis("q_proj.k2", 2, "mcg", g), **dense_trellis("q_proj.k3", 3, "pmcgx", g),
         **dense_trellis("o_proj", 4, "pmcgx", g), "q_proj.bias": torch.randn(256, generator=g).half()}
    write_st(os.path.join(root, "dense", "exlr8-dense-000.safetensors"), t, meta)
    return t


DENSE_META = {"format": "exlr8-v4", "layer": "0", "tensor_prefix": "m.", "codebooks": PMCGX_TEXT}


def test_dense_widths(tmp_path):
    t = dense_file(str(tmp_path), DENSE_META)
    for K, cb in ((2, "mcg"), (3, "pmcgx")):
        got = x.load_layer(str(tmp_path), {"layers": {}, "dense": {"0": {"q_proj": K}}}, 0, torch.float32,
                           verify=True)
        assert sorted(got) == ["m.0.o_proj.weight", "m.0.q_proj.bias", "m.0.q_proj.weight"]
        base = f"q_proj.k{K}"
        data = {"trellis": t[f"{base}.trellis" + ("" if cb == "mcg" else ".pmcgx")],
                "suh": t[f"{base}.suh"], "svh": t[f"{base}.svh"]}
        assert torch.equal(got["m.0.q_proj.weight"], x.decode_dense(t[f"{base}.record"], data, cb=cb))
    assert not torch.equal(x.decode_tiles(data["trellis"][0, :96].view(1, 96), 3, "pmcgx"),
                           x.decode_tiles(data["trellis"][0, :96].view(1, 96), 3, "mcg"))


@pytest.mark.parametrize("dense, meta", [
    ({}, DENSE_META),                                       # q_proj has two widths and no pick
    ({"q_proj": 4}, DENSE_META),                            # a width the file does not hold
    ({"q_proj": 2, "o_proj": 4}, DENSE_META),               # o_proj has one width
    ({"q_proj": 2}, {**DENSE_META, "codebooks": ""}),       # pmcgx tensors, no codebooks metadata
])
def test_dense_widths_refused(tmp_path, dense, meta):
    dense_file(str(tmp_path), meta)
    with pytest.raises(ValueError):
        x.load_layer(str(tmp_path), {"layers": {}, "dense": {"0": dense}}, 0)


def int8_matrix(base, side, G, g, out=256, inp=384):
    """An int8 gG matrix (FORMAT.md "int8 groups") on output side 1 or input side 2, its weight computed directly
    from the row-major codes and scales, and its checksum by the spec's wording. Its format is `int8 gG`, which a
    file describes in `matrices` (int8_desc)."""
    q = torch.randint(-127, 128, (out, inp), dtype=torch.int8, generator=g)
    sc = torch.rand(out, inp // G, generator=g).half()
    sc[0, 0] = 0                                                # a zero scale decodes to zeros
    want = q.float() * sc.float().repeat_interleave(G, 1)
    if side == 2:       # [in/128, out, 128] and [in/128, out, 128/G]
        q8 = q.view(out, inp // 128, 128).transpose(0, 1).contiguous()
        s8 = sc.view(out, inp // 128, 128 // G).transpose(0, 1).contiguous()
        sums = [x.xxh64(q8[b], s8[b]) for b in range(inp // 128)]
    else:
        q8, s8 = q, sc
        sums = [x.xxh64(q8[128 * b:128 * b + 128], s8[128 * b:128 * b + 128]) for b in range(out // 128)]
    t = {f"{base}.record": torch.tensor([0, out, inp, 0, side, 128, 0, 1]),
         f"{base}.checksum": ("U64", torch.tensor([c - 2 ** 64 if c >= 2 ** 63 else c for c in sums])),
         f"{base}.q8": q8, f"{base}.s8": s8}
    return t, want


INT8_META = {"format": "exlr8-v4", "layer": "0", "tensor_prefix": "m."}


def int8_desc(side, G):
    return {"format": f"int8 g{G}", "side": "in" if side == 2 else "out", "unit": 128, "solve": "hessian"}


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("G", [16, 32, 64, 128])
def test_int8_groups(tmp_path, side, G):
    g = torch.Generator().manual_seed(G + side)
    t, want = int8_matrix("o_proj", side, G, g)
    os.makedirs(tmp_path / "dense")
    meta = {**INT8_META, "matrices": json.dumps({"o_proj": int8_desc(side, G)})}
    write_st(str(tmp_path / "dense" / "exlr8-dense-000.safetensors"), t, meta)
    got = x.load_layer(str(tmp_path), {"layers": {}}, 0, torch.float32, verify=True)
    assert torch.equal(got["m.0.o_proj.weight"], want)
    assert not got["m.0.o_proj.weight"][0, :G].any()


@pytest.mark.parametrize("change", ["code", "scale", "nan", "group", "format"])
def test_int8_groups_refused(change):
    g = torch.Generator().manual_seed(3)
    t, _ = int8_matrix("m", 1, 64, g)
    rec, data = t["m.record"].clone(), {"q8": t["m.q8"].clone(), "s8": t["m.s8"].clone()}
    rec[0], rec[3] = x.parse_format("int8 g64")          # as matrix_record gives it
    if change == "code":
        data["q8"][1, 1] = -128
    elif change == "scale":
        data["s8"][1, 1] = -1
    elif change == "nan":
        data["s8"][1, 1] = float("nan")
    elif change == "group":
        rec[3] = 48
    else:
        rec[0] = 11
    with pytest.raises(ValueError):
        x.decode_dense(rec, data)


@pytest.mark.parametrize("text, kind", [("trellis k4", (x.TRELLIS, 4)), ("trellis k4p5", (x.TRELLIS, -9)),
                                        ("row-trellis k2", (x.ROWS, 2)), ("int8 g16", (x.INT8, 16)),
                                        ("int4 g32", (x.INT4, 0)), ("mxfp8", (x.MXFP8, 0)),
                                        ("fp8 b128", (x.FP8_BLOCK, 0)), ("fp8 b32", (x.FP8_B32, 0)),
                                        ("fp8 row", (x.FP8_ROW, 0)), ("bf16", (x.BF16, 0)), ("fp32", (x.FP32, 0))])
def test_format_strings(text, kind):
    assert x.parse_format(text) == kind


@pytest.mark.parametrize("text", ["int8 g48", "int8 g256", "trellis", "MXFP8", "fp8 block", "int4 g64", ""])
def test_format_strings_refused(text):
    with pytest.raises(ValueError):
        x.parse_format(text)


def test_matrices_match_legacy_fields(tmp_path):
    """The same trellis file read by its legacy record fields, and by `matrices` with those fields 0, decodes the
    same."""
    legacy = dense_file(str(tmp_path / "a"), DENSE_META)
    want = x.load_layer(str(tmp_path / "a"), {"layers": {}, "dense": {"0": {"q_proj": 3}}}, 0, torch.float32,
                        verify=True)
    t = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in legacy.items()}
    for k in ("q_proj.k2.record", "q_proj.k3.record", "o_proj.record"):
        t[k][[0, 3, 4, 5, 6, 7]] = 0
    desc = {"side": "out", "unit": 128, "rotation": "own", "solve": "hessian"}
    matrices = {"q_proj.k2": {"format": "trellis k2", **desc}, "q_proj.k3": {"format": "trellis k3", **desc},
                "o_proj": {"format": "trellis k4", **desc}}
    os.makedirs(tmp_path / "b" / "dense")
    write_st(str(tmp_path / "b" / "dense" / "exlr8-dense-000.safetensors"), t,
             {**DENSE_META, "matrices": json.dumps(matrices)})
    got = x.load_layer(str(tmp_path / "b"), {"layers": {}, "dense": {"0": {"q_proj": 3}}}, 0, torch.float32,
                       verify=True)
    assert got.keys() == want.keys() and all(torch.equal(got[k], want[k]) for k in got)


@pytest.mark.parametrize("change", ["missing", "unknown", "extra", "code", "side", "unit", "solve", "v2", "none",
                                    "key", "side-name", "solve-name", "no-unit", "replicated-unit", "rotation"])
def test_matrices_refused(tmp_path, change):
    g = torch.Generator().manual_seed(4)
    t, _ = int8_matrix("o_proj", 1, 64, g)
    d, meta = int8_desc(1, 64), dict(INT8_META)
    matrices = {"o_proj": d}
    if change == "missing":
        matrices = {}
    elif change == "unknown":
        d["format"] = "int8 g48"
    elif change == "extra":
        matrices["q_proj"] = {"format": "bf16", "side": "replicated", "solve": "copy"}
    elif change == "code":
        t["o_proj.record"][0] = x.INT4          # legacy fields must be 0 or agree
    elif change == "side":
        t["o_proj.record"][4] = 2
    elif change == "unit":
        t["o_proj.record"][5] = 64
    elif change == "solve":
        d["solve"] = "rtn"                      # the record says 1 (hessian)
    elif change == "v2":
        meta["format"] = "exlr8-v2"
    elif change == "key":
        d["group"] = 64
    elif change == "side-name":
        d["side"] = "output"
    elif change == "solve-name":
        d["solve"] = "gptq"
    elif change == "no-unit":
        del d["unit"]
    elif change == "replicated-unit":
        d["side"] = "replicated"
        t["o_proj.record"][4] = 0
    elif change == "rotation":
        d["rotation"] = "attn_in"               # int8 has no trellis rotation
    if change != "none":
        meta["matrices"] = json.dumps(matrices)
    os.makedirs(tmp_path / "dense")
    write_st(str(tmp_path / "dense" / "exlr8-dense-000.safetensors"), t, meta)
    with pytest.raises(ValueError):
        x.load_layer(str(tmp_path), {"layers": {}}, 0)


def test_matrices_rotation_names(tmp_path):
    """A trellis matrix's rotation name agrees with legacy field 6 through `rotations`, or the default names."""
    g = torch.Generator().manual_seed(6)
    t = dense_trellis("q_proj", 2, "mcg", g)
    t["q_proj.record"][6] = 2
    for meta, name, ok in (({}, "q_resid", True), ({}, "attn_in", False),
                           ({"rotations": "x, y"}, "y", True), ({"rotations": "x, y"}, "q_resid", False)):
        d = {"format": "trellis k2", "side": "out", "unit": 128, "rotation": name, "solve": "hessian"}
        root = tmp_path / f"{name}{len(meta)}"
        os.makedirs(root / "dense")
        write_st(str(root / "dense" / "exlr8-dense-000.safetensors"), t,
                 {**DENSE_META, **meta, "matrices": json.dumps({"q_proj": d})})
        if ok:
            x.load_layer(str(root), {"layers": {}}, 0, verify=True)
        else:
            with pytest.raises(ValueError):
                x.load_layer(str(root), {"layers": {}}, 0)


@pytest.mark.parametrize("K", [4, 4.5])
def test_row_trellis(tmp_path, K):
    """Each row's tiles decode in stream order and rotate along the row alone (FORMAT.md "Row trellis")."""
    out, inp = 130, 512                # rows need not fill a 128-row block
    g = torch.Generator().manual_seed(9)
    codes = torch.randint(0, 256, (out, inp // 256 * int(32 * K)), dtype=torch.uint8, generator=g)
    suh, svh = torch.randn(inp, generator=g).half(), torch.randn(out, generator=g).half()
    sums = [x.xxh64(codes[128 * b:128 * b + 128], svh[128 * b:128 * b + 128], suh) for b in range(2)]
    t = {"ple.record": torch.tensor([x.ROWS, out, inp, K if K == int(K) else -int(2 * K), 1, 1, 0, 0]),
         "ple.checksum": ("U64", torch.tensor([c - 2 ** 64 if c >= 2 ** 63 else c for c in sums])),
         "ple.rows.pmcgx": codes, "ple.suh": suh, "ple.svh": svh}
    os.makedirs(tmp_path / "dense")
    write_st(str(tmp_path / "dense" / "exlr8-extra-ple.safetensors"), t, {"format": "exlr8-v4", "codebooks": PMCGX_TEXT})
    f = x.SafeFile(str(tmp_path / "dense" / "exlr8-extra-ple.safetensors"))
    got = x.load_matrices(f, lambda s: s, torch.float32, "cpu", True)["ple.weight"]
    tiles = codes.reshape(-1, int(32 * K))
    ends = (torch.arange(128) + 1) * int(2 * K)
    v = x.codebook(x.ring_states(x.tile_bits(tiles, K), ends, 20), "pmcgx").reshape(out, inp)
    want = (v.reshape(-1, 128) @ x.hadamard(128)).view(out, inp) * suh.float() * svh.float().unsqueeze(1)
    assert torch.equal(got, want)


# ---------------------------------------------------------------- multi-group dense releases

KEEP_META = {"format", "contents", "record", "checksum", "scheme", "layer", "group", "model_profile"}


def _u64_i64(sums):
    return ("U64", torch.tensor([c - 2 ** 64 if c >= 2 ** 63 else c for c in sums], dtype=torch.int64))


def _dense_int4(base, out, inp, side, g):
    q4_rm = torch.randint(0, 256, (out, inp // 2), dtype=torch.uint8, generator=g)
    s4_rm = (torch.rand(out, inp // 32, generator=g) * 0.2 + 0.05).half()
    if side == x.IN:
        q4 = q4_rm.reshape(out, -1, 64).transpose(0, 1).contiguous()
        s4 = s4_rm.reshape(out, -1, 4).transpose(0, 1).contiguous()
        sums = [x.xxh64(q4[b], s4[b]) for b in range(inp // 128)]
    else:
        q4, s4 = q4_rm, s4_rm
        sums = [x.xxh64(q4[128 * b:128 * b + 128], s4[128 * b:128 * b + 128]) for b in range(-(-out // 128))]
    rec = torch.tensor([x.INT4, out, inp, 0, side, 128, 0, 1], dtype=torch.int64)
    return {f"{base}.record": rec, f"{base}.checksum": _u64_i64(sums), f"{base}.q4": q4, f"{base}.s4": s4}


def _dense_trellis_side(base, out, inp, K, side, g):
    ent_n = inp // 128 if side == x.IN else out // 128
    ent_b = (out if side == x.IN else inp) * 128 * K // 8
    ent = torch.randint(0, 256, (ent_n, ent_b), dtype=torch.uint8, generator=g)
    suh = (torch.randn(inp, generator=g).sign() * (torch.rand(inp, generator=g) * 0.5 + 0.2)).half()
    svh = (torch.randn(out, generator=g).sign() * (torch.rand(out, generator=g) * 0.5 + 0.2)).half()
    if side == x.IN:
        sums = [x.xxh64(ent[b], suh[128 * b:128 * b + 128], svh) for b in range(ent_n)]
    else:
        sums = [x.xxh64(ent[b], svh[128 * b:128 * b + 128], suh) for b in range(ent_n)]
    rec = torch.tensor([x.TRELLIS, out, inp, K, side, 128, 0, 1], dtype=torch.int64)
    return {f"{base}.record": rec, f"{base}.checksum": _u64_i64(sums),
            f"{base}.trellis": ent, f"{base}.suh": suh, f"{base}.svh": svh}


def _dense_fp8_row(base, out, inp, g):
    w = torch.randn(out, inp, generator=g) * 0.1
    amax = w.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    fp8 = (w / scale.unsqueeze(1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    sums = [x.xxh64(fp8[128 * b:128 * b + 128], scale[128 * b:128 * b + 128]) for b in range(-(-out // 128))]
    rec = torch.tensor([x.FP8_ROW, out, inp, 0, 1, 128, 0, 0], dtype=torch.int64)
    return {f"{base}.record": rec, f"{base}.checksum": _u64_i64(sums), f"{base}.fp8": fp8, f"{base}.scale": scale}


def _dense_bf16(base, out, inp, g):
    bf16 = (torch.randn(out, inp, generator=g) * 0.05).to(torch.bfloat16)
    sums = [x.xxh64(bf16[128 * b:128 * b + 128]) for b in range(-(-out // 128))]
    rec = torch.tensor([x.BF16, out, inp, 0, 0, 0, 0, 0], dtype=torch.int64)
    return {f"{base}.record": rec, f"{base}.checksum": _u64_i64(sums), f"{base}.bf16": ("BF16", bf16)}


def _strip_dir(src, dst):
    """Simulate strip_meta.py: keep only KEEP_META keys, leaving data sections byte-identical."""
    os.makedirs(dst, exist_ok=True)
    for n in sorted(os.listdir(src)):
        if not n.endswith(".safetensors"):
            continue
        with open(os.path.join(src, n), "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            body = f.read()
        hdr["__metadata__"] = {k: v for k, v in sorted(hdr.get("__metadata__", {}).items()) if k in KEEP_META}
        raw = json.dumps(hdr, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        with open(os.path.join(dst, n), "wb") as f:
            f.write(struct.pack("<Q", len(raw)) + raw + body)


def test_load_dense_model_embedding_gemma2(tmp_path):
    g = torch.Generator().manual_seed(41)
    base_meta = {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2", "model_profile": "embedding_gemma2",
                 "encoder_commit": "abc", "hessian": "eg2-v2"}
    rel = tmp_path / "release"
    os.makedirs(rel)
    write_st(str(rel / "exlr8-dense-text-000.safetensors"), {
        **_dense_trellis_side("self_attn.q_proj", 128, 128, 4, 1, g),
        **_dense_int4("self_attn.o_proj", 128, 128, x.IN, g),
        "input_layernorm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
    }, {**base_meta, "contents": "dense", "group": "text", "layer": "0"})
    write_st(str(rel / "exlr8-dense-vision-002.safetensors"), {
        **_dense_trellis_side("self_attn.q_proj.linear", 128, 128, 4, 1, g),
    }, {**base_meta, "contents": "dense", "group": "vision", "layer": "2"})
    write_st(str(rel / "exlr8-dense-audio-001.safetensors"), {
        **_dense_trellis_side("self_attn.post.linear", 128, 128, 4, x.IN, g),
    }, {**base_meta, "contents": "dense", "group": "audio", "layer": "1"})
    write_st(str(rel / "exlr8-dense-global-000.safetensors"), {
        **_dense_trellis_side("language_model.embedding_projection", 128, 128, 4, 1, g),
        "language_model.norm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
    }, {**base_meta, "contents": "dense", "group": "global", "layer": "0"})
    write_st(str(rel / "exlr8-vocab.safetensors"), {
        **_dense_fp8_row("embed_tokens", 200, 128, g),
    }, {**base_meta, "contents": "vocab"})

    unstripped = x.load_dense_model(str(rel), dtype=torch.float32, verify=True)
    assert sorted(unstripped) == [
        "audio_tower.layers.1.self_attn.post.linear.weight",
        "language_model.embed_tokens.weight",
        "language_model.embedding_projection.weight",
        "language_model.layers.0.input_layernorm.weight",
        "language_model.layers.0.self_attn.o_proj.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.norm.weight",
        "vision_tower.encoder.layers.2.self_attn.q_proj.linear.weight",
    ]
    assert unstripped["language_model.embed_tokens.weight"].shape == (200, 128)
    assert unstripped["language_model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16

    stripped_dir = tmp_path / "stripped"
    _strip_dir(str(rel), str(stripped_dir))
    stripped = x.load_dense_model(str(stripped_dir), dtype=torch.float32, verify=True)
    assert list(stripped) == list(unstripped)
    for k in unstripped:
        assert torch.equal(unstripped[k], stripped[k]), k


def test_load_dense_model_qwen3vl_and_parity(tmp_path):
    """Multi-group qwen3_vl release (text, vision incl. 8B 4304 FP8_ROW side=OUT, global incl. BF16 pos_embed &
    5D patch_embed conv, vocab incl. embed_tokens + norm.weight), before and after strip_meta, plus bit-exact parity
    against dense_format.dequant."""
    g = torch.Generator().manual_seed(42)
    base_meta = {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2", "model_profile": "qwen3_vl",
                 "encoder_commit": "def", "hessian": "qwen3vl-v1"}
    rel = tmp_path / "release"
    os.makedirs(rel)

    text_t = {
        **_dense_trellis_side("self_attn.q_proj", 128, 128, 4, 1, g),
        **_dense_trellis_side("self_attn.o_proj", 128, 128, 4, x.IN, g),
        **_dense_int4("mlp.down_proj", 128, 256, x.IN, g),
        "input_layernorm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
    }
    vision_t = {
        **_dense_trellis_side("attn.qkv", 384, 128, 4, 1, g),
        **_dense_fp8_row("mlp.linear_fc1", 4304, 128, g),
        **_dense_fp8_row("mlp.linear_fc2", 128, 4304, g),
        "mlp.linear_fc1.bias": ("BF16", torch.zeros(4304, dtype=torch.bfloat16)),
    }
    global_t = {
        **_dense_bf16("model.visual.pos_embed", 256, 128, g),
        **_dense_trellis_side("model.visual.merger.linear_fc1", 256, 256, 4, 1, g),
        **_dense_trellis_side("model.visual.deepstack_merger_list.0.linear_fc2", 128, 256, 4, x.IN, g),
        "model.visual.patch_embed.proj.weight": ("BF16", torch.randn(128, 3, 2, 16, 16, generator=g).to(torch.bfloat16)),
    }
    vocab_t = {
        **_dense_fp8_row("embed_tokens", 300, 128, g),
        "norm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
    }
    write_st(str(rel / "exlr8-dense-text-000.safetensors"), text_t,
             {**base_meta, "contents": "dense", "group": "text", "layer": "0"})
    write_st(str(rel / "exlr8-dense-vision-000.safetensors"), vision_t,
             {**base_meta, "contents": "dense", "group": "vision", "layer": "0"})
    write_st(str(rel / "exlr8-dense-global-000.safetensors"), global_t,
             {**base_meta, "contents": "dense", "group": "global", "layer": "0"})
    write_st(str(rel / "exlr8-vocab.safetensors"), vocab_t, {**base_meta, "contents": "vocab"})

    loaded = x.load_dense_model(str(rel), dtype=torch.float32, verify=True)
    assert sorted(loaded) == [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.norm.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.mlp.linear_fc1.bias",
        "model.visual.blocks.0.mlp.linear_fc1.weight",
        "model.visual.blocks.0.mlp.linear_fc2.weight",
        "model.visual.deepstack_merger_list.0.linear_fc2.weight",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight",
    ]
    assert loaded["model.visual.blocks.0.mlp.linear_fc1.weight"].shape == (4304, 128)
    assert loaded["model.visual.blocks.0.mlp.linear_fc2.weight"].shape == (128, 4304)
    assert loaded["model.visual.pos_embed.weight"].shape == (256, 128)
    assert loaded["model.visual.patch_embed.proj.weight"].shape == (128, 3, 2, 16, 16)

    # Identical after strip_meta.py
    stripped_dir = tmp_path / "stripped"
    _strip_dir(str(rel), str(stripped_dir))
    loaded_stripped = x.load_dense_model(str(stripped_dir), dtype=torch.float32, verify=True)
    assert list(loaded_stripped) == list(loaded)
    for k in loaded:
        assert torch.equal(loaded[k], loaded_stripped[k]), k

    # Also works under root/dense/ layout
    nested = tmp_path / "nested"
    _strip_dir(str(rel), str(nested / "dense"))
    loaded_nested = x.load_dense_model(str(nested), dtype=torch.bfloat16, verify=True)
    for k in loaded:
        want = loaded[k].to(torch.bfloat16) if loaded[k].dtype == torch.float32 else loaded[k]
        assert torch.equal(loaded_nested[k], want), k

    # Bit-exact parity against dense_format.dequant when exlr8quant sibling repo is available
    for cand in ("exlr8quant-qwen3vl", "exlr8quant"):
        df_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              cand, "requant", "dense")
        if os.path.isfile(os.path.join(df_dir, "dense_format.py")):
            import importlib.util
            import types
            spec = importlib.util.spec_from_file_location("_df_parity", os.path.join(df_dir, "dense_format.py"))
            D = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(D)
            had = x.hadamard(128, "cpu")
            Q = types.SimpleNamespace(
                had_k=128, had_n=128,
                preapply_had_l=lambda w, n: (had @ w.view(-1, n, w.shape[1])).view_as(w),
                preapply_had_r=lambda w, n: (w.view(w.shape[0], -1, n) @ had).view_as(w),
            )
            for fname, pre in (("exlr8-dense-text-000.safetensors", "model.language_model.layers.0."),
                               ("exlr8-dense-vision-000.safetensors", "model.visual.blocks.0."),
                               ("exlr8-dense-global-000.safetensors", ""),
                               ("exlr8-vocab.safetensors", None)):
                sf = x.SafeFile(str(rel / fname))
                mats, _ = x.dense_matrices(sf)
                for m, (base, _cb, data_map) in mats.items():
                    rec = sf.get(base + ".record").numpy()
                    tdata = {s: sf.get(n) for s, n in data_map.items()}
                    w_df, _ = D.dequant(rec, tdata, Q, "cpu", tiles_from_file=True)
                    hf_name = ("model.language_model.embed_tokens.weight" if pre is None else pre + m + ".weight")
                    assert torch.equal(loaded[hf_name], w_df), (fname, m)
            break

    # Checksum verification catches corruption
    bad_path = str(stripped_dir / "exlr8-dense-vision-000.safetensors")
    sf = x.SafeFile(bad_path)
    off = sf.start + sf.header["mlp.linear_fc2.fp8"]["data_offsets"][0]
    with open(bad_path, "r+b") as fh:
        fh.seek(off)
        b = fh.read(1)
        fh.seek(off)
        fh.write(bytes([b[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        x.load_dense_model(str(stripped_dir), verify=True)


def test_load_dense_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        x.load_dense_model(str(tmp_path))
    g = torch.Generator().manual_seed(1)
    write_st(str(tmp_path / "exlr8-dense-text-000.safetensors"),
             _dense_bf16("m", 128, 128, g),
             {"format": "exlr8-v2", "contents": "dense", "model_profile": "unknown_prof", "group": "text", "layer": "0"})
    with pytest.raises(ValueError, match="unknown model_profile"):
        x.load_dense_model(str(tmp_path))
    write_st(str(tmp_path / "exlr8-dense-text-000.safetensors"),
             _dense_bf16("m", 128, 128, g),
             {"format": "exlr8-v2", "contents": "dense", "model_profile": "qwen3_vl", "group": "bogus", "layer": "0"})
    with pytest.raises(ValueError, match="unknown group"):
        x.load_dense_model(str(tmp_path))


def test_load_dense_model_precedence_and_split_layout(tmp_path):
    """root/dense/ takes precedence over root/, and split layouts (dense/ shards + root vocab) are discovered."""
    g = torch.Generator().manual_seed(77)
    base_meta = {"format": "exlr8-v2", "model_profile": "qwen3_vl"}

    # 1. Both root/ and root/dense/ populated -> root/dense/ wins
    both = tmp_path / "both"
    os.makedirs(both / "dense")
    write_st(str(both / "exlr8-dense-text-000.safetensors"),
             _dense_bf16("from_root", 64, 64, g),
             {**base_meta, "contents": "dense", "group": "text", "layer": "0"})
    write_st(str(both / "dense" / "exlr8-dense-text-000.safetensors"),
             _dense_bf16("from_dense", 64, 64, g),
             {**base_meta, "contents": "dense", "group": "text", "layer": "0"})
    got_both = x.load_dense_model(str(both), dtype=torch.float32, verify=True)
    assert "model.language_model.layers.0.from_dense.weight" in got_both
    assert "model.language_model.layers.0.from_root.weight" not in got_both

    # 2. Split layout: dense shards in root/dense/, vocab in root/
    split = tmp_path / "split"
    os.makedirs(split / "dense")
    write_st(str(split / "dense" / "exlr8-dense-text-000.safetensors"),
             _dense_bf16("self_attn.q_proj", 64, 64, g),
             {**base_meta, "contents": "dense", "group": "text", "layer": "0"})
    write_st(str(split / "exlr8-vocab.safetensors"),
             _dense_fp8_row("embed_tokens", 64, 64, g),
             {**base_meta, "contents": "vocab"})
    got_split = x.load_dense_model(str(split), dtype=torch.float32, verify=True)
    assert sorted(got_split) == [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    ]

    # 3. Mixed model_profile fails closed
    mixed = tmp_path / "mixed"
    os.makedirs(mixed)
    write_st(str(mixed / "exlr8-dense-text-000.safetensors"),
             _dense_bf16("p", 64, 64, g),
             {"format": "exlr8-v2", "contents": "dense", "model_profile": "qwen3_vl", "group": "text", "layer": "0"})
    write_st(str(mixed / "exlr8-dense-vision-000.safetensors"),
             _dense_bf16("p", 64, 64, g),
             {"format": "exlr8-v2", "contents": "dense", "model_profile": "embedding_gemma2",
              "group": "vision", "layer": "0"})
    with pytest.raises(ValueError, match="conflicts"):
        x.load_dense_model(str(mixed))


@pytest.mark.parametrize("fmt,data", [
    (x.MXFP8, {"fp8": torch.zeros(64, 256, dtype=torch.uint8), "e8m0": torch.zeros(64, 8, dtype=torch.uint8)}),
    (x.FP8_BLOCK, {"fp8": torch.zeros(64, 256, dtype=torch.uint8), "scale": torch.ones(1, 2)}),
    (x.FP8_ROW, {"fp8": torch.zeros(64, 256, dtype=torch.uint8), "scale": torch.ones(64)}),
    (x.BF16, {"bf16": torch.zeros(64, 256, dtype=torch.bfloat16)}),
    (x.FP32, {"fp32": torch.zeros(64, 256, dtype=torch.float32)}),
    (x.ROWS, {"rows": torch.zeros(64, 128, dtype=torch.uint8), "svh": torch.ones(64, dtype=torch.float16),
              "suh": torch.ones(256, dtype=torch.float16)}),
])
def test_dense_checksums_rejects_side_in_for_row_major_formats(fmt, data):
    rec = torch.tensor([fmt, 64, 256, 4 if fmt == x.ROWS else 0, x.IN, 128, 0, 0], dtype=torch.int64)
    with pytest.raises(ValueError, match="row-major"):
        x.dense_checksums(rec, data)


# ---------------------------------------------------------------- Drama 3 (fish_omni)

def _dense_fp8_block(base, out, inp, g):
    w = torch.randn(out, inp, generator=g) * 0.1
    bo, bi = -(-out // 128), -(-inp // 128)
    wp = torch.zeros(bo * 128, bi * 128)
    wp[:out, :inp] = w
    amax = wp.view(bo, 128, bi, 128).abs().amax(dim=(1, 3))
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (w / scale.repeat_interleave(128, 0)[:out].repeat_interleave(128, 1)[:, :inp]).clamp(-448, 448)
    fp8 = q.to(torch.float8_e4m3fn).view(torch.uint8)
    sums = [x.xxh64(fp8[128 * b:128 * b + 128], scale[b]) for b in range(bo)]
    rec = torch.tensor([x.FP8_BLOCK, out, inp, 0, 1, 128, 0, 0], dtype=torch.int64)
    want = fp8.view(torch.float8_e4m3fn).float() * scale.repeat_interleave(128, 0)[:out].repeat_interleave(128, 1)[:, :inp]
    return {f"{base}.record": rec, f"{base}.checksum": _u64_i64(sums), f"{base}.fp8": fp8, f"{base}.scale": scale}, want


def _build_drama3_trellis_layer(root, L, H_dim, I_dim, n_exp, K, cb, g):
    """Build synthetic `experts/exlr8-layer-LLL.{index,kK}.safetensors` with routed + shared expert."""
    os.makedirs(os.path.join(root, "experts"), exist_ok=True)
    base = os.path.join(root, "experts", f"exlr8-layer-{L:03d}")
    w = x.width_name(K)
    U = I_dim // 128
    B = int(16 * K * H_dim)
    suf = "" if cb == "mcg" else f".{cb}"
    fmt = "exlr8-v4" if cb.endswith("-f16") or K not in (2, 3, 4) else "exlr8-v2"
    cb_meta = f"{cb} {' '.join(x.CODEBOOKS[cb][1])}"

    w_tensors = {}
    for p in x.PROJS:
        w_tensors[f"trellis.{p}{suf}"] = torch.randint(0, 256, (n_exp * U * B,), dtype=torch.uint8, generator=g)
        w_tensors[f"shared.trellis.{p}{suf}"] = torch.randint(0, 256, (U * B,), dtype=torch.uint8, generator=g)
    start, hdr = write_st(f"{base}.{w}.safetensors", w_tensors, {"format": fmt, "codebook": cb_meta, "k": str(K)})

    def make_side(pre, count):
        side = {
            f"{pre}{w}.gate.su_block": torch.rand(count, H_dim // 128, generator=g).half() + 0.5,
            f"{pre}{w}.up.su_block": torch.rand(count, H_dim // 128, generator=g).half() + 0.5,
            f"{pre}{w}.down.sv_block": torch.rand(count, H_dim // 128, generator=g).half() + 0.5,
            f"{pre}{w}.gate.svh": torch.randn(count, I_dim, generator=g).half(),
            f"{pre}{w}.up.svh": torch.randn(count, I_dim, generator=g).half(),
            f"{pre}{w}.down.suh": torch.randn(count, I_dim, generator=g).half(),
        }
        index = torch.zeros(count, 3, U, 2, dtype=torch.int64)
        checks = torch.zeros(count, 3, U, dtype=torch.int64)
        for j in range(count):
            for pi, p in enumerate(x.PROJS):
                tname = f"{pre}trellis.{p}{suf}"
                t_off = start + hdr[tname]["data_offsets"][0]
                for h in range(U):
                    u_idx = j * U + h
                    off = t_off + u_idx * B
                    index[j, pi, h] = torch.tensor([off, B])
                    raw = w_tensors[tname][u_idx * B:(u_idx + 1) * B]
                    c = slice(128 * h, 128 * h + 128)
                    chan = side[f"{pre}{w}.down.suh"][j, c] if p == "down" else side[f"{pre}{w}.{p}.svh"][j, c]
                    blk = side[f"{pre}{w}.down.sv_block" if p == "down" else f"{pre}{w}.{p}.su_block"][j]
                    cs = x.xxh64(raw, chan, blk)
                    checks[j, pi, h] = cs - 2 ** 64 if cs >= 2 ** 63 else cs
        return {**side, f"{pre}{w}.index": index, f"{pre}{w}.checksum": ("U64", checks)}

    signs = {
        "signs.in": (torch.randint(0, 2, (H_dim,), generator=g) * 2 - 1).to(torch.int8),
        "signs.out": (torch.randint(0, 2, (H_dim,), generator=g) * 2 - 1).to(torch.int8),
    }
    idx_tensors = {**signs, **make_side("", n_exp), **make_side("shared.", 1)}
    write_st(
        f"{base}.index.safetensors",
        idx_tensors,
        {
            "format": fmt,
            "widths": str(K),
            "shared_widths": str(K),
            "hidden_size": str(H_dim),
            "moe_intermediate_size": str(I_dim),
            "n_routed_experts": str(n_exp),
        },
    )


def _write_synthetic_drama3_dense(rel, g):
    """Write a synthetic two-layer Drama 3 dense release: gdn:0, attn:1, mtp_shared:0, mtp:0, global:0, vocab."""
    base_meta = {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2", "model_profile": "fish_omni",
                 "encoder_commit": "abc", "hessian": "drama3-calib-v1"}
    os.makedirs(rel, exist_ok=True)
    write_st(str(rel / "exlr8-dense-gdn-000.safetensors"), {
        **_dense_trellis_side("linear_attn.in_proj_qkv", 256, 128, 4, 1, g),
        **_dense_trellis_side("linear_attn.out_proj", 128, 128, 4, x.IN, g),
        **_dense_bf16("linear_attn.in_proj_a", 32, 128, g),
        **_dense_bf16("mlp.gate", 2, 128, g),
        **_dense_bf16("mlp.shared_expert_gate", 1, 128, g),
        "input_layernorm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
        "linear_attn.conv1d.weight": ("BF16", torch.randn(256, 1, 4, generator=g).to(torch.bfloat16)),
    }, {**base_meta, "contents": "dense", "group": "gdn", "layer": "0"})
    write_st(str(rel / "exlr8-dense-attn-001.safetensors"), {
        **_dense_trellis_side("self_attn.q_proj", 256, 128, 4, 1, g),
        **_dense_int4("self_attn.o_proj", 128, 128, x.IN, g),
        **_dense_bf16("mlp.gate", 2, 128, g),
        **_dense_bf16("mlp.shared_expert_gate", 1, 128, g),
        "self_attn.q_norm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
    }, {**base_meta, "contents": "dense", "group": "attn", "layer": "1"})
    write_st(str(rel / "exlr8-dense-mtp_shared-000.safetensors"), {
        **_dense_trellis_side("eh_proj", 128, 256, 4, x.IN, g),
        **_dense_bf16("semantic_embedding", 128, 128, g),
        "enorm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
    }, {**base_meta, "contents": "dense", "group": "mtp_shared", "layer": "0"})
    write_st(str(rel / "exlr8-dense-mtp-000.safetensors"), {
        **_dense_trellis_side("eh_proj", 128, 256, 4, x.IN, g),
        **_dense_bf16("codebook_embedding", 128, 128, g),
        "hnorm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
    }, {**base_meta, "contents": "dense", "group": "mtp", "layer": "0"})
    write_st(str(rel / "exlr8-dense-global-000.safetensors"), {
        **_dense_bf16("audio_decoder.codebook_embeddings", 256, 128, g),
        **_dense_fp8_row("audio_decoder_mtp.heads.0", 128, 128, g),
    }, {**base_meta, "contents": "dense", "group": "global", "layer": "0"})
    write_st(str(rel / "exlr8-vocab.safetensors"), {
        **_dense_fp8_row("embed_tokens", 200, 128, g),
        **_dense_fp8_row("lm_head", 200, 128, g),
        "norm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
    }, {**base_meta, "contents": "vocab"})


def _write_synthetic_drama3_ref_experts(rel, L, n_exp, g):
    base_meta = {"format": "exlr8-v2", "scheme": "exlr8-ref-bf16", "model_profile": "fish_omni",
                 "contents": "dense", "group": "experts", "layer": str(L)}
    t = {}
    want = {"gate": [], "up": [], "down": []}
    for e in range(n_exp):
        for p in x.PROJS:
            blk, w = _dense_fp8_block(f"mlp.experts.{e}.{p}_proj", 128, 128, g)
            t.update(blk)
            want[p].append(w)
    for p in x.PROJS:
        t.update(_dense_bf16(f"mlp.shared_expert.{p}_proj", 128, 128, g))
    write_st(str(rel / f"exlr8-dense-experts-{L:03d}.safetensors"), t, base_meta)
    return want


def test_load_dense_model_fish_omni(tmp_path):
    g = torch.Generator().manual_seed(51)
    rel = tmp_path / "dense_rel"
    _write_synthetic_drama3_dense(rel, g)

    loaded = x.load_dense_model(str(rel), dtype=torch.float32, verify=True)
    assert sorted(loaded) == [
        "audio_decoder.codebook_embeddings.weight",
        "audio_decoder_mtp.heads.0.weight",
        "audio_decoder_mtp.layers.0.codebook_embedding.weight",
        "audio_decoder_mtp.layers.0.eh_proj.weight",
        "audio_decoder_mtp.layers.0.hnorm.weight",
        "audio_decoder_mtp_shared.layers.0.eh_proj.weight",
        "audio_decoder_mtp_shared.layers.0.enorm.weight",
        "audio_decoder_mtp_shared.layers.0.semantic_embedding.weight",
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        "model.language_model.layers.1.mlp.gate.weight",
        "model.language_model.layers.1.mlp.shared_expert_gate.weight",
        "model.language_model.layers.1.self_attn.o_proj.weight",
        "model.language_model.layers.1.self_attn.q_norm.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight",
        "model.language_model.norm.weight",
    ]
    assert loaded["model.language_model.layers.0.linear_attn.conv1d.weight"].shape == (256, 1, 4)

    stripped = tmp_path / "stripped"
    _strip_dir(str(rel), str(stripped))
    loaded_s = x.load_dense_model(str(stripped), dtype=torch.float32, verify=True)
    assert list(loaded_s) == list(loaded)
    for k in loaded:
        assert torch.equal(loaded[k], loaded_s[k]), k


def test_load_drama3_two_layer_release(tmp_path):
    """Synthetic two-layer Drama 3 release across dev-ref (FP8_BLOCK dense experts), full trellis experts/,
    and hybrid (layer 0 trellis overriding dense experts + layer 1 FP8_BLOCK dense experts)."""
    g = torch.Generator().manual_seed(52)
    n_exp = 2

    # 1. dev-ref mode: both layers from exlr8-dense-experts-LLL.safetensors
    ref_dir = tmp_path / "dev_ref"
    _write_synthetic_drama3_dense(ref_dir, g)
    want0 = _write_synthetic_drama3_ref_experts(ref_dir, 0, n_exp, g)
    want1 = _write_synthetic_drama3_ref_experts(ref_dir, 1, n_exp, g)

    sd_ref = x.load_drama3(str(ref_dir), dtype=torch.float32, verify=True)
    for L, want in ((0, want0), (1, want1)):
        pre = f"model.language_model.layers.{L}."
        w13_want = torch.stack([torch.cat((want["gate"][e], want["up"][e]), dim=0) for e in range(n_exp)], dim=0)
        w2_want = torch.stack([want["down"][e] for e in range(n_exp)], dim=0)
        assert torch.equal(sd_ref[f"{pre}mlp.experts.w13_weight"], w13_want)
        assert torch.equal(sd_ref[f"{pre}mlp.experts.w2_weight"], w2_want)
        for p in x.PROJS:
            assert f"{pre}mlp.shared_expert.{p}_proj.weight" in sd_ref
            assert f"{pre}mlp.experts.0.{p}_proj.weight" not in sd_ref

    # 2. Full trellis experts/ mode (v4 pmul1x-f16) with manifest in manifests/
    full_dir = tmp_path / "full_rel"
    _write_synthetic_drama3_dense(full_dir, g)
    for L in (0, 1):
        _build_drama3_trellis_layer(str(full_dir), L, 128, 128, n_exp, 4, "pmul1x-f16", g)
    os.makedirs(full_dir / "manifests")
    manifest = {
        "format": "exlr8-v4-manifest",
        "requires": {"trellis": ["k4 pmul1x-f16"], "stores": [], "mixing": "none"},
        "layers": {
            str(L): {
                "gate": [4] * n_exp, "up": [4] * n_exp, "down": [4] * n_exp,
                "shared": {"gate": 4, "up": 4, "down": 4},
            }
            for L in (0, 1)
        },
    }
    with open(full_dir / "manifests" / "exlr8-manifest-bit400.json", "w") as f:
        json.dump(manifest, f)

    sd_full = x.load_drama3(str(full_dir), "bit400", dtype=torch.bfloat16, verify=True)
    for L in (0, 1):
        pre = f"model.language_model.layers.{L}."
        assert sd_full[f"{pre}mlp.experts.w13_weight"].shape == (n_exp, 256, 128)
        assert sd_full[f"{pre}mlp.experts.w2_weight"].shape == (n_exp, 128, 128)
        assert sd_full[f"{pre}mlp.experts.w13_weight"].dtype == torch.bfloat16
        el = x.ExpertLayer(str(full_dir), L)
        assert torch.equal(
            sd_full[f"{pre}mlp.experts.w13_weight"][0, :128],
            el.projection(0, "gate", 4, verify=True).to(torch.bfloat16),
        )
        assert torch.equal(
            sd_full[f"{pre}mlp.experts.w13_weight"][1, 128:],
            el.projection(1, "up", 4, verify=True).to(torch.bfloat16),
        )
        assert torch.equal(
            sd_full[f"{pre}mlp.shared_expert.down_proj.weight"],
            el.projection("shared", "down", 4, verify=True).to(torch.bfloat16),
        )

    # Missing manifest_name when experts/ has layers fails closed
    with pytest.raises(ValueError, match="manifest_name"):
        x.load_drama3(str(full_dir))

    # 3. Hybrid mode: layer 0 has BOTH experts/ (trellis) and dense experts (experts/ wins); layer 1 uses dense experts
    hybrid_dir = tmp_path / "hybrid_rel"
    _write_synthetic_drama3_dense(hybrid_dir, g)
    _write_synthetic_drama3_ref_experts(hybrid_dir, 0, n_exp, g)
    want1_h = _write_synthetic_drama3_ref_experts(hybrid_dir, 1, n_exp, g)
    _build_drama3_trellis_layer(str(hybrid_dir), 0, 128, 128, n_exp, 4, "pmcgx", g)
    hybrid_manifest = {
        "format": "exlr8-v2-manifest",
        "layers": {
            "0": {
                "gate": [4] * n_exp, "up": [4] * n_exp, "down": [4] * n_exp,
                "shared": {"gate": 4, "up": 4, "down": 4},
            }
        },
    }
    with open(hybrid_dir / "exlr8-manifest-bit400.json", "w") as f:
        json.dump(hybrid_manifest, f)

    sd_hyb = x.load_drama3(str(hybrid_dir), "bit400", dtype=torch.float32, verify=True)
    el0 = x.ExpertLayer(str(hybrid_dir), 0)
    assert torch.equal(sd_hyb["model.language_model.layers.0.mlp.experts.w2_weight"][0],
                       el0.projection(0, "down", 4, verify=True))
    assert torch.equal(sd_hyb["model.language_model.layers.1.mlp.experts.w2_weight"][1], want1_h["down"][1])
