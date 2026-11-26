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

ST_DTYPES = {torch.int64: "I64", torch.int8: "I8", torch.uint8: "U8", torch.float16: "F16"}


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
    file names in `formats`."""
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


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("G", [16, 32, 64, 128])
def test_int8_groups(tmp_path, side, G):
    g = torch.Generator().manual_seed(G + side)
    t, want = int8_matrix("o_proj", side, G, g)
    os.makedirs(tmp_path / "dense")
    meta = {**INT8_META, "formats": json.dumps({"o_proj": f"int8 g{G}"})}
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


def test_formats_match_legacy_codes(tmp_path):
    """The same trellis file read by its legacy codes, and with formats and field 0 = 0, decodes the same."""
    legacy = dense_file(str(tmp_path / "a"), DENSE_META)
    want = x.load_layer(str(tmp_path / "a"), {"layers": {}, "dense": {"0": {"q_proj": 3}}}, 0, torch.float32,
                        verify=True)
    t = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in legacy.items()}
    for k in ("q_proj.k2.record", "q_proj.k3.record", "o_proj.record"):
        t[k][0] = 0
    formats = {"q_proj.k2": "trellis k2", "q_proj.k3": "trellis k3", "o_proj": "trellis k4"}
    os.makedirs(tmp_path / "b" / "dense")
    write_st(str(tmp_path / "b" / "dense" / "exlr8-dense-000.safetensors"), t,
             {**DENSE_META, "formats": json.dumps(formats)})
    got = x.load_layer(str(tmp_path / "b"), {"layers": {}, "dense": {"0": {"q_proj": 3}}}, 0, torch.float32,
                       verify=True)
    assert got.keys() == want.keys() and all(torch.equal(got[k], want[k]) for k in got)


@pytest.mark.parametrize("change", ["missing", "unknown", "extra", "width", "code", "v2", "no-formats"])
def test_formats_refused(tmp_path, change):
    g = torch.Generator().manual_seed(4)
    t, _ = int8_matrix("o_proj", 1, 64, g)
    formats, meta = {"o_proj": "int8 g64"}, dict(INT8_META)
    if change == "missing":
        formats = {}
    elif change == "unknown":
        formats["o_proj"] = "int8 g48"
    elif change == "extra":
        formats["q_proj"] = "bf16"
    elif change == "width":
        t["o_proj.record"][3] = 64              # field 3 is 0 for int8: G lives in the string
    elif change == "code":
        t["o_proj.record"][0] = x.INT4
    elif change == "v2":
        meta["format"] = "exlr8-v2"
    if change != "no-formats":
        meta["formats"] = json.dumps(formats)
    os.makedirs(tmp_path / "dense")
    write_st(str(tmp_path / "dense" / "exlr8-dense-000.safetensors"), t, meta)
    with pytest.raises(ValueError):
        x.load_layer(str(tmp_path), {"layers": {}}, 0)


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
