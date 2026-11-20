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
}
# sha256 of the fp16 table and of the E4M3 table (one byte per value)
TABLES = {
    "mcg": ("95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9",
            "852259a151b9d7c4d8cc6b7537ba5f548e78681488e926953c04e94b275591b9"),
    "mul1": ("bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8",
             "ee3283e85586daec879a03d2cf93b7f33e8f3400b1c2b55e9eb4563f152a45cc"),
    "pmcgx": ("c2f33d5515f86e4c296867d9ca4f6871f8c1441248b24410bebc13e49da442a5",
              "7f66c49ed68c726cc5b602dc7df7788e06662000893b5fa52d4b042a834be07e"),
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
    ("mcg-f16", 6): "626bfe33614027e1", ("mul1-f16", 6): "173d104e9b4fb44b",
    ("pmcgx-f16", 5.5): "8d06db3bf149c785", ("pmcgx-f16", 6): "b888e32d22699015",
}
WIDTHS = {"mcg": [1, 2, 3, 4, 5], "pmcgx": [k / 2 for k in range(1, 11)],
          "mcg-f16": [1, 2, 3, 4, 5, 6], "pmcgx-f16": [k / 2 for k in range(1, 13)]}


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
        assert hex16(x.codebook(s, cb + "-f16").half()) == v


@pytest.mark.parametrize("cb", sorted(TABLES))
def test_codebook_tables(cb):
    states = torch.arange(1 << x.CODEBOOKS[cb][0])
    f16 = x.codebook_f16(states, cb).reshape(-1).contiguous()
    e4 = x.codebook(states, cb).to(torch.float8_e4m3fn).reshape(-1).contiguous()
    assert sha(f16.view(torch.int16).numpy().tobytes()) == TABLES[cb][0]
    assert sha(e4.view(torch.uint8).numpy().tobytes()) == TABLES[cb][1]
    assert torch.equal(x.codebook(states, cb + "-f16").reshape(-1), f16.float())


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
