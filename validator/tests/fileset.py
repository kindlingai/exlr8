"""Small synthetic exlr8 v2 filesets for the validator tests. Standard library only.

Geometry: H=256, I=256 (2 units), 4 routed experts, 4 main layers + 1 MTP layer, MoE from layer 1. Widths follow the
GLM-5.3 release: routed K2/K3/K4 and shared K3/K4, and K3 only on the MTP layer. Tensor contents are zeros except where the
validator reads them (records, index offsets).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct

ALIGN = 65536
H, I, E, LAYERS, MTP, FIRST_MOE, VOCAB = 256, 256, 4, 4, 1, 1, 512
U = I // 128
SIZES = {"U8": 1, "I8": 1, "I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4, "I64": 8, "U64": 8}


def write_safetensors(path: str, tensors: list[tuple[str, str, list[int], bytes | None]], meta: dict[str, str],
                      align: bool = True) -> dict[str, int]:
    """Write tensors in order, each at a 64 KiB multiple with __pad.N fills, header padded to 64 KiB.
    Returns {name: absolute data offset}."""
    header: dict[str, object] = {"__metadata__": meta}
    layout = []
    pos = pad = 0
    for name, dtype, shape, data in tensors:
        n = SIZES[dtype]
        for d in shape:
            n *= d
        if align and pos % ALIGN:
            gap = ALIGN - pos % ALIGN
            header[f"__pad.{pad}"] = {"dtype": "U8", "shape": [gap], "data_offsets": [pos, pos + gap]}
            layout.append(bytes(gap))
            pos += gap
            pad += 1
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [pos, pos + n]}
        layout.append(data if data is not None else bytes(n))
        assert len(layout[-1]) == n, name
        pos += n
    blob = json.dumps(header).encode()
    hlen = (-(-(8 + len(blob)) // ALIGN) * ALIGN - 8) if align else len(blob)
    start = 8 + hlen
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", hlen) + blob + b" " * (hlen - len(blob)))
        for b in layout:
            f.write(b)
    return {k: start + v["data_offsets"][0] for k, v in header.items() if k != "__metadata__"}


def i64(*xs: int) -> bytes:
    return struct.pack(f"<{len(xs)}q", *xs)


CODEBOOKS = {"mul1": "mul1 0x83DCD12D, E4M3 rounding"}
MODEL_META = {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2", "hidden_size": str(H),
              "moe_intermediate_size": str(I), "n_routed_experts": str(E), "num_hidden_layers": str(LAYERS),
              "first_moe_layer": str(FIRST_MOE), "model": "Test"}


def record(name: str, fmt: int, out: int, inp: int, K: int = 0, side: int = 0) -> list:
    """A matrix record, its checksum and data tensors."""
    chan = 0 if side == 0 else 128
    blocks = -(-(inp if side == 2 else out) // 128)
    t = [(f"{name}.record", "I64", [8], i64(fmt, out, inp, K, side, chan, 0, 1 if fmt == 1 else 0)),
         (f"{name}.checksum", "U64", [blocks], None)]
    if fmt == 1:
        entries, nbytes = (inp // 128, 16 * K * out) if side == 2 else (out // 128, 16 * K * inp)
        t += [(f"{name}.trellis", "U8", [entries, nbytes], None), (f"{name}.suh", "F16", [inp], None),
              (f"{name}.svh", "F16", [out], None)]
    elif fmt == 3:
        t += [(f"{name}.fp8", "U8", [out, inp], None), (f"{name}.e8m0", "U8", [out, inp // 32], None)]
    elif fmt == 5:
        t += [(f"{name}.fp8", "U8", [out, inp], None), (f"{name}.scale", "F32", [out], None)]
    elif fmt == 6:
        t += [(f"{name}.bf16", "BF16", [out, inp], None)]
    return t


def dense_file(root: str, L: int) -> None:
    t = record("self_attn.q_proj", 3, 512, H) + record("self_attn.o_proj", 1, H, 512, K=4, side=2)
    if L < FIRST_MOE:
        t += record("mlp.gate_proj", 1, 512, H, K=4, side=1) + record("mlp.down_proj", 1, H, 512, K=4, side=2)
    else:
        t += record("mlp.gate", 6, E, H)
    t += [("input_layernorm.weight", "BF16", [H], None)]
    names = {n[:-len(".record")]: f"model.layers.{L}.{n[:-len('.record')]}.weight" for n, *_ in t
             if n.endswith(".record")}
    names["input_layernorm.weight"] = f"model.layers.{L}.input_layernorm.weight"
    meta = dict(MODEL_META, layer=str(L), contents="dense", source_names=json.dumps(names))
    write_safetensors(os.path.join(root, "dense", f"exlr8-dense-{L:03d}.safetensors"), t, meta)


def vocab_file(root: str) -> None:
    t = record("embed_tokens", 5, VOCAB, H) + record("lm_head", 5, VOCAB, H) + [("norm.weight", "BF16", [H], None)]
    write_safetensors(os.path.join(root, "dense", "exlr8-vocab.safetensors"), t, dict(MODEL_META, contents="vocab"))


def expert_layer(root: str, L: int, widths: list[int], shared: list[int], errors: bool = True,
                 codebooks: dict[int, str] | None = None) -> None:
    """One MoE layer's index and kK files, offsets laid out as FORMAT.md says. codebooks maps a width to a
    codebook other than mcg, which suffixes that file's trellis names and names itself in its metadata."""
    base = os.path.join(root, "experts", f"exlr8-layer-{L:03d}")
    offsets = {}
    stored = sorted(set(widths) | set(shared))
    for K in stored:
        B = H * 16 * K
        cb = (codebooks or {}).get(K)
        sfx = f".{cb}" if cb else ""
        t = [(f"trellis.{p}{sfx}", "U8", [E * U * B], None) for p in ("gate", "up", "down") if K in widths]
        t += [(f"shared.trellis.{p}{sfx}", "U8", [U * B], None) for p in ("gate", "up", "down") if K in shared]
        meta = dict(MODEL_META, contents="trellis", layer=str(L), k=str(K), pad_halves="false",
                    index_file=f"exlr8-layer-{L:03d}.index.safetensors", gate_up_order="grouped",
                    down_order="item128", **({"codebook": CODEBOOKS[cb]} if cb else {}))
        offsets[K] = (write_safetensors(f"{base}.k{K}.safetensors", t, meta), B, sfx)
    t = [("signs.in", "I8", [H], None), ("signs.out", "I8", [H], None)]
    for pre, n, ks in (("", E, widths), ("shared.", 1, shared)):
        for K in ks:
            off, B, sfx = offsets[K]
            idx = []
            for e in range(n):
                for p in ("gate", "up", "down"):
                    for u in range(U):
                        idx += [off[f"{pre}trellis.{p}{sfx}"] + (U * e + u) * B, B]
            p = f"{pre}k{K}."
            t += [(p + "index", "I64", [n, 3, U, 2], i64(*idx)), (p + "checksum", "U64", [n, 3, U], None),
                  (p + "gate.svh", "F16", [n, I], None), (p + "gate.su_block", "F16", [n, H // 128], None),
                  (p + "up.svh", "F16", [n, I], None), (p + "up.su_block", "F16", [n, H // 128], None),
                  (p + "down.suh", "F16", [n, I], None), (p + "down.sv_block", "F16", [n, H // 128], None)]
            if errors:
                t += [(p + "err_gate", "F32", [n, U], None), (p + "err_up", "F32", [n, U], None),
                      (p + "err_down", "F32", [n], None)]
        if errors:
            t += [(pre + "heldout_tokens", "I32", [n], None)]
    meta = dict(MODEL_META, contents="index", layer=str(L), widths=",".join(map(str, widths)),
                shared_widths=",".join(map(str, shared)), experts=f"0-{E - 1}",
                trellis_files=",".join(f"exlr8-layer-{L:03d}.k{K}.safetensors" for K in stored), pad_halves="false",
                routed_experts=str(E))
    write_safetensors(f"{base}.index.safetensors", t, meta)


def layer_widths(L: int) -> list[int]:
    return [3] if L >= LAYERS else [2, 3, 4]


def shared_widths(L: int) -> list[int]:
    """As in the GLM-5.3 release: no K2 shared expert, and K3 only on the MTP layer."""
    return [3] if L >= LAYERS else [3, 4]


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def set_id(root: str) -> str:
    rels = sorted(os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/")
                  for top in ("dense", "experts") for d, _, ns in os.walk(os.path.join(root, top)) for n in ns)
    h = hashlib.sha256()
    for rel in rels:
        p = os.path.join(root, rel)
        h.update(f"{rel} {os.path.getsize(p)} {sha(p)}\n".encode())
    return h.hexdigest()


def manifest(root: str, name: str, sid: str, k: int = 3) -> None:
    layers = {}
    for L in range(FIRST_MOE, LAYERS + MTP):
        kk = k if k in layer_widths(L) else layer_widths(L)[0]
        layers[str(L)] = {"gate": [[kk] * U] * E, "up": [[kk] * U] * E, "down": [kk] * E,
                          "shared": {"gate": [kk] * U, "up": [kk] * U, "down": kk}}
    m = {"format": "exlr8-v2-manifest", "name": name, "version": "1", "set_id": sid, "rule": "test", "layers": layers}
    os.makedirs(os.path.join(root, "manifests"), exist_ok=True)
    with open(os.path.join(root, "manifests", f"exlr8-manifest-{name}.json"), "w") as f:
        json.dump(m, f)


def index_slices(root: str, sid: str, tp: int = 2) -> None:
    for K in (2, 3, 4):
        d = os.path.join(root, "index-slices", f"tp{tp}", f"k{K}")
        os.makedirs(d, exist_ok=True)
        for R in range(tp):
            t, idx = [], {}
            for L in range(FIRST_MOE, LAYERS + MTP):
                w, ws = layer_widths(L), shared_widths(L)
                if K not in w and K not in ws:
                    continue
                n = sum(1 for e in range(E) if K in w for u in range(U) if (U * e + u + L) % tp == R) + \
                    sum(1 for u in range(U) if K in ws and (U * E + u + L) % tp == R)
                p = f"L{L:03d}."
                t += [(p + "units", "I16", [n, 2], None), (p + "offset", "I64", [n, 3], None),
                      (p + "checksum", "U64", [n, 3], None), (p + "chan", "F16", [n, 3, 128], None),
                      (p + "block", "F16", [E + (K in ws), 3, H // 128], None), (p + "signs", "I8", [2, H], None)]
                idx[str(L)] = sha(os.path.join(root, "experts", f"exlr8-layer-{L:03d}.index.safetensors"))
            meta = {"format": "exlr8-v2-index-slice", "scheme": "exlr8-e4m3-v2", "tp": str(tp), "rank": str(R),
                    "k": str(K), "set_id": sid, "index_sha256": json.dumps(idx)}
            write_safetensors(os.path.join(d, f"rank{R}.safetensors"), t, meta)


def write_sidecars(root: str) -> None:
    for top in ("dense", "experts", "index-slices"):
        for d, _, names in os.walk(os.path.join(root, top)):
            for n in names:
                p = os.path.join(d, n)
                rel = os.path.relpath(p, root).replace(os.sep, "/")
                side = rel[len("experts/"):] if rel.startswith("experts/") else rel
                out = os.path.join(root, "hashes", side + ".sha256")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    f.write(f"{sha(p)} {os.path.getsize(p)}\n")


def vision_file(root: str) -> None:
    t = [("model.visual.blocks.0.attn.proj.weight", "BF16", [64, 64], None),
         ("model.visual.blocks.0.attn.proj.bias", "BF16", [64], None)]
    write_safetensors(os.path.join(root, "dense", "vision-bf16.safetensors"), t, {"format": "pt"}, align=False)


def build(root: str, *, errors: bool = True, manifests: bool = True, slices: bool = True,
          sidecars: bool = True, codebooks: dict[int, str] | None = None) -> str:
    """A complete, valid fileset at root."""
    for d in ("dense", "experts"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    config = {"architectures": ["Test"], "text_config": {
        "hidden_size": H, "moe_intermediate_size": I, "n_routed_experts": E, "num_hidden_layers": LAYERS,
        "num_nextn_predict_layers": MTP, "first_k_dense_replace": FIRST_MOE, "vocab_size": VOCAB,
        "tie_word_embeddings": False}}
    files = {"config.json": json.dumps(config), "generation_config.json": "{}", "tokenizer.json": "{}",
             "tokenizer_config.json": "{}", "chat_template.jinja": "{{ messages }}", "LICENSE": "MIT\n"}
    for name, text in files.items():
        with open(os.path.join(root, name), "w") as f:
            f.write(text)
    for L in range(LAYERS + MTP):
        dense_file(root, L)
    vocab_file(root)
    for L in range(FIRST_MOE, LAYERS + MTP):
        expert_layer(root, L, layer_widths(L), shared_widths(L), errors=errors, codebooks=codebooks)
    sid = set_id(root)
    if manifests:
        manifest(root, "home", sid)
    if slices:
        index_slices(root, sid)
    if sidecars:
        write_sidecars(root)
    return root
