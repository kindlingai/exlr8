# exlr8 v2 file format

Reference decoder: [`loader/exlr8.py`](loader/exlr8.py).

## Symbols

| symbol | meaning | from | GLM-5.3 |
|---|---|---|---|
| H | hidden size | `signs.in` length; metadata `hidden_size` | 6144 |
| I | routed expert intermediate size | `PkK.gate.svh` width; metadata `moe_intermediate_size` | 2048 |
| U | units per expert, I / 128 | | 16 |
| E | routed experts in a layer | `kK.index` rows; metadata `experts` | 256 |
| K | trellis bits per weight | `widths`, `shared_widths` | 2, 3, 4 |
| L | layer id | file name, metadata `layer` | 0-78 |
| unit u | intermediate channels 128u..128u+127 of one expert (`half` in metadata names) | | |
| B | bytes per unit at width K = H·16·K | | K2 196,608 · K3 294,912 · K4 393,216 |
| S | stride between units: B, or ⌈B / 65,536⌉·65,536 if `pad_halves` | | B |

H and I are multiples of 128; the shared expert has intermediate size I. Layers 0..`first_moe_layer`−1 are dense MLP; layers ≥ `num_hidden_layers` are MTP layers.

## Model metadata

Present in every `__metadata__` of a model other than GLM-5.3 (`GlmMoeDsaForCausalLM`); absent means GLM-5.3 values.

| key | GLM-5.3 value |
|---|---|
| `model` | `GlmMoeDsaForCausalLM` |
| `hidden_size` | 6144 |
| `moe_intermediate_size` | 2048 |
| `n_routed_experts` | 256 |
| `num_hidden_layers` | 78 |
| `first_moe_layer` | 3 |
| `tensor_prefix` | `model.layers.` |

HF tensor name = `tensor_prefix` + L + `.` + stored name. A file may carry its own `tensor_prefix` (e.g. MTP layers).

## Release layout

```
config.json  generation_config.json  tokenizer.json  tokenizer_config.json  chat_template.jinja  LICENSE
dense/exlr8-dense-LLL.safetensors              every non-expert matrix and vector of layer L
dense/exlr8-vocab.safetensors                  embed_tokens, lm_head, final norm
dense/exlr8-extra-NAME.safetensors             tensors outside the decoder layers (vision tower, ...)
dense/exlr8-split-NAME.pPPP.safetensors        row ranges of one matrix too large for one file
experts/exlr8-layer-LLL.index.safetensors      MoE layers: offsets, side data, checksums, error tables
experts/exlr8-layer-LLL.kK.safetensors         MoE layers: trellis bytes, one file per stored K
manifests/exlr8-manifest-NAME.json             width per unit
index-slices/tpN/kK/rankR.safetensors          one rank's share of every index file
hashes/<path>.sha256                           "<sha256> <bytes>"; expert files without the experts/ prefix
```

## All safetensors files

| rule | value |
|---|---|
| byte order | little-endian |
| data start | 65,536 (header padded with spaces) |
| tensor offsets | multiples of 65,536 |
| gaps | `__pad.N` uint8 zeros |
| `__metadata__.format` | `exlr8-v2` (`exlr8-v2-index-slice` for index slices) |
| `__metadata__.scheme` | `exlr8-e4m3-v2` |

Index slices are the exception: they are packed, with the header padded with spaces to a multiple of 8, tensors in
order of decreasing item size and then by name (so each is naturally aligned), no `__pad.N` and no `scheme`.

## experts/exlr8-layer-LLL.kK.safetensors

| tensor | dtype | shape | contents |
|---|---|---|---|
| `trellis.gate` `trellis.up` `trellis.down` | U8 | [E·U·S] | unit u of expert e at byte (U·e + u)·S |
| `shared.trellis.gate` `.up` `.down` | U8 | [U·S] | shared expert's units |

A codebook other than mcg suffixes every one of these names, e.g. `trellis.gate.mul1`; see [Codebook](#codebook).

| metadata | value |
|---|---|
| `layer`, `k`, `hessian`, `pad_halves`, `index_file` | |
| `gate_up_order` | `grouped` |
| `down_order` | `item128` |
| `codebook` | required when the codebook is not mcg |

## experts/exlr8-layer-LLL.index.safetensors

`P` = `` (routed, E' = E) or `shared.` (E' = 1). One `PkK.` block per stored width.

| tensor | dtype | shape | contents |
|---|---|---|---|
| `signs.in` | I8 | [H] | ±1, gate/up input side |
| `signs.out` | I8 | [H] | ±1, down output side |
| `PkK.index` | I64 | [E', 3, U, 2] | (gate, up, down) × unit → (byte offset in `.kK` file, B) |
| `PkK.checksum` | U64 | [E', 3, U] | per unit |
| `PkK.gate.su_block` `PkK.up.su_block` | F16 | [E', H/128] | input scale per 128-channel block |
| `PkK.down.sv_block` | F16 | [E', H/128] | output scale per 128-channel block |
| `PkK.gate.svh` `PkK.up.svh` | F16 | [E', I] | output value per channel, sign included |
| `PkK.down.suh` | F16 | [E', I] | input value per channel, sign included |
| `PkK.err_gate` `PkK.err_up` | F32 | [E', U] | rel. output error per unit at width K |
| `PkK.err_down` | F32 | [E'] | rel. output error of the whole down at width K |
| `Pheldout_tokens` | I32 | [E'] | held-out tokens behind the error tables |

| metadata | contents |
|---|---|
| `experts` | routed range, `0-(E−1)` |
| `widths`, `shared_widths` | stored K, comma list |
| `trellis_files` | comma list |
| `codebook` | codebook name, multiplier and `E4M3 rounding`, e.g. `mcg 0xCBAC1FED, E4M3 rounding` |
| `rotation`, `sign_seed`, `sharding`, `checksum` | text forms of the tables below |
| `hessian`, `hessian_rule`, `shrink` | calibration set and Hessian treatment |
| `error_reference`, `error_inputs`, `max_heldout`, `min_heldout` | error-table provenance |
| `encoder_commit`, `writer_commit`, `exllamav3_version` | provenance |

### Expert side vectors

| projection | suh (input, length) | svh (output, length) |
|---|---|---|
| gate, up | `signs.in` × `su_block` repeated ×128 (H) | `svh` (I) |
| down | `suh` (I) | `signs.out` × `sv_block` repeated ×128 (H) |

### Expert checksum

xxhash64, seed 0, over: the unit's B trellis bytes ‖ its 128 channel values (gate/up `svh[128u:128u+128]`, down
`suh[128u:128u+128]`, F16) ‖ the projection's H/128 block scales (F16).

## Tiles

| quantity | value |
|---|---|
| tile | 16 × 16 weights = 32·K bytes = 16·K little-endian int16, exllamav3 `pack_trellis` order |
| orientation | W (in, out); gate/up (H, I), down (I, H) |
| gate/up unit u | tiles (k16 = 0..H/16−1, c = 0..7) at byte (8·k16 + c)·32K; col16 = 8u + c |
| down unit u | tiles (b = 0..H/128−1, r = 0..7, c = 0..7) at byte (64b + 8r + c)·32K; k16 = 8u + r, col16 = 8b + c |

## Decode

| step | operation |
|---|---|
| bits → states | int16 words in pairs, the two words of each pair swapped; bits MSB first; K bits per value, 256 values; state i = the 16 bits ending at value i, wrapping |
| codebook | the file's codebook, below |
| rounding | v → E4M3 (nearest even, clamp ±448) → fp32 (required) |
| placement | stream position j = 8t + i, t = 0..31, i = 0..7; r0 = 2(t mod 4), c0 = ⌊t/4⌋; (row, col) = [(r0,c0), (r0+1,c0), (r0+8,c0), (r0+9,c0), (r0,c0+8), (r0+1,c0+8), (r0+8,c0+8), (r0+9,c0+8)][i] |
| rotation | Hb = block-diagonal Sylvester Hadamard, 128 × 128, scaled 1/√128 |
| matrix | W = diag(suh) · Hb · T · Hb · diag(svh); HF weight = Wᵀ |

## Codebook

The hash from trellis state to value. Σb is the sum of the four bytes of x; both codebooks round the result to E4M3.

| codebook | tensor name | state → v |
|---|---|---|
| mcg | `trellis.gate` | x = (state · 0xCBAC1FED) mod 2³²; x = (x & 0x8FFF8FFF) ^ 0x3B603B60; v = fp16(x & 0xFFFF) + fp16(x >> 16) |
| mul1 | `trellis.gate.mul1` | x = (state · 0x83DCD12D) mod 2³²; v = fp16(1024 + Σb) · c + d, c = fp16 `0x1EEE`, d = fp16 `0xC931` |

mul1's constants put v at (Σb − 510) · c; the fp16 operation order is part of the definition, since the result is
rounded to E4M3.

The tensor name selects the codebook. mcg is the unmarked default and keeps the bare tensor names; any other
codebook suffixes them and sets `codebook` in the trellis file's metadata. Resolve each projection as
`trellis.<proj>`, then `trellis.<proj>.<codebook>` for the codebooks you implement, and refuse the file if neither
is present. The name carries it so that a reader cannot decode one codebook's codes with another's hash, which
yields plausible wrong weights and no error.

Each trellis file selects its own codebook, so the widths of one layer may use different codebooks. The index
file's `codebook` records the encode it was written from and selects nothing.

## Sharding

| unit | rank at TP = N |
|---|---|
| routed unit (e, u) of layer L | (U·e + u + L) mod N |
| shared expert unit u | (U·E + u + L) mod N |

## Dense, vocab, extra and split files

Each matrix `M` = `M.record` + `M.checksum` + data tensors. Vectors (norms, biases, gates) stored as-is under their
stored names.

`M.record`, I64 [8]:

| field | values |
|---|---|
| 0 format | 1 trellis, 2 int4 g32, 3 MXFP8, 4 FP8 128×128 block, 5 FP8 per row, 6 BF16, 7 FP32 |
| 1 out, 2 in | full matrix shape |
| 3 K | trellis width, else 0 |
| 4 sharded side | 0 replicated, 1 output channels, 2 input channels |
| 5 channels per sharding unit | 0 if replicated |
| 6 shared input rotation (trellis) | 0 own; i ≥ 1: entry i of metadata `rotations` (absent: 1 `attn_in`, 2 `q_resid`, 3 `mlp_in`) |
| 7 solve | 0 RTN or copied, 1 Hessian (GPTQ / LDLQ) |

| format | data tensors | value |
|---|---|---|
| 1 trellis | `trellis` U8 [entries, bytes], `suh` F16 [in], `svh` F16 [out] | side 2: entry g = input rows 128g.., down tile order; else entry g = output cols 128g.., gate/up tile order; decode as above |
| 2 int4 g32 | `q4` U8, `s4` F16; side 2: [in/128, out, 64], [in/128, out, 4]; else [out, in/2], [out, in/32] | (nibble − 8) · s4, low nibble first, one scale per 32 inputs |
| 3 MXFP8 | `fp8` U8 [out, in], `e8m0` U8 [out, in/32] | e4m3 · 2^(e8m0 − 127) |
| 4 FP8 block | `fp8` U8 [out, in], `scale` F32 [⌈out/128⌉, ⌈in/128⌉] | e4m3 · scale[⌊r/128⌋, ⌊c/128⌋] |
| 5 FP8 row | `fp8` U8 [out, in], `scale` F32 [out] | e4m3 · scale[r] |
| 6 BF16 | `bf16` [out, in] | as stored |
| 7 FP32 | `fp32` [out, in] | as stored |

| file | contents |
|---|---|
| `exlr8-dense-LLL` | layer L's matrices and vectors; metadata `layer` |
| `exlr8-vocab` | `embed_tokens`, `lm_head`, `norm.weight` |
| `exlr8-extra-NAME` | tensors outside the decoder layers, full HF names |
| `exlr8-split-NAME.pPPP` | `M.record` (full shape) + rows [r0, r1) of M's row-major data tensors; metadata `rows` = `r0-r1`, `parts` |

Dense checksum: one U64 per 128-channel block of the sharded side (per 128 output rows if replicated), xxhash64 seed 0
over that block's rows of each data tensor (int4 side 2: entry b of `q4` then of `s4`; FP8 block: scale row b;
trellis: entry b ‖ its 128 sharded-side vector values ‖ the whole other-side vector).

### GLM-5.3 release

| matrix | layers | format | sharded side |
|---|---|---|---|
| `self_attn.q_a_proj`, `self_attn.kv_a_proj_with_mqa` | 0-78 | MXFP8 | replicated |
| `self_attn.q_b_proj` | 0, 2 | int4 g32 | output |
| `self_attn.q_b_proj` | other | trellis K4 | output |
| `self_attn.kv_b_proj` | 0-78 | FP8 block | output |
| `self_attn.o_proj` | 0, 2, 3 | int4 g32 | input |
| `self_attn.o_proj` | other | trellis K4 | input |
| `self_attn.indexer.wq_b` | 22 indexer layers | MXFP8 | output |
| `self_attn.indexer.wk` | 22 indexer layers | MXFP8 | replicated |
| `self_attn.indexer.weights_proj` | 22 indexer layers | BF16 | output |
| `mlp.gate_proj`, `mlp.up_proj` | 0-2 | trellis K4 | output |
| `mlp.down_proj` | 0-2 | trellis K4 | input |
| `mlp.gate` (router) | 3-78 | BF16 | replicated |
| `eh_proj` | 78 | MXFP8 | replicated |
| `embed_tokens`, `lm_head` | vocab | FP8 per row | replicated |

## manifests/exlr8-manifest-NAME.json

| key | value |
|---|---|
| `format` | `exlr8-v2-manifest` |
| `name`, `version`, `set_id`, `rule`, `mtp_layer`, `source`, `changelog` | strings / list |
| `layers["L"].gate`, `.up` | [E][U] widths per (expert, unit) |
| `layers["L"].down` | [E] widths, one per expert |
| `layers["L"].shared` | `{"gate": [U], "up": [U], "down": K}` |

GLM-5.3 names: `bit225` `bit243` `bit260` `bit325` `bit338` (≈ 2.25 … 3.38 routed bits/weight, KL-optimised),
`down1` / `home` / `up1` (≈ 2.25 / 3.25 / 4.0).

## index-slices/tpN/kK/rankR.safetensors

All units rank R owns at TP = N, width K; one tensor set per MoE layer `Lnnn` (e.g. `L040`); n = units owned.

| tensor | dtype | shape | contents |
|---|---|---|---|
| `Lnnn.units` | I16 | [n, 2] | (expert, unit); expert E = shared |
| `Lnnn.offset` | I64 | [n, 3] | byte offset in `experts/exlr8-layer-LLL.kK.safetensors`, (gate, up, down) |
| `Lnnn.checksum` | U64 | [n, 3] | expert checksum |
| `Lnnn.chan` | F16 | [n, 3, 128] | the unit's channel values (gate/up `svh`, down `suh`) |
| `Lnnn.block` | F16 | [E+1, 3, H/128] | block scales (gate/up `su_block`, down `sv_block`) |
| `Lnnn.signs` | I8 | [2, H] | `signs.in`, `signs.out` |

| metadata | contents |
|---|---|
| `tp`, `rank`, `k`, `layers`, `half_bytes`, `rule`, `files` | slice identity |
| `hidden_size`, `n_routed_experts`, and the other model metadata keys | as in the index files; absent for GLM-5.3 |
| `set_id` | sha256 of lines `"<path> <bytes> <sha256>\n"` over `experts/*` and `dense/*`, sorted by path |
| `index_sha256` | JSON {layer: sha256 of its index file} |
| `source` | release name and version |
