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
| store | where a unit's bytes live: `kK` for trellis width K, or a [native store](#native-stores) name | | `k2` `k3` `k4` |
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
| `expert_names` | `{"gate":"mlp.experts.{e}.gate_proj","up":"mlp.experts.{e}.up_proj","down":"mlp.experts.{e}.down_proj","shared.gate":"mlp.shared_experts.gate_proj","shared.up":"mlp.shared_experts.up_proj","shared.down":"mlp.shared_experts.down_proj"}` |

HF tensor name = `tensor_prefix` + L + `.` + stored name. A file may carry its own `tensor_prefix` (e.g. MTP layers),
or `hf_prefix`, the whole prefix, when the HF name does not contain L (DeepSeek-V4.1 MTP layer 40 is `mtp.0.`).
`expert_names` is JSON: the HF name of each projection without `.weight`, `{e}` standing for the expert.

## Release layout

```
config.json  generation_config.json  tokenizer.json  tokenizer_config.json  chat_template.jinja  LICENSE
dense/exlr8-dense-LLL.safetensors              every non-expert matrix and vector of layer L
dense/exlr8-vocab.safetensors                  embed_tokens, lm_head, final norm
dense/exlr8-extra-NAME.safetensors             tensors outside the decoder layers (vision tower, ...)
dense/exlr8-split-NAME.pPPP.safetensors        row ranges of one matrix too large for one file
experts/exlr8-layer-LLL.index.safetensors      MoE layers: offsets, side data, checksums, error tables
experts/exlr8-layer-LLL.kK.safetensors         MoE layers: trellis bytes, one file per stored K
experts/exlr8-layer-LLL.STORE.safetensors      MoE layers: native store bytes, one file per native store
manifests/exlr8-manifest-NAME.json             width per unit
index-slices/tpN/STORE/rankR.safetensors       one rank's share of every index file, per store (kK or native)
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
| `__metadata__.scheme` | `exlr8-e4m3-v2`; `exlr8-native-v2` in a release with no trellis data |

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

`P` = `` (routed, E' = E) or `shared.` (E' = 1). One `PkK.` block per stored width. A native store's block is in
[Native stores](#native-stores). `signs.in` and `signs.out` are present when the layer stores a trellis width.

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
| `stores`, `shared_stores` | native stores, comma list (absent: none) |
| `trellis_files`, `store_files` | comma lists |
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

## Native stores

A native store keeps a source checkpoint's own quantized expert bytes, cut into the units above. The store takes a
name where a trellis width takes `kK`: its file is `experts/exlr8-layer-LLL.<store>.safetensors`, its tensors
`<store>.gate` `<store>.up` `<store>.down` (`shared.<store>.…` for the shared expert, U8 [E'·U·B], unit u of
expert e at byte (U·e + u)·B), and its index block `P<store>.`. The names keep a trellis reader from decoding
native bytes as trellis codes. Native units have no rotation, signs or side vectors.

| store | values | B | gate/up unit u | down unit u |
|---|---|---|---|---|
| `mxfp4` | E2M1 · 2^(e8m0 − 127), one E8M0 per 32 inputs | 68·H | codes [128, H/2] ‖ scales [128, H/32] | codes [H, 64] ‖ scales [H, 4] |
| `fp8b32` | E4M3 · 2^(e8m0 − 127), one E8M0 per 32 × 32 block | 128·H + H/8 | fp8 [128, H] ‖ scales [4, H/32] | fp8 [H, 128] ‖ scales [H/32, 4] |

Units are in HF orientation (out, in), row-major. Gate/up unit u holds HF rows 128u..128u+127; down unit u holds
the HF input channels 128u..128u+127 of every row: `mxfp4` code bytes 64u..64u+63 and scale columns 4u..4u+3,
`fp8b32` fp8 columns 128u.. and scale columns 4u..4u+3. An FP4 code byte holds two weights, the lower input index
in the low nibble. E2M1 codes 0-7 are 0, 0.5, 1, 1.5, 2, 3, 4, 6; codes 8-15 are the same values negated.
Concatenating the units (gate/up along rows, down along columns) gives the source tensors `<name>.weight`
(codes) and `<name>.scale` (E8M0) byte for byte, with `<name>` from `expert_names`.

| tensor | dtype | shape | contents |
|---|---|---|---|
| `P<store>.index` | I64 | [E', 3, U, 2] | (gate, up, down) × unit → (byte offset in the store file, B) |
| `P<store>.checksum` | U64 | [E', 3, U] | xxhash64, seed 0, over the unit's B bytes |

A manifest names a native store by its name where it would give a width K.

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
| 0 format | 1 trellis, 2 int4 g32, 3 MXFP8, 4 FP8 128×128 block, 5 FP8 per row, 6 BF16, 7 FP32, 8 FP8 32×32 block |
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
| 8 FP8 32×32 block | `fp8` U8, `e8m0` U8; side 2: [in/128, out, 128], [in/128, ⌈out/32⌉, 4]; else [out, in], [⌈out/32⌉, ⌈in/32⌉] | e4m3 · 2^(e8m0[⌊r/32⌋, ⌊c/32⌋] − 127) |

| file | contents |
|---|---|
| `exlr8-dense-LLL` | layer L's matrices and vectors; metadata `layer` |
| `exlr8-vocab` | `embed_tokens`, `lm_head`, `norm.weight` |
| `exlr8-extra-NAME` | tensors outside the decoder layers, full HF names |
| `exlr8-split-NAME.pPPP` | `M.record` (full shape) + rows [r0, r1) of M's row-major data tensors + `M.checksum` of those rows (r0 a multiple of 128); metadata `rows` = `r0-r1`, `parts`, and for a layer's matrix `layer` and NAME = `LLL.<stored name>` |

Dense checksum: one U64 per 128-channel block of the sharded side (per 128 output rows if replicated), xxhash64 seed 0
over that block's rows of each data tensor (int4 side 2: entry b of `q4` then of `s4`; FP8 block: scale row b;
trellis: entry b ‖ its 128 sharded-side vector values ‖ the whole other-side vector; FP8 32×32 block: rows 128b.. of
`fp8` ‖ rows 4b..4b+3 of `e8m0`, or entry b of each on side 2).

The vocab file may carry `hf_names`, JSON from its stored names to HF names, when they are not GLM-5.3's.

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

### DeepSeek-V4.1-Flash native release

Every weight is the source checkpoint's own: routed experts in store `mxfp4`, the shared expert in `fp8b32`.
Layers 40-42 are the MTP stages `mtp.0.` to `mtp.2.` (`hf_prefix`), with 128 routed experts each.

| key | value |
|---|---|
| `model` | `DeepseekV41ForCausalLM` |
| `hidden_size`, `moe_intermediate_size`, `n_routed_experts` | 5120, 2304 (U = 18), 384 |
| `num_hidden_layers`, `first_moe_layer`, `tensor_prefix` | 40, 0, `layers.` |
| `expert_names` | `ffn.experts.{e}.w1` / `.w3` / `.w2` (gate, up, down), `ffn.shared_experts.w1` / `.w3` / `.w2` |

| matrix | layers | format | sharded side, channels per unit |
|---|---|---|---|
| `attn.wq_a`, `attn.wkv` | all | FP8 32×32 block | replicated |
| `attn.wq_b` | all | FP8 32×32 block | output, 512 (one head) |
| `attn.wo_a` | all | FP8 32×32 block | output, 1024 (one output group) |
| `attn.wo_b` | all | FP8 32×32 block | input, 1024 |
| `attn.indexer.wq_b` | 2, 8, 14, 20, 24, 28, 32, 36 | FP8 32×32 block | output, 128 (one head) |
| `attn.indexer.weights_proj` | 2, 8, 14, 20, 24, 28, 32, 36 | BF16 | output, 1 |
| `attn.indexer.wk`, `attn.compressor.wkv`, `attn.compressor.wgate` | 2, 8, 14, 20 | BF16 | replicated |
| `engram.wkv` | 1, 14 | FP8 32×32 block | replicated |
| `engram.embed` | 1, 14 | MXFP8, split files `LLL.engram.embed` | output, 1 |
| `ffn.gate` (router) | all | BF16 | replicated |
| `main_proj` | 40 | FP8 32×32 block | replicated |
| `markov_head.embed`, `markov_head.head` | 42 | BF16 | output, 1 |
| `confidence_head.proj` | 42 | BF16 | replicated |
| `embed_tokens`, `lm_head` (`embed.weight`, `head.weight`) | vocab | BF16 | output, 1 |

`hc_*`, `attn.attn_sink`, `engram.q_weight`, `engram.k_weight`, `ffn.gate.bias`, `ffn.gate.bias_vl` and the norms
are vectors, stored as-is. The vision tower and aligner are `exlr8-extra-vision`, as-is.

## manifests/exlr8-manifest-NAME.json

| key | value |
|---|---|
| `format` | `exlr8-v2-manifest` |
| `name`, `version`, `set_id`, `rule`, `mtp_layer`, `source`, `changelog` | strings / list |
| `layers["L"].gate`, `.up` | [E][U] widths per (expert, unit) |
| `layers["L"].down` | [E] widths, one per expert |
| `layers["L"].shared` | `{"gate": [U], "up": [U], "down": K}` |

A width is K, or a native store's name.

GLM-5.3 names: `bit225` `bit243` `bit260` `bit325` `bit338` (≈ 2.25 … 3.38 routed bits/weight, KL-optimised),
`down1` / `home` / `up1` (≈ 2.25 / 3.25 / 4.0).

## index-slices/tpN/kK/rankR.safetensors

All units rank R owns at TP = N, width K; one tensor set per MoE layer `Lnnn` (e.g. `L040`); n = units owned.
A native store's slices are `index-slices/tpN/<store>/rankR.safetensors`. They hold `Lnnn.units`, `Lnnn.offset`
(into the store file) and `Lnnn.checksum` only, and their metadata gives `store` and `unit_bytes` in place of `k` and
`half_bytes`.

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
| `layer_experts` | JSON {layer: E} for layers whose E differs from `n_routed_experts` (absent: none) |
| `set_id` | sha256 of lines `"<path> <bytes> <sha256>\n"` over `experts/*` and `dense/*`, sorted by path |
| `index_sha256` | JSON {layer: sha256 of its index file} |
| `source` | release name and version |
