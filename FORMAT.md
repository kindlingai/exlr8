# exlr8 v4 file format

Reference decoder: [`loader/exlr8.py`](loader/exlr8.py).

## Versions

| version | adds | format strings |
|---|---|---|
| v2 | trellis widths K2-K4 in the mcg codebook, native stores | `exlr8-v2` |
| v3 | the mul1 and pmcgx codebooks, selected by tensor name | `exlr8-v2` (unchanged) |
| v4 | widths K0.5-K6 with pmcgx half widths, fp16 (`-f16`) codebooks, mixed codebooks, manifest `requires` | `exlr8-v4` |

A v4 release writes the v4 strings in every file and manifest. A v4 reader also reads `exlr8-v2` files, whose
features are a subset; a v2 reader refuses v4 files by their `format`.

## Symbols

| symbol | meaning | from | GLM-5.3 |
|---|---|---|---|
| H | hidden size | `signs.in` length; metadata `hidden_size` | 6144 |
| I | routed expert intermediate size | `PkK.gate.svh` width; metadata `moe_intermediate_size` | 2048 |
| U | units per expert, I / 128 | | 16 |
| E | routed experts in a layer | `kK.index` rows; metadata `experts` | 256 |
| K | trellis bits per weight: 1 to 5 (6 for an `-f16` codebook), or a half-integer from 0.5 for a [pair codebook](#widths) | `widths`, `shared_widths` | 2, 3, 4 |
| kK | width name: `k` + K, with `p5` for the half (`k4p5` is K = 4.5, `k0p5` is K = 0.5) | | `k2` `k3` `k4` |
| store | where a unit's bytes live: `kK` for trellis width K, or a [native store](#native-stores) name | | `k2` `k3` `k4` |
| L | layer id | file name, metadata `layer` | 0-78 |
| unit u | intermediate channels 128u..128u+127 of one expert (`half` in metadata names) | | |
| B | bytes per unit at width K = H·16·K | | K0.5 49,152 · K1 98,304 · K2 196,608 · K3 294,912 · K4 393,216 · K4.5 442,368 · K5 491,520 · K6 589,824 |
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
experts/exlr8-layer-LLL.kK.safetensors         MoE layers: trellis bytes, one file per stored K (width name)
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
| `__metadata__.format` | `exlr8-v4` (`exlr8-v4-index-slice` for index slices) |
| `__metadata__.scheme` | `exlr8-trellis-v4`; `exlr8-native-v4` in a release with no trellis data. Rounding belongs to each [codebook](#codebook) |

Index slices are the exception: they are packed, with the header padded with spaces to a multiple of 8, tensors in
order of decreasing item size and then by name (so each is naturally aligned), no `__pad.N` and no `scheme`.

## experts/exlr8-layer-LLL.kK.safetensors

| tensor | dtype | shape | contents |
|---|---|---|---|
| `trellis.gate` `trellis.up` `trellis.down` | U8 | [E·U·S] | unit u of expert e at byte (U·e + u)·S |
| `shared.trellis.gate` `.up` `.down` | U8 | [U·S] | shared expert's units |

A codebook other than mcg suffixes every one of these names, e.g. `trellis.gate.mul1`; see [Codebook](#codebook).
A file whose experts use different codebooks suffixes them `.mix` and adds a codebook table; see
[Mixed codebooks](#mixed-codebooks).

| metadata | value |
|---|---|
| `layer`, `hessian`, `pad_halves`, `index_file` | |
| `k` | K as a decimal, e.g. `4` or `4.5` |
| `gate_up_order` | `grouped` |
| `down_order` | `item128` |
| `codebook` | required when the codebook is not mcg; `mix` in a mixed file |
| `codebooks` | mixed files only: the codebook of each id |

## experts/exlr8-layer-LLL.index.safetensors

`P` = `` (routed, E' = E) or `shared.` (E' = 1). One `PkK.` block per stored width, named by the width name
(`k4p5.index`). A native store's block is in
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
`suh[128u:128u+128]`, F16) ‖ the projection's H/128 block scales (F16). In a mixed file, the expert's codebook id
(one byte) follows.

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
| bits → stream | int16 words in pairs, the two words of each pair swapped; bits MSB first; 256·K bits, read as a ring (wrapping) |
| stream → states | mcg, mul1 (and their `-f16` forms): K bits per value; state i = the 16 bits ending at bit K(i + 1). pmcgx: see [Codebook](#codebook) |
| codebook | the file's codebook, below, or in a mixed file the expert's |
| rounding | E4M3 codebooks: v → E4M3 (nearest even, clamp ±448) → fp32 (required). `-f16` codebooks: none, v is the fp16 value |
| placement | stream position j = 8t + i, t = 0..31, i = 0..7; r0 = 2(t mod 4), c0 = ⌊t/4⌋; (row, col) = [(r0,c0), (r0+1,c0), (r0+8,c0), (r0+9,c0), (r0,c0+8), (r0+1,c0+8), (r0+8,c0+8), (r0+9,c0+8)][i] |
| rotation | Hb = block-diagonal Sylvester Hadamard, 128 × 128, scaled 1/√128 |
| matrix | W = diag(suh) · Hb · T · Hb · diag(svh); HF weight = Wᵀ |

## Codebook

The hash from trellis state to value. Σb is the sum of the four bytes of x. Each codebook below rounds the result to
E4M3; its `-f16` form, under [fp16 codebooks](#fp16-codebooks), keeps the fp16 value.

| codebook | tensor name | state → v |
|---|---|---|
| mcg | `trellis.gate` | x = (state · 0xCBAC1FED) mod 2³²; x = (x & 0x8FFF8FFF) ^ 0x3B603B60; v = fp16(x & 0xFFFF) + fp16(x >> 16) |
| mul1 | `trellis.gate.mul1` | x = (state · 0x83DCD12D) mod 2³²; v = fp16(1024 + Σb) · c + d, c = fp16 `0x1EEE`, d = fp16 `0xC931` |
| pmcgx | `trellis.gate.pmcgx` | values 2p and 2p + 1 share one 20-bit state w, the 20 bits ending at bit 2K(p + 1), wrapping; x₁ = (w · 0xCBAC1FED) mod 2³², x₂ = (w · 0xBCA14C51) mod 2³², each masked as mcg's x; v₂ₚ = fp16(x₁ & 0xFFFF) + fp16(x₂ >> 16), v₂ₚ₊₁ = fp16(x₁ >> 16) + fp16(x₂ & 0xFFFF) |

pmcgx keeps K bits per value: each 20-bit state takes 2K new bits, so K may be a half-integer. Its `codebook`
metadata names both multipliers.
mul1's constants put v at (Σb − 510) · c; the fp16 operation order is part of the definition, since the result is
rounded to E4M3 (or, in mul1-f16, used as is).

The tensor name selects the codebook. mcg is the unmarked default and keeps the bare tensor names; any other
codebook suffixes them and sets `codebook` in the trellis file's metadata. Resolve each projection as
`trellis.<proj>`, then `trellis.<proj>.<codebook>` for the codebooks you implement, and refuse the file if neither
is present. The name carries it so that a reader cannot decode one codebook's codes with another's hash, which
yields plausible wrong weights and no error.

Each trellis file selects its own codebook, so the widths of one layer may use different codebooks. The index
file's `codebook` records the encode it was written from and selects nothing.

### Widths

| codebook | K |
|---|---|
| mcg, mul1 | 1, 2, 3, 4, 5 |
| mcg-f16, mul1-f16 | 1, 2, 3, 4, 5, 6 |
| pmcgx (a pair codebook: 2K new bits per state) | 0.5 to 5 in steps of 0.5 (2K = 1 … 10) |
| pmcgx-f16 | 0.5 to 6 in steps of 0.5 (2K = 1 … 12) |

A tile is 32·K bytes at every width (16 at K0.5, 144 at K4.5), so its 16·K words still come in pairs. Refuse a
half-integer K with a codebook that is not a pair codebook, and K6 with an E4M3 codebook. The E4M3 rounding caps
quality near K5: on GLM-5.3 experts it raised the Hessian-weighted error 2% at K4.5, 12% at K5 and 2.3x at K6,
against the same encode without it, which is why the E4M3 widths stop at K5. K0.5 is meant for rarely routed
experts.

### fp16 codebooks

`mcg-f16`, `mul1-f16` and `pmcgx-f16` hash exactly as mcg, mul1 and pmcgx, and skip the E4M3 rounding: the decoded
value is the fp16 v, and the encoder searched with those values. They suit an engine that computes in fp16 or bf16.
An fp16 engine reproduces the encoder's weights exactly. A bf16 engine rounds v a second time (bf16 keeps 8
significant bits, v up to 11); that error is near 2^-8 relative, far below K6's, and the format leaves it to the
engine.

| codebook | tensor name | `codebook` metadata, e.g. |
|---|---|---|
| mcg-f16 | `trellis.gate.mcg-f16` | `mcg-f16 0xCBAC1FED, fp16` |
| mul1-f16 | `trellis.gate.mul1-f16` | `mul1-f16 0x83DCD12D, fp16` |
| pmcgx-f16 | `trellis.gate.pmcgx-f16` | `pmcgx-f16 V2 L20 0xCBAC1FED 0xBCA14C51, fp16` |

The suffix keeps an E4M3 reader from decoding them as their rounded forms. An engine that computes in FP8 can serve
an `-f16` codebook only by rounding its values on load, which loses more than an E4M3 encode of the same width, since
the search never saw the rounded values. It refuses `-f16` codebooks unless its operator asks for that trade. A
release may use `-f16` codebooks at the widths where they pay off (K4.5 and above) and E4M3 codebooks below, since
each width file selects its own codebook.

### Mixed codebooks

A trellis file may give each expert its own codebook. Its trellis tensors then take the suffix `.mix`
(`trellis.gate.mix`, `shared.trellis.down.mix`), every one of them, so a reader without mixed support finds no
name it knows and refuses the file. Units keep the layout above; only the hash differs between experts. Tiles are
tail-biting and stand alone, so no state crosses from one expert to the next.

| tensor | dtype | shape | contents |
|---|---|---|---|
| `trellis.codebook` | U8 | [E] | codebook id of each routed expert, for all three projections |
| `shared.trellis.codebook` | U8 | [1] | codebook id of the shared expert, when the file has it |

| metadata | value |
|---|---|
| `codebook` | `mix` |
| `codebooks` | the codebook of each id, in id order, `; ` between them, each as a single-codebook file's `codebook`, e.g. `mcg 0xCBAC1FED, E4M3 rounding; pmcgx V2 L20 0xCBAC1FED 0xBCA14C51, E4M3 rounding` |

Refuse the file if an id has no entry in `codebooks`, or if an expert's id names a codebook you do not implement.
At a half-integer K every entry must be a pair codebook, and at K6 an `-f16` codebook. The id joins the [expert checksum](#expert-checksum), so a
unit checked against its checksum is checked against its codebook too.

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
| `format` | `exlr8-v4-manifest` |
| `name`, `version`, `set_id`, `rule`, `mtp_layer`, `source`, `changelog` | strings / list |
| `layers["L"].gate`, `.up` | [E][U] widths per (expert, unit) |
| `layers["L"].down` | [E] widths, one per expert |
| `layers["L"].shared` | `{"gate": [U], "up": [U], "down": K}` |
| `requires` | what an engine needs to serve the manifest, below |

A width is K (a JSON number: `4`, `4.5`), or a native store's name.

### requires

The union over every unit the manifest selects, so an engine can refuse a manifest from this one object, before it
opens an expert file. The files stay authoritative: an engine still resolves each file's tensor names and checks
each unit as above.

| key | value |
|---|---|
| `requires.trellis` | sorted list of `"<width name> <codebook>"`, one per (width, codebook) any selected unit uses, e.g. `["k2 mul1", "k4 mul1", "k4p5 pmcgx"]`; experts of a mixed file count under their own codebooks |
| `requires.stores` | sorted list of the native stores any selected unit uses (`[]`: none) |
| `requires.mixing` | `none`: each layer's selected units use one codebook; `width`: some layer's widths use different codebooks, each width one; `expert`: some layer has a mixed file with more than one codebook among its selected experts |

Refuse the manifest if `requires` names a width, codebook, store or mixing level the engine does not serve. Every
v4 manifest carries `requires`; an `exlr8-v2-manifest` has none, and the engine learns the same from the files,
layer by layer.

GLM-5.3 names: `bit225` `bit243` `bit260` `bit325` `bit338` (≈ 2.25 … 3.38 routed bits/weight, KL-optimised),
`down1` / `home` / `up1` (≈ 2.25 / 3.25 / 4.0).

## index-slices/tpN/kK/rankR.safetensors

All units rank R owns at TP = N, width K; one tensor set per MoE layer `Lnnn` (e.g. `L040`); n = units owned.
A native store's slices are `index-slices/tpN/<store>/rankR.safetensors`. They hold `Lnnn.units`, `Lnnn.offset`
(into the store file) and `Lnnn.checksum` only, and their metadata gives `store` and `unit_bytes` in place of `k` and
`half_bytes`. The directory takes the width name, e.g. `index-slices/tp4/k4p5/`.

| tensor | dtype | shape | contents |
|---|---|---|---|
| `Lnnn.units` | I16 | [n, 2] | (expert, unit); expert E = shared |
| `Lnnn.offset` | I64 | [n, 3] | byte offset in `experts/exlr8-layer-LLL.kK.safetensors`, (gate, up, down) |
| `Lnnn.checksum` | U64 | [n, 3] | expert checksum |
| `Lnnn.chan` | F16 | [n, 3, 128] | the unit's channel values (gate/up `svh`, down `suh`) |
| `Lnnn.block` | F16 | [E+1, 3, H/128] | block scales (gate/up `su_block`, down `sv_block`) |
| `Lnnn.signs` | I8 | [2, H] | `signs.in`, `signs.out` |
| `Lnnn.codebook` | U8 | [n] | the unit's codebook id; layers whose width file is mixed only |

| metadata | contents |
|---|---|
| `tp`, `rank`, `k`, `layers`, `half_bytes`, `rule`, `files` | slice identity |
| `hidden_size`, `n_routed_experts`, and the other model metadata keys | as in the index files; absent for GLM-5.3 |
| `layer_experts` | JSON {layer: E} for layers whose E differs from `n_routed_experts` (absent: none) |
| `codebooks` | JSON {layer: the width file's `codebook`, or its `codebooks` for a mixed file}, every layer |
| `set_id` | sha256 of lines `"<path> <bytes> <sha256>\n"` over `experts/*` and `dense/*`, sorted by path |
| `index_sha256` | JSON {layer: sha256 of its index file} |
| `source` | release name and version |
