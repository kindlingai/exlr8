# exlr8 v4 file format

Reference decoder: [`loader/exlr8.py`](loader/exlr8.py).

## Versions

| version | adds | format strings |
|---|---|---|
| v2 | trellis widths K2-K4 in the mcg codebook, native stores | `exlr8-v2` |
| v3 | the mul1 and pmcgx codebooks, selected by tensor name | `exlr8-v2` (unchanged) |
| v4 | widths K0.5-K6 with pair-codebook half widths, the pmul1x codebook, fp16 (`-f16`) codebooks, mixed codebooks, manifest `requires`, dense codebooks, dense matrices at several widths, dense [matrix descriptions](#matrix-descriptions) in metadata `matrices`, [int8 dense groups](#int8-groups) | `exlr8-v4` |

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
| stream → states | mcg, mul1 (and their `-f16` forms): K bits per value; state i = the 16 bits ending at bit K(i + 1). pmcgx, pmul1x (the pair codebooks): see [Codebook](#codebook) |
| codebook | the file's codebook, below, or in a mixed file the expert's |
| rounding | E4M3 codebooks: v → E4M3 (nearest even, clamp ±448) → fp32 (required). `-f16` codebooks: none, v is the fp16 value |
| placement | stream position j = 8t + i, t = 0..31, i = 0..7; r0 = 2(t mod 4), c0 = ⌊t/4⌋; (row, col) = [(r0,c0), (r0+1,c0), (r0+8,c0), (r0+9,c0), (r0,c0+8), (r0+1,c0+8), (r0+8,c0+8), (r0+9,c0+8)][i] |
| rotation | Hb = block-diagonal Sylvester Hadamard, 128 × 128, scaled 1/√128 |
| matrix | W = diag(suh) · Hb · T · Hb · diag(svh); HF weight = Wᵀ |

## Codebook

The hash from trellis state to value. Σb is the sum of the four bytes of x, and x.bᵢ is byte i of x (x.b₀ the lowest).
Each codebook below rounds the result to E4M3; its `-f16` form, under [fp16 codebooks](#fp16-codebooks), keeps the
unrounded value.

| codebook | tensor name | state → v |
|---|---|---|
| mcg | `trellis.gate` | x = (state · 0xCBAC1FED) mod 2³²; x = (x & 0x8FFF8FFF) ^ 0x3B603B60; v = fp16(x & 0xFFFF) + fp16(x >> 16) |
| mul1 | `trellis.gate.mul1` | x = (state · 0x83DCD12D) mod 2³²; v = fp16(1024 + Σb) · c + d, c = fp16 `0x1EEE`, d = fp16 `0xC931` |
| pmcgx | `trellis.gate.pmcgx` | values 2p and 2p + 1 share one 20-bit state w, the 20 bits ending at bit 2K(p + 1), wrapping; x₁ = (w · 0xCBAC1FED) mod 2³², x₂ = (w · 0xBCA14C51) mod 2³², each masked as mcg's x; v₂ₚ = fp16(x₁ & 0xFFFF) + fp16(x₂ >> 16), v₂ₚ₊₁ = fp16(x₁ >> 16) + fp16(x₂ & 0xFFFF) |
| pmul1x | `trellis.gate.pmul1x` | values 2p and 2p + 1 share one 20-bit state w, as in pmcgx; x₁ = (w · 0x13755EC1) mod 2³², x₂ = (w · 0xF8D9B559) mod 2³²; S₂ₚ = x₁.b₀ + x₁.b₁ + x₂.b₂ + x₂.b₃, S₂ₚ₊₁ = x₁.b₂ + x₁.b₃ + x₂.b₀ + x₂.b₁; vⱼ = fp16(1024 + Sⱼ) · c + d, with mul1's c and d |

pmcgx and pmul1x are the pair codebooks. Each keeps K bits per value: each 20-bit state takes 2K new bits, so K may be a
half-integer. Its `codebook` metadata names both multipliers. pmul1x is mul1's byte sum on a pair state: each value sums
two bytes of each product, the low pair of one and the high pair of the other, so both values depend on every state bit,
and it shares mul1's c, d and fp16 operation order. mul1's constants put v near (Σb − 510) · c (d is the fp16 nearest
−1534 c, so the centre sits at Σb = 510.27); the fp16 operation order is part of the definition, since the result is
rounded to E4M3. mul1-f16 does not use this form: see [fp16 codebooks](#fp16-codebooks).

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
| pmcgx, pmul1x (the pair codebooks: 2K new bits per state) | 0.5 to 5 in steps of 0.5 (2K = 1 … 10) |
| pmcgx-f16, pmul1x-f16 | 0.5 to 6 in steps of 0.5 (2K = 1 … 12) |

A tile is 32·K bytes at every width (16 at K0.5, 144 at K4.5), so its 16·K words still come in pairs. Refuse a
half-integer K with a codebook that is not a pair codebook, and K6 with an E4M3 codebook. The E4M3 rounding caps
quality near K5: on GLM-5.3 experts it raised the Hessian-weighted error 2% at K4.5, 12% at K5 and 2.3x at K6,
against the same encode without it, which is why the E4M3 widths stop at K5. K0.5 is meant for rarely routed
experts.

### fp16 codebooks

`mcg-f16`, `mul1-f16`, `pmcgx-f16` and `pmul1x-f16` hash exactly as mcg, mul1, pmcgx and pmul1x, and skip the E4M3
rounding. mcg-f16 and pmcgx-f16 decode to the fp16 v. mul1-f16 decodes to exactly (Σb − 510) · c, and pmul1x-f16 each of
its two values to exactly (Sⱼ − 510) · c, c = fp16 `0x1EEE` as in mul1, with no fp16 rounding of the product (it is
exact in fp32). The encoder searched with these values. They suit an engine that computes in fp16 or bf16. An fp16
engine reproduces the encoder's weights exactly (for mul1-f16 and pmul1x-f16, by keeping the integer, below). A bf16
engine rounds v a second time (bf16 keeps 8 significant bits, v up to 11); that error is near 2^-8 relative, far below
K6's, and the format leaves it to the engine.

mul1-f16's integer Σb − 510 is exact in fp16 (at most 510 in size), so an fp16 engine may keep the integer as the
weight and apply c with the channel or block scales. It may decode the integer as fp16(1024 + Σb) − 1534: a dp4a
of the four bytes that accumulates into `0x6400` (fp16 1024.0, whose ulp is 1), then one HSUB2. It may also leave
the 1534 in every weight and subtract it at the output instead: with the input-side scales applied to the
activations, that is 1534 times the sum of the scaled activations, one number per token for every output. Both
use the encoder's weights exactly (the second differs only in fp32 accumulation order); mul1's fp16 multiply-add
would not, which is why mul1-f16 is defined on the integer. A bf16 engine that keeps the integer rounds those
above 256 in size, as above. mul1's own E4M3 definition is unchanged, and mcg-f16 and pmcgx-f16 are unaffected.

pmul1x-f16 decodes the same way, two values per state. With x₁ and x₂ in registers, PRMT `0x7610` gathers S₂ₚ's four
bytes (x₁.b₀, x₁.b₁, x₂.b₂, x₂.b₃) and PRMT `0x5432` gathers S₂ₚ₊₁'s; a dp4a of each into `0x6400` gives
fp16(1024 + Sⱼ), and one PRMT `0x5410` packs the two into a half2. The 1534 comes off with one HSUB2 or at the
output, as for mul1-f16.

| codebook | tensor name | `codebook` metadata, e.g. |
|---|---|---|
| mcg-f16 | `trellis.gate.mcg-f16` | `mcg-f16 0xCBAC1FED, fp16` |
| mul1-f16 | `trellis.gate.mul1-f16` | `mul1-f16 0x83DCD12D, fp16` |
| pmcgx-f16 | `trellis.gate.pmcgx-f16` | `pmcgx-f16 V2 L20 0xCBAC1FED 0xBCA14C51, fp16` |
| pmul1x-f16 | `trellis.gate.pmul1x-f16` | `pmul1x-f16 V2 L20 0x13755EC1 0xF8D9B559, fp16` |

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

### Test values

Every value here is an fp16 bit pattern in hex. v is the codebook's fp16 value; the E4M3 column is v rounded to
E4M3, written back as fp16. mcg-f16 and pmcgx-f16 decode to v; the mul1-f16 and pmul1x-f16 columns are the integers
Σb − 510 and Sⱼ − 510, and the weight is that times c.

| state | mcg v | mcg E4M3 | mul1 v | mul1 E4M3 | mul1-f16 Σb − 510 |
|---|---|---|---|---|---|
| `0x0000` | `3f60` | `3f80` | `c2e8` | `c300` | −510 |
| `0x0001` | `304e` | `3080` | `3921` | `3900` | 95 |
| `0x1234` | `3e0f` | `3e00` | `b89e` | `b880` | −85 |
| `0x7fff` | `4172` | `4180` | `3fca` | `4000` | 288 |
| `0x8000` | `b915` | `b900` | `baf2` | `bb00` | −128 |
| `0xbeef` | `c293` | `c280` | `3124` | `3100` | 24 |
| `0xffff` | `3acd` | `3b00` | `b936` | `b900` | −96 |

| pmcgx state | v₂ₚ | v₂ₚ₊₁ | v₂ₚ E4M3 | v₂ₚ₊₁ E4M3 |
|---|---|---|---|---|
| `0x00000` | `3f60` | `3f60` | `3f80` | `3f80` |
| `0x00001` | `b268` | `34cb` | `b280` | `3500` |
| `0x12345` | `3e6d` | `c0b4` | `3e80` | `c080` |
| `0x7ffff` | `c120` | `1e00` | `c100` | `1e00` |
| `0x80000` | `3ea4` | `395c` | `3e80` | `3980` |
| `0xbeef5` | `bd2e` | `4070` | `bd00` | `4080` |
| `0xfffff` | `b9b0` | `ac60` | `b980` | `ac80` |

| pmul1x state | v₂ₚ | v₂ₚ₊₁ | v₂ₚ E4M3 | v₂ₚ₊₁ E4M3 | pmul1x-f16 S₂ₚ − 510 | S₂ₚ₊₁ − 510 |
|---|---|---|---|---|---|---|
| `0x00000` | `c2e8` | `c2e8` | `c300` | `c300` | −510 | −510 |
| `0x00001` | `3e8b` | `b9a5` | `3e80` | `b980` | 242 | −104 |
| `0x12345` | `385f` | `2c37` | `3880` | `2c00` | 81 | 10 |
| `0x7ffff` | `3af8` | `3990` | `3b00` | `3980` | 129 | 103 |
| `0x80000` | `bb98` | `bef0` | `bb80` | `bf00` | −140 | −256 |
| `0xbeef5` | `ae2e` | `9770` | `ae00` | `9800` | −14 | 0 |
| `0xfffff` | `ad50` | `3974` | `ad80` | `3980` | −12 | 101 |

Whole tables: the sha256 over every state in order, 0 to 2¹⁶ − 1 (pair codebooks 2²⁰ − 1). The fp16 table holds each v
as a little-endian u16 (pair codebooks: v₂ₚ, then v₂ₚ₊₁); the E4M3 table holds each rounded value as one E4M3 byte.

| codebook | sha256 of the fp16 table | sha256 of the E4M3 table |
|---|---|---|
| mcg | `95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9` | `852259a151b9d7c4d8cc6b7537ba5f548e78681488e926953c04e94b275591b9` |
| mul1 | `bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8` | `ee3283e85586daec879a03d2cf93b7f33e8f3400b1c2b55e9eb4563f152a45cc` |
| pmcgx | `c2f33d5515f86e4c296867d9ca4f6871f8c1441248b24410bebc13e49da442a5` | `7f66c49ed68c726cc5b602dc7df7788e06662000893b5fa52d4b042a834be07e` |
| pmul1x | `c7f10379ba9c11052b774bdac2a53e67c21f26fa57dc2a9e9bd301129c289388` | `b6973dd205d4bc1b87469240f73cac8c05898ac7e12fd37abcb7e0411928d300` |

mul1-f16's table, each Σb − 510 as a little-endian i16: `85abd370a7531193db8b9df21a70a0a4dd870464ea74d7f33fff5db3213b929b`.
pmul1x-f16's table, S₂ₚ − 510 then S₂ₚ₊₁ − 510 for each state, the same way:
`1371509153e88d27e56616f2aadff81a2c3e3c5043f8fd1829606198cc9cd9a8`.

Tiles: one tile of 32·K bytes with byte j = (37j + 11) mod 256, decoded through stream, states, codebook, rounding
and placement to 256 fp32 values, row-major (input, output), little-endian. The sha256 of those 1,024 bytes:

| codebook | K: first 16 hex digits of the sha256 |
|---|---|
| mcg | K1 `0f55b1ebabddf3dc` · K2 `8dde4d941e508fa7` · K3 `640c2065964c409d` · K4 `43b9bff37b64f579` · K5 `cc48fcc66caf3a41` |
| mul1 | K1 `24848da1282fbd70` · K2 `5828c7200d079dad` · K3 `eaa28e21af3b58e2` · K4 `e24c0798dc46e799` · K5 `3d1043cb44bdf76d` |
| pmcgx | K0.5 `dd6bc537b173f3e8` · K1 `f0c40ae8fe7f23d1` · K1.5 `36a1897a31ccd5d4` · K2 `d5b6cc4ed594469f` · K2.5 `1428ff10cbc4044b` · K3 `d4005af0b8591dc1` · K3.5 `b9898c3e54ae2d7e` · K4 `e407108ab19f224c` · K4.5 `143277248d13e27c` · K5 `c037dde67a1f5de4` |
| mcg-f16 | K6 `626bfe33614027e1` |
| mul1-f16 | K6 `8194763e86479b2e` |
| pmcgx-f16 | K5.5 `8d06db3bf149c785` · K6 `b888e32d22699015` |
| pmul1x | K0.5 `2d5246bb48deff89` · K1 `8025036245972646` · K1.5 `37f9be48867ea152` · K2 `98da956a07adbc79` · K2.5 `f4946c25b34b2fe0` · K3 `7a72d1dd5f2c8478` · K3.5 `90a0efed5e77528e` · K4 `79d9ba536e499ff0` · K4.5 `62c17e5462268df4` · K5 `8829e7991f56a46c` |
| pmul1x-f16 | K5.5 `77a36b43b71e4c6b` · K6 `aebf5b38d2a71e33` |

These match gr8's separate implementation (engine/loader-rs) on every state of each codebook and on random tiles at
K1-K5 in each codebook, except mul1-f16, pmul1x and pmul1x-f16, which only the reference loader checks so far.
`loader/test_exlr8.py` checks the reference loader against them.

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

Each matrix `M` = `M.record` + `M.checksum` + data tensors, and an entry in the file's `matrices` metadata.
Vectors (norms, biases, gates) stored as-is under their stored names.

### Matrix descriptions

Metadata `matrices` is a JSON object from each matrix's base name (`M`, or `M.kK` for a matrix stored at several
[widths](#dense-widths)) to its description. A v4 writer describes every matrix of a dense, vocab, extra or split
file there.

| key | value |
|---|---|
| `format` | the [format string](#format-strings) |
| `side` | the sharded side: `replicated`, `out` (output channels) or `in` (input channels) |
| `unit` | channels per sharding unit, a JSON integer; absent when `side` is `replicated` |
| `rotation` | trellis and row trellis only: `own`, or the name of a shared input rotation (`attn_in`, `q_resid`, `mlp_in`, ...). Matrices of a layer that name the same rotation share their input signs and Hadamard, so an engine rotates that input once. Absent means `own` |
| `solve` | how the codes were chosen: `copy` (the source's own values), `rtn` (round to nearest) or `hessian` (GPTQ / LDLQ against the input Hessian) |

Refuse a file whose `matrices` leaves out one of its matrices, names a matrix it does not hold, has a key or value
you do not know, or gives a format string you do not implement. A file with no `matrices` (every `exlr8-v2` file,
and v4 files written before it) describes each matrix by its record's legacy fields, below; `matrices` in an
`exlr8-v2` file is a v4 feature.

### Format strings

| format string | legacy code | data tensors | value |
|---|---|---|---|
| `trellis kK` | 1 | `trellis` U8 [entries, bytes], `suh` F16 [in], `svh` F16 [out] | side `in`: entry g = input rows 128g.., down tile order; else entry g = output cols 128g.., gate/up tile order; decode as above |
| `int4 g32` | 2 | `q4` U8, `s4` F16; side `in`: [in/128, out, 64], [in/128, out, 4]; else [out, in/2], [out, in/32] | (nibble − 8) · s4, low nibble first, one scale per 32 inputs |
| `int8 gG` | none | `q8` I8, `s8` F16; side `in`: [in/128, out, 128], [in/128, out, 128/G]; else [out, in], [out, in/G] | q8 · s8, one scale per G consecutive inputs; see [int8 groups](#int8-groups) |
| `mxfp8` | 3 | `fp8` U8 [out, in], `e8m0` U8 [out, in/32] | e4m3 · 2^(e8m0 − 127) |
| `fp8 b128` | 4 | `fp8` U8 [out, in], `scale` F32 [⌈out/128⌉, ⌈in/128⌉] | e4m3 · scale[⌊r/128⌋, ⌊c/128⌋] |
| `fp8 b32` | 8 | `fp8` U8, `e8m0` U8; side `in`: [in/128, out, 128], [in/128, ⌈out/32⌉, 4]; else [out, in], [⌈out/32⌉, ⌈in/32⌉] | e4m3 · 2^(e8m0[⌊r/32⌋, ⌊c/32⌋] − 127) |
| `fp8 row` | 5 | `fp8` U8 [out, in], `scale` F32 [out] | e4m3 · scale[r] |
| `bf16` | 6 | `bf16` [out, in] | as stored |
| `fp32` | 7 | `fp32` [out, in] | as stored |
| `row-trellis kK` | 9 | `rows` U8 [out, in/256 · 32K], `suh` F16 [in], `svh` F16 [out] | see [Row trellis](#row-trellis) |

kK is a [width name](#symbols) (`trellis k4`, `row-trellis k4p5`). The codebook stays in the tensor name
(`M.trellis.pmcgx`, see [Dense codebooks](#dense-codebooks)), not in the format string.

### Records

`M.record`, I64 [8]. Fields 1 and 2 give the full matrix shape. The other fields are legacy: a file without
`matrices` describes each matrix with them, and a file with `matrices` writes each as 0 or as the description's
legacy value.

| field | legacy value |
|---|---|
| 0 format | the format string's legacy code |
| 1 out, 2 in | full matrix shape (not legacy) |
| 3 K | trellis and row trellis: the width, a half-integer K as −2K (−9 for K4.5); else 0 |
| 4 side | 0 `replicated`, 1 `out`, 2 `in` |
| 5 unit | channels per sharding unit; 0 if replicated |
| 6 rotation | 0 `own`; i ≥ 1: entry i of metadata `rotations`, a comma list (absent: 1 `attn_in`, 2 `q_resid`, 3 `mlp_in`) |
| 7 solve | 0 `rtn` (or `copy`), 1 `hessian` |

Refuse a matrix whose nonzero legacy field disagrees with its description. A format with no legacy code, such as
`int8 gG`, needs `matrices`.

| file | contents |
|---|---|
| `exlr8-dense-LLL` | layer L's matrices and vectors; metadata `layer` |
| `exlr8-vocab` | `embed_tokens`, `lm_head`, `norm.weight` |
| `exlr8-extra-NAME` | tensors outside the decoder layers, full HF names; a matrix under its HF name less `.weight` |
| `exlr8-split-NAME.pPPP` | `M.record` (full shape) + rows [r0, r1) of M's row-major data tensors + `M.checksum` of those rows (r0 a multiple of 128); metadata `rows` = `r0-r1`, `parts`, and for a layer's matrix `layer` and NAME = `LLL.<stored name>` |

Dense checksum: one U64 per 128-channel block of the sharded side (per 128 output rows if replicated), xxhash64 seed 0
over that block's rows of each data tensor in the table's order. Exceptions: `int4 g32`, `int8 gG` and `fp8 b32` on side
`in` hash entry b of each data tensor; `fp8 b128` hashes rows 128b.. of `fp8` ‖ scale row b; `fp8 b32` on other sides
hashes rows 128b.. of `fp8` ‖ rows 4b..4b+3 of `e8m0`; trellis hashes entry b ‖ its 128 sharded-side vector values ‖ the
whole other-side vector; row trellis hashes rows 128b.. of `rows` ‖ `svh` of those rows ‖ the whole `suh`.

### Row trellis

For lookup tables such as embeddings, where an engine reads single rows. Each row decodes from its own bytes: the
rotation runs along the row only, and no tile spans two rows.

Row r is in/256 tiles. Tile t holds the row's values 256t..256t+255 at stream positions 0..255 in order, with no
tile permutation, decoded through stream, states and codebook as above. With v_r the row's in decoded values,

W[r, c] = suh[c] · (Hb v_r)[c] · svh[r]

where Hb applies the 128 × 128 Hadamard to each run of 128 values, as in the trellis decode. `suh` carries the
column signs. It is described as side `out` with 1 channel per sharding unit (any row split works), its width in the
format string,
and the codebook names `rows` as it names `trellis` (`rows.pmcgx`). A table whose rows hold one 256-value slice per
layer, as Gemma 4's per-layer embeddings do, reads one tile per layer per token.

### int8 groups

`int8 gG` stores a matrix as signed 8-bit codes with one scale per group of G consecutive inputs in a row:

W[r, c] = q8[r, c] · s8[r, ⌊c / G⌋]

- G is a power of two from 16 to 128 (`int8 g16` … `int8 g128`), so a group never crosses an entry of 128 inputs
  on side `in`. `in` is a multiple of G, and of 128 when `side` is `in`.
- `q8` is in [−127, 127]. −128 is not a code: refuse a matrix that holds it.
- `s8` is a finite fp16 ≥ 0. A group whose scale is 0 decodes to zeros whatever its codes.
- The product is exact in fp32 (7 + 11 significant bits), so every decoder gets the same weights. An int8 kernel can
  read `q8` as stored and apply `s8` to each group's partial sums.

The description's `solve` says how the codes were chosen. `rtn`: s8 = fp16(max |w| over the group / 127),
q8 = clamp(round-half-even(w / s8), −127, 127). `hessian`: GPTQ against the matrix's input Hessian, columns in input
order, each group's scale set by the same rule from the group's error-corrected weights when the solve reaches its
first column. The decode is the same for both.

### Dense codebooks

A trellis matrix names its codebook as an expert file does: `M.trellis` is mcg, and `M.trellis.<codebook>` names
any other (`o_proj.trellis.pmcgx`). A file holding any such matrix sets `codebooks` in its metadata: every non-mcg
codebook its matrices use, `; ` between them, each as a single-codebook file's `codebook`. Refuse a matrix whose
codebook is missing from `codebooks` or not one you implement, and check its K against the codebook's
[widths](#widths).

### Dense widths

A trellis matrix may be stored at several widths, so that manifests can choose between them. Each stored width then
takes the prefix `M.kK.` in place of `M.`: `M.kK.record`, `M.kK.checksum` and `M.kK.trellis[.<codebook>]`, `M.kK.suh`,
`M.kK.svh`, with K the width name (`q_proj.k2.record`, `q_proj.k3.record`), and its own `matrices` entry whose format
string holds the same K. Each width is a separate encode with its own side vectors and may use its own codebook, but
every width of M has the same shape, sharded side and channels per sharding unit. Such a matrix has no `M.record`.

The manifest's [`dense`](#manifestsexlr8-manifest-namejson) object picks one width for each such matrix. Refuse a
manifest that leaves one out or names a width the file does not hold. Only `exlr8-dense-LLL` files hold several
widths. A matrix stored at one width keeps the plain `M.` names, and a manifest gives it no entry.

The vocab file may carry `hf_names`, JSON from its stored names to HF names, when they are not GLM-5.3's.

### GLM-5.3 release

| matrix | layers | format | sharded side |
|---|---|---|---|
| `self_attn.q_a_proj`, `self_attn.kv_a_proj_with_mqa` | 0-78 | `mxfp8` | replicated |
| `self_attn.q_b_proj` | 0, 2 | `int4 g32` | output |
| `self_attn.q_b_proj` | other | `trellis k4` | output |
| `self_attn.kv_b_proj` | 0-78 | `fp8 b128` | output |
| `self_attn.o_proj` | 0, 2, 3 | `int4 g32` | input |
| `self_attn.o_proj` | other | `trellis k4` | input |
| `self_attn.indexer.wq_b` | 22 indexer layers | `mxfp8` | output |
| `self_attn.indexer.wk` | 22 indexer layers | `mxfp8` | replicated |
| `self_attn.indexer.weights_proj` | 22 indexer layers | `bf16` | output |
| `mlp.gate_proj`, `mlp.up_proj` | 0-2 | `trellis k4` | output |
| `mlp.down_proj` | 0-2 | `trellis k4` | input |
| `mlp.gate` (router) | 3-78 | `bf16` | replicated |
| `eh_proj` | 78 | `mxfp8` | replicated |
| `embed_tokens`, `lm_head` | vocab | `fp8 row` | replicated |

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
| `attn.wq_a`, `attn.wkv` | all | `fp8 b32` | replicated |
| `attn.wq_b` | all | `fp8 b32` | output, 512 (one head) |
| `attn.wo_a` | all | `fp8 b32` | output, 1024 (one output group) |
| `attn.wo_b` | all | `fp8 b32` | input, 1024 |
| `attn.indexer.wq_b` | 2, 8, 14, 20, 24, 28, 32, 36 | `fp8 b32` | output, 128 (one head) |
| `attn.indexer.weights_proj` | 2, 8, 14, 20, 24, 28, 32, 36 | `bf16` | output, 1 |
| `attn.indexer.wk`, `attn.compressor.wkv`, `attn.compressor.wgate` | 2, 8, 14, 20 | `bf16` | replicated |
| `engram.wkv` | 1, 14 | `fp8 b32` | replicated |
| `engram.embed` | 1, 14 | `mxfp8`, split files `LLL.engram.embed` | output, 1 |
| `ffn.gate` (router) | all | `bf16` | replicated |
| `main_proj` | 40 | `fp8 b32` | replicated |
| `markov_head.embed`, `markov_head.head` | 42 | `bf16` | output, 1 |
| `confidence_head.proj` | 42 | `bf16` | replicated |
| `embed_tokens`, `lm_head` (`embed.weight`, `head.weight`) | vocab | `bf16` | output, 1 |

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
| `dense["L"]` | `{"<stored name>": K}` for every matrix of `exlr8-dense-LLL` stored at several [widths](#dense-widths); absent when no file has one |
| `requires` | what an engine needs to serve the manifest, below |

A width is K (a JSON number: `4`, `4.5`), or a native store's name.

### requires

The union over every unit and `dense` width the manifest selects, so an engine can refuse a manifest from this one object, before it
opens an expert file. The files stay authoritative: an engine still resolves each file's tensor names and checks
each unit as above.

| key | value |
|---|---|
| `requires.trellis` | sorted list of `"<width name> <codebook>"`, one per (width, codebook) any selected unit or `dense` width uses, e.g. `["k2 mul1", "k4 mul1", "k4p5 pmcgx"]`; experts of a mixed file count under their own codebooks |
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
