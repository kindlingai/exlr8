# exlr8 piece file format (`exlr8-v2-job`)

A **piece** is the safetensors file an encoder uploads for one claimed `(job, layer, expert)` slot under the [site/encoder protocol](protocol/README.md). It holds one expert's encoded trellis tiles and side vectors at every width required by the layer plan, plus the layer's shared sign vectors and encoding provenance.

A release assembler (`exlr8writer`) later packs every layer's pieces into the release files specified in [`FORMAT.md`](FORMAT.md). A job feeder independently decodes each piece and checks its quantization error against the original weights and Hessians before closing the slot.

Structural JSON Schema for the piece header: [`protocol/schemas/piece-header.schema.json`](protocol/schemas/piece-header.schema.json).

## Symbols

| symbol | meaning | source |
|---|---|---|
| `H` | `hidden_size` (positive multiple of 128) | model config |
| `I` | `moe_intermediate_size` (positive multiple of 128) | model config |
| `E` | `routed_experts` (also the shared expert's slot index when `shared_expert` is true) | model config |
| `L` | layer index (`0 <= L < layers`, within `moe_layers`) | claim `layer` |
| `e` | slot index (`0..E-1` routed, `E` shared) | claim `expert` |
| `P` | tensor name prefix: `""` for routed slots, `"shared."` for the shared expert slot | claim `shared` |
| `W` | slot widths: `encode.widths` (routed) or `encode.shared_widths` (shared), elements in `{2, 3, 4}` | claim `encode` |
| `K` | trellis width in bits per weight (`2`, `3`, or `4`) | element of `W` |
| `nw` | number of widths in `W` (`len(W)`) | |

Note: offline multi-expert `exlr8-v2-job` files generalize leading dimension `1` to `n` experts and may hold both routed and `shared.` groups in one file. A site piece always encodes **one** slot (leading dimension `1`, either routed or shared, never both).

## Container layout

A piece is a little-endian [safetensors](https://github.com/huggingface/safetensors) file:

1. Bytes `0..8`: `uint64` little-endian JSON header byte length `N`.
2. Bytes `8..8+N`: UTF-8 JSON object containing `"__metadata__"` (string-to-string map) and one entry `{"dtype": ..., "shape": [...], "data_offsets": [start, end]}` per tensor, with byte offsets relative to `8 + N`.
3. Bytes `8+N..total`: raw little-endian tensor bytes.

### Framing rules enforced by the site

| rule | constraint |
|---|---|
| header length | `N <= 1,048,576` (`1 MiB`) and `8 + N <= total` (else `400`) |
| header JSON | valid UTF-8 JSON object (else `400`) |
| tensor count | at most `64` tensors excluding `__metadata__` (else `422`) |
| allowed dtypes | `I8` (1), `U8` (1), `I16` (2), `F16` (2), `BF16` (2), `I32` (4), `F32` (4), `I64` (8), `U64` (8), `F64` (8) |
| tensor byte length | for every tensor in the file, `data_offsets[1] - data_offsets[0] == prod(shape) * itemsize(dtype)` |
| contiguity | sorted by `data_offsets[0]`, the first tensor starts at `0`, and each next tensor starts at the previous tensor's end (no gaps, no overlaps) |
| exact file size | `8 + N + last_end == total` (no trailing bytes) |
| maximum file size | `total <= 2 * sum(required_tensor_bytes) + 16 MiB` (`maxPieceBytes`, else `413`); through-site `PUT .../piece` is also subject to the Worker 100 MB request limit |

### Canonical header convention (`exlr8enc`)

Not required by the site, but recommended for byte-reproducible pieces:

- `__metadata__` keys sorted lexicographically.
- Tensor keys ordered by ascending `data_offsets[0]`.
- Compact JSON (no spaces after `,` or `:`, ASCII-escaped strings), right-padded with ASCII spaces (`0x20`) so `N` is a multiple of 8.

## Metadata (`__metadata__`)

Every safetensors metadata value is a JSON string.

### Keys validated by the site

The site validates all keys in this table on upload (`pieceProblems`, `argsProblems`, `acceptedEncoder`). The **Feeder** column notes additional checks performed by the job feeder (C5).

| key | format / value | site validation | feeder validation (C5) |
|---|---|---|---|
| `format` | `"exlr8-v2-job"` | exact match | exact match |
| `layer` | decimal layer index, e.g. `"12"` | exact match to slot `layer` | exact match to slot `layer` |
| `experts` | decimal slot index `"<e>"` (routed), or `"none"` (shared) | routed: `== String(expert)`; shared: `== "none"` | exact match to slot |
| `shared` | `"false"` (routed) or `"true"` (shared) | shared: `== "true"`; routed: not checked by site (`exlr8enc` always writes `"false"`) | `"false"` (routed) / `"true"` (shared) |
| `widths` | comma-separated routed widths, e.g. `"2,3,4"` | routed: exact match to `plan.widths.join(",")` (`exlr8enc` writes both `widths` and `shared_widths`) | exact match to layer plan |
| `shared_widths` | comma-separated shared widths, e.g. `"3,4"` | shared: exact match to `plan.shared_widths.join(",")` | exact match to layer plan |
| `codebook` | see [Codebook strings](#codebook-strings) | fallback when `codebook_k<K>` is absent: for each `K` in `W`, `(codebook_k<K> ?? codebook)` must start with the plan's prefix | selects decode codebook when `codebook_k<K>` is absent; verified against plan |
| `codebook_k<K>` | per-width codebook string (`codebook_k2`, `codebook_k3`, `codebook_k4`); written when widths use different codebooks | checked before `codebook` for each `K` in `W`; required whenever the layer plan mixes codebooks across widths | selects per-width decode codebook; verified against plan |
| `tail_biting` | `"two passes"` (if `refine == []`), or `"two passes, then pass 2 again at steps <r1>,<r2>,..."` (written when any width uses `pmcgx`) | if `plan.refine` is non-empty and any width in `W` uses `pmcgx`, must end with `"pass 2 again at steps " + plan.refine.join(",")` | verified against plan |
| `hessian` | job's `hessian_id`, e.g. `"ds41-calib-2m-v2"` | exact match to `job.hessian_id` | exact match to job's `hessian_id` |
| `damping` | decimal `sigma_reg`, e.g. `"0.01"`. Omitted by `exlr8enc` at default `0.025` | `Number(damping ?? 0.025) === plan.damping` | exact match to layer plan |
| `weights_id` | 64-char lowercase hex SHA-256; see [Weights ID](#weights-id) | if model defines `weights.id` **and** piece sets `weights_id`, must match `model.weights.id` | **required** and must match job's `weights.id` |
| `encode_args` | space-separated encode flags; see [`encode_args` and pinned flags](#encode_args-and-pinned-flags) | required; every pinned flag must equal the layer plan's value (and extra pinned flags are rejected) | verified against job plan |
| `encoder_commit` | 7–40 lowercase hex git commit, optional `-unchecked` suffix (`-dirty` rejected) | required; validated with `encoding` by `acceptedEncoder` | recorded for provenance |
| `encoding` | positive decimal integer string (`^[1-9][0-9]{0,8}$`), e.g. `"2"` | if present, must be `>= MIN_ENCODING` and `encoder_commit` (minus `-unchecked`) must match `^[0-9a-f]{7,40}$`; if absent, `encoder_commit` must be in site `encoders` table | recorded for provenance |

### Codebook strings

The site checks `(codebook_k<K> ?? codebook).startsWith(prefix)` using these exact prefixes (also published under `GET /api/config` -> `codebooks`):

| codebook | required prefix in piece metadata | full string written by `exlr8enc` |
|---|---|---|
| `mcg` | `mcg 0xCBAC1FED` | `mcg 0xCBAC1FED, every value rounded to E4M3 inside the search (round to nearest even, saturating finite)` |
| `mul1` | `mul1 0x83DCD12D` | `mul1 0x83DCD12D, every value rounded to E4M3 inside the search (round to nearest even, saturating finite)` |
| `pmcgx` | `pmcgx V2 L20 0xCBAC1FED/0xBCA14C51` | `pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, every value rounded to E4M3 inside the search (round to nearest even, saturating finite)` |

> [!IMPORTANT]
> Notice the slash `/` between `pmcgx`'s two multipliers (`0xCBAC1FED/0xBCA14C51`) in `exlr8-v2-job` piece metadata, versus the space in `FORMAT.md` v4 release metadata examples. A piece using a space instead of `/` is rejected by the site with `422`.

When a layer mixes codebooks across widths (e.g. `k2=pmcgx,k3=mul1,k4=mul1`), `exlr8enc` writes:

- `codebook`: `"per width: k2 pmcgx V2 L20 0xCBAC1FED/0xBCA14C51; k3 mul1 0x83DCD12D; k4 mul1 0x83DCD12D, every value rounded to E4M3 inside the search (round to nearest even, saturating finite)"`
- `codebook_k2`, `codebook_k3`, `codebook_k4`: each width's single-codebook full string from the table above.

### `encode_args` and pinned flags

`encode_args` is a single space-separated string recording the command arguments that reproduce the piece (excluding `--hess`, `--bf16`, and output path). `exlr8enc` formats it as:

```text
<layer> <experts_spec> [--shared] --hess-id <id> --inputs none --widths <w> --shared-widths <sw> --shrink <v> --shrink-min-erank <v> --shrink-skip-layers <v> --hess-min <v> --hess-full <v> --gss-width <v> --codebook <cb> [--damping <v>] [--refine <v>]
```

where `<experts_spec>` is `<e>` for a routed slot and `none` (followed by `--shared`) for a shared slot.

The site parses `encode_args` by splitting on `" "` and extracting every pair `(toks[i], toks[i+1])` where `toks[i]` starts with `"--"` and `toks[i+1]` is non-empty and does not start with `"--"`. It compares the parsed map against `claim.encode.args` for every flag in the pinned set:

| pinned flag | required value |
|---|---|
| `--codebook` | uniform codebook name (e.g. `"pmcgx"`), or ascending per-width list (e.g. `"k2=pmcgx,k3=mul1,k4=mul1"`) |
| `--hess-id` | `job.hessian_id` |
| `--widths` | `plan.widths.join(",")` |
| `--shared-widths` | `plan.shared_widths.join(",")` |
| `--shrink` | job setting (default `"8000"`) |
| `--shrink-min-erank` | job setting (default `"200"`) |
| `--shrink-skip-layers` | job setting (default `"3-6"`) |
| `--hess-min` | job setting (default `"500"`) |
| `--hess-full` | job setting (default `"2000"`) |
| `--gss-width` | job setting (default `"3"`) |
| `--refine` | **present** iff any width in the layer plan uses `pmcgx`: `plan.refine.join(",")`, or `"none"` if `plan.refine` is empty. **Must be absent** if no width uses `pmcgx`. |
| `--damping` | **present** iff `plan.damping !== 0.025`: `String(plan.damping)`. **Must be absent** if `plan.damping === 0.025`. |

Any pinned flag that is missing, differs in value, or is present when the plan omits it (such as `--damping` on a default-damping job or `--refine` on a non-`pmcgx` job) causes the site to reject the piece with `422`. Passing through `claim.encode.args` verbatim guarantees exact agreement.

### Weights ID

`weights_id` is the lowercase 64-character hex SHA-256 digest of the source checkpoint's `config.json` bytes immediately followed by its `model.safetensors.index.json` bytes (no separator); see [`HESSIAN.md#weights-id-derivation`](HESSIAN.md#weights-id-derivation) for full rules and test vectors:

```text
weights_id = hex(sha256(bytes("config.json") || bytes("model.safetensors.index.json")))
```

### Provenance and diagnostic metadata (`exlr8enc`)

`exlr8enc` also writes the following keys. None are checked by the site on upload, though downstream tools (`check_v2.py` and release assembly) read several of them (noted below):

| key | example value | notes |
|---|---|---|
| `scheme` | `"exlr8-e4m3-v2"` | E4M3-rounded trellis scheme; read by `check_v2.py` and carried into release metadata |
| `routed_experts` | `"384"` | model's `routed_experts` count `E` |
| `encoder` | `"exlr8enc 0.1.0"` | encoder binary name and crate version |
| `exllamav3_version` | `"0.0.43"` | search kernel provenance; carried into release index metadata |
| `torch_version` | `"none"` | `"none"` for Rust `exlr8enc`, PyTorch version for Python encoders |
| `gpu` | `"NVIDIA GB10"` | CUDA device name |
| `hessian_rule` | `"shrink 8000 tokens: ..."` | human-readable regularization summary; carried into release index metadata |
| `shrink` | `"8000"` | shrinkage prior weight `tau` |
| `shrink_min_erank` | `"200"` | minimum damped `H_gu` entropy rank for shrinkage |
| `shrink_skip_layers` | `"3-6"` | layer ranges excluded from shrinkage |
| `sign_seed` | `"layer: 1234 + layer (...); intermediate: 100003 * layer + 1009 * expert + {gate 1, up 2, down 3}; shared expert = 384"` | sign seed description; carried into release index metadata |
| `rotation` | `"had 128 both sides; ..."` | rotation/scaling summary; carried into release index metadata |
| `gss_width` | `"3"` | wrapped diagonal tiles used for global scale search |
| `hess_min` | `"500"` | token threshold below which unshrunk experts fall back to identity |
| `hess_full` | `"2000"` | token threshold for full damping |
| `max_heldout` | `"32768"` | held-out token cap metadata |
| `min_heldout` | `"64"` | held-out token minimum metadata |
| `tile_layout` | `"exllamav3 pack_trellis: int16 [k/16, n/16, 16K] per matrix, w as (in, out), tensor-core order"` | tile packing description |
| `error_reference` | `"bf16 (one unit swapped in, every other unit BF16); ..."` | error table reference |
| `swiglu_limit` | `"0"` | checkpoint `swiglu_limit` from `config.json` (`"0"` if absent) |
| `error_inputs` | `"none: no error tables"` | `"none: no error tables"` for site encodes |
| `weights` | `"DeepSeek-V4.1-Flash-1017"` | checkpoint directory basename or Hugging Face repo name |

## Tensors

### Required tensors (validated by the site)

Let `P = ""` for a routed expert slot and `P = "shared."` for the shared expert slot. The site (`pieceTensors`, `structureProblems`) requires every tensor below to exist with the exact `dtype` and `shape`:

| tensor | dtype | shape | bytes | contents |
|---|---|---|---|---|
| `signs.in` | `I8` | `[H]` | `H` | Layer-shared gate/up input signs (`±1`) |
| `signs.out` | `I8` | `[H]` | `H` | Layer-shared down output signs (`±1`) |
| `{P}k{K}.gate.trellis` | `I16` | `[1, H/16, I/16, 16*K]` | `H·I·K / 8` | Packed trellis tiles for `gate`, `(in, out)` = `(H, I)`, for each `K` in `W` |
| `{P}k{K}.up.trellis` | `I16` | `[1, H/16, I/16, 16*K]` | `H·I·K / 8` | Packed trellis tiles for `up`, `(in, out)` = `(H, I)`, for each `K` in `W` |
| `{P}k{K}.down.trellis` | `I16` | `[1, I/16, H/16, 16*K]` | `I·H·K / 8` | Packed trellis tiles for `down`, `(in, out)` = `(I, H)`, for each `K` in `W` |
| `{P}k{K}.gate.svh` | `F16` | `[1, I]` | `2·I` | Gate output per-channel scale × intermediate sign |
| `{P}k{K}.up.svh` | `F16` | `[1, I]` | `2·I` | Up output per-channel scale × intermediate sign |
| `{P}k{K}.down.suh` | `F16` | `[1, I]` | `2·I` | Down input per-channel scale × intermediate sign (with global scale `g` folded in) |
| `{P}k{K}.gate.su_block` | `F16` | `[1, H/128]` | `H / 64` | Gate input 128-channel block scales (without `signs.in`; global scale `g` folded in) |
| `{P}k{K}.up.su_block` | `F16` | `[1, H/128]` | `H / 64` | Up input 128-channel block scales (without `signs.in`; global scale `g` folded in) |
| `{P}k{K}.down.sv_block` | `F16` | `[1, H/128]` | `H / 64` | Down output 128-channel block scales (without `signs.out`) |

For a 3-width slot (`K = 2, 3, 4`), there are `2 + 3 × 9 = 29` required tensors.

### Diagnostic tensors written by `exlr8enc`

The site allows up to 64 tensors total (any extra tensor must still obey the dtype, byte-length, and contiguity rules). `exlr8enc` writes these additional tensors for diagnostics and verification:

| tensor | dtype | shape | when written | contents |
|---|---|---|---|---|
| `experts` | `I32` | `[1]` | routed slots only (omitted for shared) | `[e]` |
| `{P}herr` | `F32` | `[1, nw, 3]` | always | Hessian-weighted relative proxy error `tr(dWᵀ H dW) / tr(Wᵀ H W)` per `(width, proj)` in `(gate, up, down)` order (`NaN` if identity Hessian) |
| `{P}gscale` | `F32` | `[1, nw, 3]` | always | Global scale `g` found by GSS per `(width, proj)` |
| `{P}heldout_tokens` | `I32` | `[1]` | always | Held-out token count (`0` for `--inputs none`) |
| `{P}calib_tokens` | `I64` | `[1]` | always | Calibration token count (`count` for routed, `count_mlp_in` for shared) |
| `{P}hess_mode` | `I8` | `[1]` | always | Hessian regularization mode: `0` = full, `1` = damped, `2` = identity, `3` = shrunk |
| `{P}sigma` | `F32` | `[1]` | always | Effective damping `sigma` applied (`0.0` in identity mode) |
| `{P}err_source` | `I8` | `[1]` | always | Error table source (`3` = none / proxy-only) |
| `{P}erank_gu` | `F32` | `[1]` | always | Entropy rank of damped `H_gu` (`NaN` if `calib_tokens == 0` or identity) |
| `{P}erank_down` | `F32` | `[1]` | always | Entropy rank of damped `H_down` (`NaN` if `calib_tokens == 0` or identity) |
| `{P}shrink_applied` | `U8` | `[1]` | always | `1` if `hess_mode == 3`, else `0` |

### Layer sign vectors (`signs.in` and `signs.out`)

Every expert in layer `L` shares a single pair of hidden-side sign vectors `signs.in` and `signs.out` in the assembled release (`FORMAT.md`). All pieces for layer `L` MUST therefore use the exact deterministic draw:

1. Initialize PyTorch's CPU Mersenne Twister (`MT19937`) with seed `1234 + L`.
2. Draw `H` float32 standard normals using PyTorch's CPU `normal_fill` (24-bit uniform `(u32 & 0xffffff) / 2^24`, Box-Muller in 16-element chunks pairing `(j, j+8)` with `u1 = 1 - d[j]`, `u2 = d[j+8]`), then map `v < 0 ? -1 : +1` as `I8`. This is `signs.in`.
3. Continue with the **same** generator state and draw `H` more signs identically. This is `signs.out`.

Intermediate-side per-expert signs (`seed = 100003 * L + 1009 * e + {gate: 1, up: 2, down: 3}`) are folded into `gate.svh`, `up.svh`, and `down.suh`, so they do not need to match across experts.

## Validation matrix

| check | site (upload) | harvester (`window.py`) | job feeder (`quantfeed`, C5/C6) |
|---|---|---|---|
| Safetensors framing (`N <= 1 MiB`, valid JSON) | Yes (`400`) | Yes | Yes |
| Exact file size (`8 + N + last_end == total`, `<= maxPieceBytes`) | Yes (`413` / `422`) | Yes (matches site `size`) | Yes |
| Presigned object size == `piece-url` `size` | Yes (`422` on `piece-done`) | — | — |
| Tensor count `<= 64`, valid dtypes, contiguous non-overlapping offsets | Yes (`structureProblems`, `422`) | — | Yes |
| Required tensors present with exact `dtype` and `shape` | Yes (`structureProblems`, `422`) | — | Yes |
| `format`, `layer`, `experts`, `shared` match slot | Yes (`pieceProblems`, `422`) | Yes (`format`, `layer`, `experts`) | Yes |
| `widths` / `shared_widths` match layer plan | Yes (`pieceProblems`, `422`) | — | Yes |
| Per-width codebook (`codebook_k<K> ?? codebook`) prefix matches plan | Yes (`pieceProblems`, `422`) | — | Yes |
| `tail_biting` suffix matches `plan.refine` when `pmcgx` refined | Yes (`pieceProblems`, `422`) | — | Yes |
| `hessian == job.hessian_id` | Yes (`pieceProblems`, `422`) | — | Yes |
| `Number(damping ?? 0.025) == plan.damping` | Yes (`pieceProblems`, `422`) | — | Yes |
| `weights_id` matches `model.weights.id` | If piece sets `weights_id` (`422`) | — | Required to be present and match |
| `encode_args` pinned flags match layer plan exactly | Yes (`argsProblems`, `422`) | — | Yes |
| `encoder_commit` / `encoding` accepted (`MIN_ENCODING` or allowlist, no `-dirty`) | Yes (`acceptedEncoder`, `422`) | — | Provenance check |
| Body SHA-256 matches uploader-reported `sha256` (when sent) | No | Yes | Yes |
| All `F16` side vectors finite and not all-zero; all `*.trellis` not all-zero | No | Yes (`audit`) | Yes |
| `signs.in` and `signs.out` are `±1` and match `MT19937(1234 + L)` | No | No | Yes |
| Full trellis decode + public Hessian `proxy_err` inside expected quality band & monotonic across widths | No | No | Yes (C5) |
| Plain unweighted relative weight error `‖dW‖² / ‖W‖²` under bound | No | No | Yes (C5) |
| Held-out input output error `‖X dW‖² / ‖X W‖²` consistent with public `proxy_err` | No | No | Yes (C6) |

## Decode and weight reconstruction

A piece fully determines its decoded weight matrices using the exact codec defined in [`FORMAT.md`](FORMAT.md#tiles) and implemented in [`loader/exlr8.py`](loader/exlr8.py).

### Piece tile grid vs release unit layout

- In a **piece** (`exlr8-v2-job`), each trellis tensor is stored in plain 2-D tile-grid order:
  - `gate`, `up`: `I16 [1, H/16, I/16, 16*K]` — slice `[0, r, c, :]` is the tile at input rows `16r..16r+15`, output columns `16c..16c+15`.
  - `down`: `I16 [1, I/16, H/16, 16*K]` — slice `[0, r, c, :]` is the tile at input rows `16r..16r+15`, output columns `16c..16c+15`.
- Each `[0, r, c, :]` slice of `16*K` little-endian `int16` words (`32*K` bytes) is one 16×16 tile in the exact bit layout of [`FORMAT.md#tiles`](FORMAT.md#tiles).
- Release files (`experts/exlr8-layer-LLL.kK.safetensors`) instead pack tiles by 128-intermediate-channel unit `u = 0..I/128-1` (`grouped` for gate/up, `item128` for down; see `exlr8writer/api.py:_tile_halves`). To decode a piece directly, **do not** apply unit regrouping — decode the `(K_in/16, N_out/16)` grid directly.

### Step-by-step reconstruction

For a projection `p` in `{gate, up, down}` at width `K` in `W`, with `K_in = H, N_out = I` (for `gate`, `up`) or `K_in = I, N_out = H` (for `down`):

1. **Select codebook**: Read `text = meta.get(f"codebook_k{K}") or meta["codebook"]`. The codebook name is its first whitespace-delimited token (`mcg`, `mul1`, or `pmcgx`). All protocol v1 pieces use E4M3 rounding (`scheme = "exlr8-e4m3-v2"`).
2. **Decode tiles**: View `{P}k{K}.{p}.trellis[0]` (`I16 [K_in/16, N_out/16, 16*K]`) as little-endian bytes `U8 [(K_in/16)·(N_out/16), 32*K]`. Decode every tile through [`FORMAT.md#decode`](FORMAT.md#decode) (`bits → stream`, `stream → states`, [`Codebook`](FORMAT.md#codebook) hash with **E4M3 round-to-nearest-even clamped to ±448**, and `placement` via `tile_perm`; reference: `loader.exlr8.decode_tiles(raw, K, cb)`). Reshape `[K_in/16, N_out/16, 16, 16]`, permute `(0, 2, 1, 3)`, and reshape to fp32 `T` of shape `(K_in, N_out)`.
3. **Assemble side vectors** (as fp32):
   - For `gate` and `up`:
     - `suh` (length `H`): `signs.in.float() * repeat_interleave({P}k{K}.{p}.su_block[0].float(), 128)`
     - `svh` (length `I`): `{P}k{K}.{p}.svh[0].float()`
   - For `down`:
     - `suh` (length `I`): `{P}k{K}.down.suh[0].float()`
     - `svh` (length `H`): `signs.out.float() * repeat_interleave({P}k{K}.down.sv_block[0].float(), 128)`
4. **Undo Hadamard rotation and scaling** (`loader.exlr8.reconstruct(T, suh, svh)`):
   Let `Hb` be the block-diagonal 128×128 Sylvester Hadamard matrix scaled by `1/√128` (`loader.exlr8.hadamard(128)`):
   - `X = (Hb · T) * suh[:, None]` (apply `Hb` to each 128-row block, then multiply row `i` by `suh[i]`)
   - `W = (X · Hb) * svh[None, :]` (apply `Hb` to each 128-column block, then multiply column `j` by `svh[j]`)
   - Equivalently: `W = diag(suh) · Hb · T · Hb · diag(svh)` of shape `(K_in, N_out)`.
   - The Hugging Face weight matrix `(out, in)` is `W_hf = Wᵀ`.

Verification test vectors for every codebook state, whole table, and packed tile are in [`FORMAT.md#test-values`](FORMAT.md#test-values) and exercised by [`loader/test_exlr8.py`](loader/test_exlr8.py).

### Feeder error metrics (C5 / C6)

Given decoded weight `W` and original reference weight `W_ref`, both in `(in, out)` orientation (`(H, I)` for gate/up, `(I, H)` for down), let `dW = W - W_ref`:

- **Public Hessian error (`proxy_err`)**: With the projection's `(in, in)` input Hessian `H` (`H_gu` for gate/up, `H_down` for down):

  $$\text{proxy\_err} = \frac{\operatorname{tr}(dW^\top H \, dW)}{\operatorname{tr}(W_{\text{ref}}^\top H \, W_{\text{ref}})}$$

  (Equivalently, in Hugging Face `(out, in)` orientation `W_hf = Wᵀ`, `tr(dW_hf H dW_hfᵀ) / tr(W_hf H W_hfᵀ)`.) When computed against the regularized Hessian used by the encoder, this matches the piece's `{P}herr` tensor (`exlr8enc` `src/encode.rs:proxy_err_with_denom`).
- **Plain weight error**:

  $$\text{plain\_err} = \frac{\|dW\|_F^2}{\|W_{\text{ref}}\|_F^2}$$

- **Held-out output error (C6)**: With held-out inputs `X` of shape `(n_tokens, in)` routed to the expert (`down` inputs computed via the original `gate` and `up` weights):

  $$\text{heldout\_err} = \frac{\|X \, dW\|_F^2}{\|X \, W_{\text{ref}}\|_F^2}$$

## Checklist for third-party encoders

To build a piece accepted by the site (and valid for the feeder and release assembly) from a `Claim` object:

1. Compute `signs.in` and `signs.out` (`I8 [H]`) from `MT19937(1234 + claim.layer)`.
2. Encode each projection (`gate`, `up`, `down`) at every width `K` in `W` (`claim.encode.shared_widths` if `claim.shared`, else `claim.encode.widths`) using the E4M3-rounded codebook `claim.encode.codebooks["k" + K]` and damping `claim.encode.damping`.
3. Pack tiles in `(in/16, out/16, 16*K)` `I16` grid order with leading dim `1` (`[1, H/16, I/16, 16*K]` for gate/up; `[1, I/16, H/16, 16*K]` for down), and store finite nonzero `F16` side vectors (`[1, I]` and `[1, H/128]`).
4. Write all required `__metadata__` keys:
   - `format = "exlr8-v2-job"`, `layer = str(claim.layer)`, `hessian = claim.encode.hessian_id`
   - `experts = "none"` and `shared = "true"` if `claim.shared`, else `experts = str(claim.expert)` and `shared = "false"`
   - `widths = ",".join(claim.encode.widths)` and `shared_widths = ",".join(claim.encode.shared_widths)`
   - `codebook` (and `codebook_k<K>` if widths use different codebooks) starting with the exact prefixes in [Codebook strings](#codebook-strings)
   - `tail_biting` when any width uses `pmcgx` (`"two passes, then pass 2 again at steps " + ",".join(claim.encode.refine)` if `refine` is non-empty, else `"two passes"`)
   - `damping = str(claim.encode.damping)` iff `claim.encode.damping != 0.025`
   - `weights_id = claim.weights_id` (required by the feeder; verified by the site when present)
   - `encode_args` containing every flag/value pair from `claim.encode.args` (and no extra pinned flags)
   - `encoder_commit` (7–40 lowercase hex, no `-dirty`) and `encoding` (integer string `>= MIN_ENCODING`)
5. Serialize as a contiguous safetensors file with header `<= 1 MiB`, `<= 64` tensors, zero gaps/overlaps, and no trailing bytes.
6. Upload via `PUT claim.upload` (with `Content-Length`) or `POST claim.upload_url` -> `PUT` -> `POST claim.upload_done`.
