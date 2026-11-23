# exlr8 Hessian and held-out input file format

This specification defines the calibration safetensors files produced by a BF16 calibration forward pass (`calib_forward*.py`), staged in a job's Hessian store for volunteer encoders ([`protocol/`](protocol/README.md)), and kept locally by the job owner for held-out piece verification:

- Routed expert Hessians: `hess-LLL-eAAA-BBB.safetensors` (and single-expert `hess-LLL-eEEE-EEE.safetensors` synthesized by the site).
- Layer dense / shared Hessians: `hess-LLL-dense.safetensors` (plus optional `hess-LLL-dense-heldout.safetensors` and site-synthesized `hess-LLL-shared.safetensors`).
- Held-out expert inputs and routing: `inputs-LLL.safetensors` (kept private on the job owner's machine; never uploaded to the public Hessian store).

Structural JSON Schemas (Draft 2020-12) for the safetensors headers:

- [`protocol/schemas/hess-expert-header.schema.json`](protocol/schemas/hess-expert-header.schema.json)
- [`protocol/schemas/hess-dense-header.schema.json`](protocol/schemas/hess-dense-header.schema.json)
- [`protocol/schemas/inputs-header.schema.json`](protocol/schemas/inputs-header.schema.json)

Related specifications: [`FORMAT.md`](FORMAT.md) (release weight layout), [`PIECE.md`](PIECE.md) (single-slot `exlr8-v2-job` pieces), [`protocol/README.md`](protocol/README.md) (site/encoder HTTP API).

## Symbols

| symbol | meaning | source |
|---|---|---|
| `H` | `hidden_size` (expert gate/up input width; positive multiple of 128) | checkpoint `config.json` |
| `I` | `moe_intermediate_size` (expert down input width; positive multiple of 128) | checkpoint `config.json` |
| `E` | `routed_experts` (`n_routed_experts` / `num_experts`) | checkpoint `config.json` |
| `L` | zero-based decoder layer index (`0 <= L < layers`) | calibration pass |
| `A`, `B` | inclusive routed expert range (`0 <= A <= B < E`), `n = B - A + 1` | filename / `experts` |
| `top_k` | active routed experts per token (`num_experts_per_tok` / `top_k`) | checkpoint `config.json` |
| `n_tok` | total held-out tokens captured for layer `L` | `heldout_sequences * S` (`S - 1` on MTP) |
| `T(d)` | packed upper-triangle element count `d * (d + 1) / 2` | — |

## File inventory and naming

All files are little-endian [safetensors](https://github.com/huggingface/safetensors) containers: an 8-byte unsigned little-endian header length `N`, `N` bytes of UTF-8 JSON header (`__metadata__` plus tensor entries `{dtype, shape, data_offsets}`), followed by contiguous raw little-endian tensor bytes with no gaps, overlaps, or trailing bytes.

| filename | producer | stored where | purpose |
|---|---|---|---|
| `hess-LLL-eAAA-BBB.safetensors` | calibration pass | job Hessian store | Packed upper-triangle F32 input Hessians (`gu`, `down`) and token counts (`count`) for routed experts `A..B` inclusive |
| `hess-LLL-eEEE-EEE.safetensors` | site `GET .../hess/expert` | streamed to encoder | Single-expert slice (`n = 1`) cut from `hess-LLL-eAAA-BBB.safetensors` via HTTP range requests |
| `hess-LLL-dense.safetensors` | calibration pass | job Hessian store | Full symmetric F32 input Hessians (`mlp_in`, `shared_down_in`, and optional dense-weight Hessians) with `count_<name>` |
| `hess-LLL-shared.safetensors` | site `GET .../hess/shared` | streamed to encoder | Subset of `hess-LLL-dense.safetensors`: `mlp_in` + `count_mlp_in` (routed claims), plus `shared_down_in` + `count_shared_down_in` (shared claims) |
| `hess-LLL-dense-heldout.safetensors` | calibration pass (`--dense-heldout`) | job owner only | Same schema as `hess-LLL-dense.safetensors`, accumulated over held-out sequences (`sequences = "held out"`) |
| `inputs-LLL.safetensors` | calibration pass (`--save-inputs`) | job owner only | Held-out expert inputs (`x`) and routing (`topk_idx`, `topk_w`) for error tables and C6 held-out piece verification |

### Naming and layer coverage rules

- `LLL` is the zero-padded 3-digit decimal layer index (`f"{L:03d}"`, matching `^\d{3}$` for layers `0..999`).
- `AAA` and `BBB` are zero-padded to at least 3 decimal digits (`f"{A:03d}"`, `f"{B:03d}"`), representing the inclusive routed expert range `0 <= A <= B < E`. The default chunk size is `32` experts per file (e.g. `hess-003-e000-031.safetensors` .. `hess-003-e224-255.safetensors`), though any chunking (`1 <= n <= E`) is valid.
- **Layer completeness**: A MoE layer `L` has complete public Hessians when the store holds `hess-LLL-dense.safetensors` and a set of `hess-LLL-eAAA-BBB.safetensors` files whose `[A, B]` intervals cover every routed expert `0..E-1` without gaps or overlaps. The site's rolling window (`window.ts`) checks `^hess-(\d{3})-(?:e(\d+)-(\d+)|dense)\.safetensors$` and marks layer `L` complete once the dense file and all `E` routed experts are registered.
- **Privacy boundary**: `inputs-LLL.safetensors` and `hess-LLL-dense-heldout.safetensors` MUST NOT be uploaded to the public Hessian store. Volunteers only receive calibration Hessians (`hess-LLL-eEEE-EEE.safetensors` and `hess-LLL-shared.safetensors`).

## Required metadata (`__metadata__`)

Every safetensors metadata value is a UTF-8 string. Every Hessian file and every held-out input file MUST include `hessian_id`, `weights_id`, and `layer`, plus the file-type keys below.

### Common required keys (all file types)

| key | format | meaning |
|---|---|---|
| `hessian_id` | non-empty string, e.g. `"qwen38fn-calib-v1"` | Identifier of the calibration run; MUST match `job.hessian_id` in the site config and the `hessian` metadata of pieces encoded from these files |
| `weights_id` | 64-char lowercase hex (`^[0-9a-f]{64}$`) | SHA-256 checkpoint identifier derived from `config.json` and `model.safetensors.index.json`; see [Weights ID derivation](#weights-id-derivation) |
| `layer` | unpadded decimal integer (`^[0-9]+$`), e.g. `"3"` | Decoder layer index `L`; MUST equal `int(LLL)` in the filename |

### File-specific metadata

| file family | key | required | value / format |
|---|---|---|---|
| `hess-LLL-eAAA-BBB` / `eEEE-EEE` | `experts` | **yes** | Unpadded decimal inclusive range `"<A>-<B>"` (e.g. `"0-31"`, `"5-5"`). MUST match `int(AAA)` and `int(BBB)` in the filename, and `B - A + 1` MUST equal `gu.shape[0]`, `down.shape[0]`, and `count.shape[0]` |
| `hess-LLL-eAAA-BBB` / `eEEE-EEE` | `layout` | **yes** | Exact string `"upper triangle, row-major, fp32"` |
| `hess-LLL-dense` / `-dense-heldout` / `-shared` | `layout` | **yes** | Exact string `"full, fp32"` |
| `hess-LLL-dense` / `-dense-heldout` / `-shared` | `sequences` | **yes** | `"calibration"` for `hess-LLL-dense.safetensors` (and `hess-LLL-shared.safetensors` sliced from it), or `"held out"` for `hess-LLL-dense-heldout.safetensors` |
| `inputs-LLL` | `heldout_sequences` | **yes** | Positive decimal integer (`^[1-9][0-9]*$`), number of held-out sequences whose tokens are concatenated in `x` |
| site-synthesized `eEEE-EEE` / `-shared` | `source` | site-added | Basename of the store object sliced by the site (e.g. `"hess-003-e000-031.safetensors"`, `"hess-003-dense.safetensors"`). The site copies all other `__metadata__` keys from the source file verbatim, so `hessian_id` and `weights_id` propagate automatically |

> [!NOTE]
> **Legacy pre-A3 files**: Calibration files produced before this spec (such as early Qwen3.8-Flash-Next and GLM-5.3 runs) omitted `hessian_id` and `weights_id`, writing only `layer`, `experts`, `layout` (for expert files), `layer`, `layout`, `sequences` (for dense files), and `layer`, `heldout_sequences` (for `inputs-LLL.safetensors`). Producers MUST now emit both `hessian_id` and `weights_id`; consumers handling historical stores may opt into a compatibility fallback only when explicitly configured.

## Weights ID derivation

`weights_id` binds Hessians, held-out inputs, jobs, and encoded pieces to one exact source checkpoint. It is defined identically to `exlr8enc`'s `hf::weights_id` (`src/hf.rs`, called by `weights::Weights::new` in `src/weights.rs`):

```text
weights_id = hex_lower(SHA-256(bytes("config.json") || bytes("model.safetensors.index.json")))
```

Rules:

1. Read the **exact raw bytes** of `config.json` from the checkpoint root (at the exact Hugging Face revision or local checkpoint directory).
2. Read the **exact raw bytes** of `model.safetensors.index.json` from the same checkpoint root.
3. Concatenate them in that order (`config.json` first, `model.safetensors.index.json` second) with **no separator** and **no JSON parsing or re-serialization** (whitespace, key order, and trailing newlines affect the digest).
4. Compute SHA-256 and encode the 32-byte digest as 64 lowercase hexadecimal ASCII characters (`^[0-9a-f]{64}$`).

Reference implementations:

```sh
# Shell (Linux / macOS)
cat "$MODEL_DIR/config.json" "$MODEL_DIR/model.safetensors.index.json" | sha256sum | awk '{print $1}'
# (on macOS without sha256sum: shasum -a 256 | awk '{print $1}')
```

```python
import hashlib
from pathlib import Path

def weights_id(model_dir: str | Path) -> str:
    root = Path(model_dir)
    h = hashlib.sha256()
    h.update((root / "config.json").read_bytes())
    h.update((root / "model.safetensors.index.json").read_bytes())
    return h.hexdigest()
```

Note on single-file checkpoints: every model targeted by exlr8 is sharded and carries `model.safetensors.index.json`. If a calibration pass is run against a quantized overlay (`calib_forward.py --weights ...`), `--model` still supplies the reference BF16 checkpoint directory, or `--weights-id` may override the digest explicitly.

## `hess-LLL-eAAA-BBB.safetensors` (routed expert Hessians)

Holds the **unnormalized** second-moment activation sums accumulated over the calibration sequences for routed experts `e = A .. B` (`n = B - A + 1`), in expert order `j = e - A`:

- $H_{\text{gu}}[e] = \sum_{t \in \mathcal{T}_e} x_t x_t^\top \in \mathbb{R}^{H \times H}$, where $\mathcal{T}_e$ is the set of calibration tokens routed to expert $e$ and $x_t \in \mathbb{R}^H$ is the expert input vector (shared by `gate` and `up`).
- $H_{\text{down}}[e] = \sum_{t \in \mathcal{T}_e} a_t a_t^\top \in \mathbb{R}^{I \times I}$, where $a_t = \operatorname{act}(\text{gate}(x_t)) \odot \text{up}(x_t) \in \mathbb{R}^I$ is the input to `down_proj` (with the model's activation function and any `swiglu_limit` clamping applied during the forward pass).
- $\text{count}[j] = |\mathcal{T}_e| \ge 0$ (may be `0` if no calibration token routed to expert `e`, in which case row `j` of `gu` and `down` is all zeros).

Encoders divide by `count[j]` to obtain the empirical mean Hessian $H / \text{count}$ before applying shrinkage and damping (see `exlr8enc` `src/hess.rs` and `hess.py`).

### Tensors

| tensor | dtype | shape | bytes | contents |
|---|---|---|---|---|
| `gu` | `F32` | `[n, T(H)]` | `4 · n · H(H+1)/2` | Packed upper triangle of $H_{\text{gu}}[A+j]$ in row-major order |
| `down` | `F32` | `[n, T(I)]` | `4 · n · I(I+1)/2` | Packed upper triangle of $H_{\text{down}}[A+j]$ in row-major order |
| `count` | `I64` | `[n]` | `8 · n` | Number of calibration tokens routed to expert `A + j` (`>= 0`) |

No other tensors are defined in `hess-LLL-eAAA-BBB.safetensors`. Because each tensor has leading dimension `n`, the site slices row `j = e - A` of each tensor directly with three contiguous HTTP byte ranges (`site/src/hess.ts:row`) to serve `hess-LLL-eEEE-EEE.safetensors` (`n = 1`).

### Upper-triangle layout

Each symmetric $d \times d$ matrix $M$ is stored as a 1-D vector $v$ of length $T(d) = d(d+1)/2$ containing entries $M[r, c]$ for $0 \le r \le c < d$ in **row-major** order — identical to `torch.triu_indices(d, d)`:

- Row `0`: `(0, 0), (0, 1), ..., (0, d-1)` (`d` entries)
- Row `1`: `(1, 1), (1, 2), ..., (1, d-1)` (`d - 1` entries)
- ...
- Row `d-1`: `(d-1, d-1)` (`1` entry)

Closed-form flat index for $0 \le r \le c < d$:

$$\operatorname{idx}(r, c) = \frac{r (2d + 1 - r)}{2} + (c - r)$$

Example for a $3 \times 3$ symmetric matrix:

```text
M = [[1, 2, 3],
     [2, 4, 5],    ==>    v = [1, 2, 3, 4, 5, 6]
     [3, 5, 6]]
```

Unpacking $v$ (length $m = T(d)$) back to full symmetric $d \times d$ `F32` (matching `exlr8enc/src/hess.rs:unpack` and `hess.py:unpack`):

1. Recover dimension $d = (\lfloor\sqrt{8m + 1}\rfloor - 1) / 2$ using exact integer square root, and verify $d(d+1)/2 = m$.
2. For every $0 \le r \le c < d$, set $M[r, c] = M[c, r] = v[\operatorname{idx}(r, c)]$.

### Reference sizes

| model | `H` | `T(H)` | `I` | `T(I)` | per expert | 32-expert file | full layer (`E` experts) |
|---|---|---|---|---|---|---|---|
| GLM-5.3 (`E = 256`) | 6144 | 18,877,440 | 2048 | 2,098,176 | 83.90 MB | 2.68 GB | 21.48 GB |
| DeepSeek-V4.1-Flash (`E = 384`) | 5120 | 13,109,760 | 2304 | 2,655,360 | 63.06 MB | 2.02 GB | 24.22 GB |
| Qwen3.8-Flash-Next (`E = 512`) | 2560 | 3,278,080 | 640 | 205,120 | 13.93 MB | 0.45 GB | 7.13 GB |

(Decimal MB/GB = $10^6 / 10^9$ bytes, excluding the small JSON header.)

## `hess-LLL-dense.safetensors` (and `-dense-heldout`, `-shared`)

Holds **unnormalized** full symmetric `F32` input Hessians $\sum x_t x_t^\top$ (`[d, d]`, row-major) for layer-level dense and shared-expert projections, paired with 1-element `I64` token count tensors `count_<name>` (`[1]`).

### Tensors

| tensor | dtype | shape | requirement | contents |
|---|---|---|---|---|
| `mlp_in` | `F32` | `[H, H]` | **required** | Pooled MoE input Hessian $\sum x_t x_t^\top$ over all calibration tokens. Used both as the routed-expert shrinkage prior ($P = \text{mlp\_in} / \text{count\_mlp\_in}$) and as the shared expert's `H_gu` |
| `count_mlp_in` | `I64` | `[1]` | **required** | Token count behind `mlp_in` (`> 0` for valid calibration files) |
| `shared_down_in` | `F32` | `[I, I]` | **required when `shared_expert` is true** | Shared expert `down_proj` input Hessian $\sum a_t a_t^\top$ |
| `count_shared_down_in` | `I64` | `[1]` | **required when `shared_expert` is true** | Token count behind `shared_down_in` (required by `site/src/hess.ts` when slicing shared claims; local `exlr8enc` falls back to `count_mlp_in` if absent) |
| `<other>` | `F32` | `[d, d]` | optional (model-specific) | Additional dense-projection input Hessians (e.g. `attn_in`, `q_resid`, `kv_latent`, `o_in`, `dense_down_in`, `eh_in`) |
| `count_<other>` | `I64` | `[1]` | paired with `<other>` | Token count behind `<other>` |

Every matrix tensor `<name>` MUST be square (`shape == [d, d]`), symmetric, finite `F32`, and paired with `count_<name>` of dtype `I64` and shape `[1]`.

Minimum payload size of `mlp_in` + `shared_down_in` + counts:

- GLM-5.3: 167.77 MB
- DeepSeek-V4.1-Flash: 126.09 MB
- Qwen3.8-Flash-Next: 27.85 MB

### Site-synthesized `hess-LLL-shared.safetensors`

When an encoder fetches `GET /api/claims/{job}/{layer}/{expert}/hess/shared` (`site/src/hess.ts:sharedHessians`), the site range-reads only the required tensors from `hess-LLL-dense.safetensors` and streams them as `hess-LLL-shared.safetensors`:

- **Routed slot (`shared == false`)**: `mlp_in`, `count_mlp_in`.
- **Shared slot (`shared == true`)**: `mlp_in`, `count_mlp_in`, `shared_down_in`, `count_shared_down_in`.

The synthesized header preserves every `__metadata__` entry from `hess-LLL-dense.safetensors` (including `hessian_id`, `weights_id`, `layer`, `layout`, `sequences`) and adds `source: "hess-LLL-dense.safetensors"`.

### `hess-LLL-dense-heldout.safetensors`

Optional output of `calib_forward.py --dense-heldout`. Uses the exact same tensor names, shapes, dtypes, and header schema as `hess-LLL-dense.safetensors`, except `sequences` is `"held out"` and sums are accumulated over the held-out sequences instead of the calibration sequences.

## `inputs-LLL.safetensors` (held-out expert inputs)

Captures the MoE block's input activations and router outputs on the held-out sequences (sequences excluded from Hessian accumulation). Used on the job owner's box to compute error tables (`encode_v2.py`) and to verify uploaded pieces on unseen inputs (`quantfeed` C6, see [`PIECE.md#feeder-error-metrics-c5--c6`](PIECE.md#feeder-error-metrics-c5--c6)).

### Tensors

| tensor | dtype | shape | contents |
|---|---|---|---|
| `x` | `BF16` | `[n_tok, H]` | Flattened MoE input activations (post-norm / post-HC-mix, identical to the input fed to routed and shared `gate_proj`/`up_proj`) across all `heldout_sequences` |
| `topk_idx` | `I32` | `[n_tok, top_k]` | Selected routed expert indices for each token (`0 <= idx < E`) |
| `topk_w` | `F32` | `[n_tok, top_k]` | Corresponding router weights for each `(token, slot)` entry in `topk_idx` |

Consistency rules:

- `x.shape[0] == topk_idx.shape[0] == topk_w.shape[0] == n_tok >= 1`.
- `x.shape[1] == H`.
- `topk_idx.shape[1] == topk_w.shape[1] == top_k >= 1`.
- Every entry of `topk_idx` satisfies `0 <= topk_idx[t, k] < E`.
- Every entry of `x` and `topk_w` is finite.
- For standard decoder layers with sequence length `S`, `n_tok = heldout_sequences * S`. For an MTP layer that predicts shifted next tokens (e.g. GLM-5.3 layer 78), `n_tok = heldout_sequences * (S - 1)`.

### Selecting inputs for a slot (held-out verification)

- **Routed expert `e` (`0 <= e < E`)**: Select all token indices `t` where `topk_idx[t, k] == e` for some `k`, with routing weight `w[t] = sum_{k: topk_idx[t, k] == e} topk_w[t, k]`. Let `X = x[tok]` in `F32`.
- **Shared expert**: Select all `n_tok` tokens with uniform weight `1.0`.
- **Down projection inputs**: Compute intermediate activations `A = act(X @ W_ref_gate) * (X @ W_ref_up)` (applying `swiglu_limit` if configured) using the reference checkpoint's `gate` and `up` weights, then evaluate `heldout_err` against `W_ref_down` and `W_decoded_down`.

## Validation matrix

| check | calibration writer | site (`hess.ts` / `window.ts`) | encoder (`exlr8enc`) | job feeder (`quantfeed`, C4/C6) |
|---|---|---|---|---|
| Filename matches `hess-LLL-eAAA-BBB`, `hess-LLL-dense`, or `inputs-LLL` | Yes | Yes (`window.ts` on registration; `hess.ts` on lookup) | Yes | Yes |
| Valid safetensors framing, contiguous offsets, exact tensor byte lengths | Yes | Yes (checks sliced tensor byte lengths on serve; size match on register) | Yes | Yes |
| Required tensors present with exact `dtype` and `shape` | Yes | Yes (`gu`/`down`/`count` or `mlp_in`/`count_mlp_in`[/`shared_down_in`/`count_shared_down_in`]) | Yes | Yes |
| `layer` metadata matches `int(LLL)` | Yes | — | Yes (`hess.rs:open`) | Yes |
| `experts` metadata matches `A-B` and tensor leading dim `n` | Yes | Rewritten to `<e>-<e>` on slice | — | Yes |
| `layout` matches `"upper triangle, row-major, fp32"` / `"full, fp32"` | Yes | — | — | Yes |
| `hessian_id` present and equals job's `hessian_id` | Yes (required flag) | Preserved on slice | — (can verify against claim) | Yes (C4 before uploading to store; C6 on `inputs-LLL`) |
| `weights_id` present, 64 lowercase hex, equals checkpoint `weights_id` | Yes (computed from `--model` or `--weights-id`) | Preserved on slice | — (can verify against checkpoint) | Yes (C4 before uploading to store; C6 on `inputs-LLL`) |
| Full routed coverage `0..E-1` + dense file before layer opens | — | Yes (`window.ts:markComplete`) | — | Yes |
| `inputs-LLL.safetensors` kept off public Hessian store | Yes (local `--out`) | Never registered (`parseHessKey` rejects) | Never fetched | Yes (local only) |

## Test vectors

### `weights_id`

Given raw file contents (including the trailing LF `0x0a` on each):

- `config.json`: `{"hidden_size":128}\n` (20 bytes)
- `model.safetensors.index.json`: `{"weight_map":{}}\n` (18 bytes)

```text
weights_id = "24b0d8c953197327441b27e43ecf94a2f3a7adbaf84b22c9ab0123b7c22519a1"
```

### Upper-triangle pack / unpack

For $d = 3$ ($T(3) = 6$), packed vector `v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]` (`F32`) unpacks to:

```text
row 0: [1.0, 2.0, 3.0]   # idx(0,0)=0, idx(0,1)=1, idx(0,2)=2
row 1: [2.0, 4.0, 5.0]   # idx(0,1)=1, idx(1,1)=3, idx(1,2)=4
row 2: [3.0, 5.0, 6.0]   # idx(0,2)=2, idx(1,2)=4, idx(2,2)=5
```
