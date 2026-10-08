# EXLR8 — a short explainer

## tl;dr

EXLR8 is a quantization format for serving models at low bitrates with fewer compromises on the
compute side.

On unified memory hardware, **decode speed is bound by how many bytes of expert weights you read
per token**, so the format is engineered around one question: *fewest bytes read, fewest surprises
at load time.*

The codec itself is borrowed: each 16×16 weight tile is EXL3's trellis code (QTIP-style). This is a
path through a 16-bit-state trellis where every weight contributes K new bits, chosen by Viterbi 
search.

What EXLR8 adds is:

1. **Codebook values are rounded to FP8 E4M3 inside the Viterbi search** — decoded weights are
   exactly FP8, so prefill runs on FP8 tensor cores.
2. **Every routed expert is stored at three widths (K2/K3/K4)**; the mix is chosen per expert-half
   *at load time* by a manifest — no re-encode.
3. **Rotation is shared, not per-expert**: one sign vector per layer on the hidden side, so the
   kernel rotates the token's activation once and reuses it across all 8 active experts.
4. **The layout is TP-native**: a single sharding rule serves TP=3–7 with zero re-layout, and a
   node can fetch only its own slices over HTTP range requests (~75 GB per node at TP=4 under the
   `home` manifest, instead of the whole checkpoint).

Decode quality comes from calibrated rounding (GPTQ/LDLQ against real activation statistics), which
is 2.3–33× better than round-to-nearest at the same bit width.

---

## The math, step by step

The first EXLR8 quant is GLM-5.3 from z.ai:
[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4).

### 0. The model it has to fit

GLM-5.3's MoE layers (3–77) have **256 routed experts, 8 active per token**; hidden dim 6144,
intermediate dim 2048. One expert is 37,748,736 weights (three 6144×2048/2048×6144 matrices:
gate, up, down). A layer's 256 experts take 4.83 GB at K4 (10.87 GB for all three widths) — the
design exists to make per-token reads out of that as small as the budget allows.

### 1. Rotation — the transform the quantizer sees

Each weight matrix `w (in, out)` is first put through an invertible transform on both sides. Each
side gets:

- a **per-channel sign** (`±1`),
- a **scale** (per-channel on the intermediate side; **constant per 128-block** on the hidden side),
- a fixed **128×128 Hadamard** (`exllamav3`'s, scaled 1/√128) applied to each consecutive
  128-channel block.

After scaling and rotating, the quantizer encodes the transformed matrix; the loader undoes the
transform at decode. Reconstruction is:

```
w = R( diag(suh) · L(T) ) · diag(svh)
```

where `T` is the tile decoded from the trellis, `L`/`R` apply the 128-block Hadamard on the
input/output side, and `suh`/`svh` are the stored side vectors (sign × scale, folded together).
A global scale `g` per (projection, width) — found by golden-section search on [0.1, 1.9] — is
folded into `suh` so decoded values come out at the right magnitude.

**The sharing trick.** The hidden side's signs (`signs.in`, `signs.out`, 6144 values each) are
**one vector per layer**, shared by all 256 experts, the shared expert, and all three widths
(sequences drawn from seed `1234 + layer`). The intermediate side keeps per-expert signs, but they
are folded into the stored per-channel vectors, so they live inside the weights and cost nothing at
runtime. 

Why: with 8 experts active per token, per-expert hidden-side rotations would force the
kernel to rotate the activation 8 times (or keep 8 rotated copies). With shared signs and
block-constant scales, the activation is rotated **once**, and one A operand serves all 8 experts'
matmuls. Measured cost: NMSE ratio 0.9996–1.0003 vs. per-expert signs (i.e. free); MoE prefill
23.07 → 18.67 ms at layer 40/M=4096 (16.5 ms with FP8 activations; reference kernel 31.8–36.9 ms).

### 2. The trellis code (EXL3/QTIP)

A 16×16 tile is encoded jointly, not weight-by-weight: it is a **path through a 16-bit-state
trellis**, each weight contributing K new bits along the path. The Viterbi search picks the path
minimizing the tile's squared error, with per-step values drawn from the **mcg codebook**
(multiplier `0xCBAC1FED`, a format constant). Joint encoding is the whole trick: the trellis's
state carries information between neighboring weights, so K bits per weight here buy more than K
independent bits would.

A tile of K-bit weights is `256 × K` bits = **32·K bytes**, packed exactly as exllamav3's
`pack_trellis` packs it: 16K little-endian int16 values in tensor-core order. Decoding is
exllamav3's `reconstruct` plus the E4M3 rounding below.

### 3. FP8-rounded codebook (the codec-level delta)

Every value the codebook produces is rounded to **E4M3 (round-to-nearest-even, saturating) inside
the Viterbi search**. The search therefore only considers paths whose decoded weights are exactly
representable in FP8, and prefill can feed decoded weights straight to FP8 tensor-core MMA.

Cost of doing it inside vs. after the search (NMSE ratio vs. fp16 codebook, lower is better):

| where rounded | K3 | K4 |
|---|---|---|
| inside the search (exlr8) | 0.9993 | 1.0000 |
| after the search | +4.0% error | +15.3% error |

A stock EXL3 decoder reading the same files without the rounding gets **+2.3% error** — the rounding
is part of the format, and loaders must apply it.

### 4. Calibration — the Hessians and LDLQ

Round-to-nearest ignores how weights are actually *used*. EXLR8 doesn't:

- For each routed expert e, calibration data gives
  `H_gu[e] = Σ x xᵀ` over tokens routed to e (x = the 6144-wide expert input, shared by gate and
  up), and `H_down[e] = Σ a aᵀ` (a = `silu(gate x) ⊙ up x`, 2048-wide). The calibration set is
  977 rows / 2.0M tokens (+64 held-out rows); the least-routed expert still sees ~13k tokens
  (median ~61k). Hessians are damped (σ = 0.025 × mean diag) and optionally shrunk toward a pooled
  prior for thin experts.
- The Hessian is pulled into the quantizer's rotated domain (`H' = S D Had H Had D S`) and
  factorized in 16×16 blocks: `H = L D Lᵀ`.
- **LDLQ** then walks the input rows *from the last 16-row block to the first*: each block is
  quantized *after* adding the error already committed by the blocks behind it (fed back through
  `L`), and its own residual error is fed forward. Error cancellation along the input dimension is
  what calibrated rounding buys: **−14% held-out layer error vs. identity-Hessian encoding**
  (layer 40: 0.1644 → 0.1413), and round-to-nearest is **2.3–33× worse** than LDLQ at the same
  format. One reason RTN loses so badly: these Hessians are near rank-one (massive activations —
  one eigendirection holds 7–95% of the trace), and only feedback-based rounding can cancel error
  along that direction.

### 5. Widths — three encodes, one knapsack

Trellis codes **don't nest** (a 2-bit path is not a prefix of the 3-bit path), so each expert is
encoded **separately at K2, K3 and K4** — one file set per width. A layer's 256 experts cost
2.42 / 3.62 / 4.83 GB at K2/K3/K4; all three, all 75 layers, ≈ 819 GB on disk. That's the price of
the next bullet.

At **load time**, a manifest picks the width per unit:

- **gate/up: per half** — a "half" is 128 of the expert's 2048 intermediate channels (the storage
  unit; halves pair into 256-channel fragments aligned to 64 KiB).
- **down: per expert** (LDLQ error feedback crosses halves along the input dimension — down's input
  *is* the intermediate — so its width must be uniform across the expert).

Choosing widths is a knapsack under a per-rank memory budget, driven by per-half/per-expert error
tables recorded at encode time. The shipped `home` manifest (64 experts at K4, 192 at K3 ≈ 3.25
bits/weight) holds 3.93 GB per layer ≈ **75 GB per node at TP=4**.

### 6. Sharding — the half rule

For TP = N, half h of expert e in layer L lives on rank:

```
rank = (16e + h + L) mod N
```

One rule serves every N from 3 to 7; nothing is rewritten per topology. At TP=4 each rank owns
exactly 4 of every expert's 16 halves (rank r owns h0, h0+4, h0+8, h0+12 with h0 = (r − L) mod 4).
Kernels get a pointer table `ptr[e][h]` (null = not owned) instead of contiguous ranges, and per-half widths are
fine because the table is re-read every launch — which also later allows paging weights in from
mapped files.

---

## Why not just use EXL3?

| | EXL3 | EXLR8 |
|---|---|---|
| Code | trellis, K bits/weight, mcg codebook | same codec + E4M3 rounding inside the search |
| Codebook values | fp16 | exactly E4M3 (prefill on FP8 MMA) |
| Granularity | one width per tensor | three widths stored; per-half manifest at load |
| Rounding | LDLQ against calibration Hessians, per tensor | calibrated LDLQ with pooled/damped Hessians, shared across widths |
| Rotation | per-tensor side vectors | shared per-layer hidden-side rotation → one activation rotate per token |
| Topology | format knows nothing about TP | one sharding rule for TP=3–7; range-fetchable slices |

The one-line version: **EXL3 is a codec; EXLR8 is a codec plus the entire serving geometry of a
744B MoE on GB10 — bytes read per token are the design target, and every deviation from stock EXL3
serves that.**
