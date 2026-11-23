# exlr8 quant site/encoder HTTP protocol (v1)

This specification documents the HTTP interface between a quant coordination site (such as `https://quant.kindlingai.dev`) and volunteer encoders (such as `exlr8enc site`) as implemented today.

It describes existing v1 behavior only. Protocol version negotiation, presigned Hessian GETs, device-code login, and job-described checkpoint layouts belong to later revisions.

Related specifications:

- Piece file format (`exlr8-v2-job`): [`PIECE.md`](../PIECE.md)
- Release weight format and trellis decode: [`FORMAT.md`](../FORMAT.md)
- Machine-readable JSON Schemas (Draft 2020-12): [`schemas/`](schemas/)

## Scope

This document covers the public discovery endpoints and the authenticated volunteer claim, Hessian download, and piece upload endpoints. Site administration (`/api/admin/*`), browser login (`/auth/*`, `/api/me`), API key management (`/api/keys`), binary distribution (`/exlr8enc/*`), and web UI/social card routes are out of scope.

## Conventions

### Base URL and encoding

- Default base URL: `https://quant.kindlingai.dev` (no trailing slash).
- All JSON requests and responses use UTF-8 `application/json`.
- Timestamps (`expires_at`, `added_at`, `last_upload`) are Unix epoch milliseconds (non-negative integers).
- Binary endpoints (`hess/expert`, `hess/shared`, `piece`) carry raw safetensors files (`application/octet-stream`).
- Every 4xx and 5xx API response returns `application/json` matching [`error.response.schema.json`](schemas/error.response.schema.json):

```json
{"error": "human-readable reason"}
```

### Authentication

Volunteer encoders authenticate with a per-user API key in the `Authorization` header:

```http
Authorization: Bearer <api-key>
```

- Keys are created from an approved account's web session (prefix `eq_`).
- If an account is unapproved or its approval is revoked (and the account is not a site admin), its API keys authenticate as **no caller** (protected endpoints return `401`; public endpoints behave anonymously).
- **Origin binding**: An encoder MUST send its `Authorization` header only to URLs on the configured site origin. Presigned upload URLs returned by `POST .../piece-url` point to an external object store and carry their own SigV4 query signature; the encoder MUST NOT send the site `Authorization` header to those URLs.
- Browser session cookies (`qs=...`) are also accepted by the site; any non-`GET` request authenticated by cookie requires the header `X-Quant: 1` (else `403`). Encoders use Bearer API keys, not cookies.

### Slots, models, and legacy paths

A job encodes one model's MoE layers (`moe_layers`, default `0..layers-1`). Dense layers outside `moe_layers` have no slots and return `404` on slot operations.

Within each MoE layer `L`:

- Routed experts are slot indices `0 .. routed_experts - 1`.
- When `shared_expert` is `true`, the shared expert is slot index `routed_experts` (so `experts = routed_experts + 1` slots per layer; `shared = routed_experts`).
- When `shared_expert` is `false`, there is no shared expert slot (`experts = routed_experts`; `shared = -1`).

Slot endpoints have the canonical form `/api/claims/{job}/{layer}/{expert}[/...]`. For backwards compatibility with single-job encoders, the site also accepts `/api/claims/{layer}/{expert}[/...]` and routes it to `default_job` whenever the first path segment after `/api/claims/` consists solely of ASCII digits (`^[0-9]+$`). Job names MUST NOT consist solely of ASCII digits.

### Caching and the no-slots latch

- Anonymous `GET /api/jobs` and `GET /api/grid` responses are edge-cached for `PUBLIC_CACHE_SECONDS` (default 20 s) with `Cache-Control: public, max-age=<N>`. Authenticated requests bypass the anonymous cache (encoders polling status for metrics should call `GET /api/jobs` and `GET /api/grid` without `Authorization` to hit the cache).
- **No-slots latch on `POST /api/claims`**: When a `POST /api/claims` call fails with `409` whose error starts with `"no open slots"`, the site caches that `409` message per edge POP for up to 1 hour, keyed by `(job or sorted weights + job_version, layer or "any", sorted codebooks or "legacy")` and a cache generation token. Subsequent `POST /api/claims` calls with the same key are answered with that cached `409` **before authentication or D1 queries run**. Admin mutations and claim releases bump the generation token in that POP; claim expirations do not. Requests with invalid `count`, `expert` with `count > 1`, or an unknown `job` bypass the latch so validation errors still fire.

### Encoder compatibility

The site enforces three compatibility gates on `POST /api/claims` (and repeats the encoder check on piece uploads):

1. **Encoder identity and encoding version (`encoder`, `encoding`)**:
   - `encoder` is the encoder's git commit hash (7–40 lowercase hex chars; `exlr8enc` sends the full 40-char commit in claims), optionally suffixed with `-unchecked` (accepted) or `-dirty` (refused with `426`). Missing or empty `encoder` returns `426`.
   - When `encoding` is sent (a positive integer, `encode::ENCODING`), the site strips any `-unchecked` suffix, requires the commit to match `^[0-9a-f]{7,40}$` (else `400`), and accepts the build if `encoding >= MIN_ENCODING` (currently `1` in production), regardless of whether the commit is in `GET /api/encoders`. If `encoding < MIN_ENCODING`, the site returns `426` naming the required minimum. Note: because the site shares its encoder checker with piece metadata, it also accepts 1–9 digit decimal strings for `encoding`, though claims send a JSON integer.
   - When `encoding` is omitted (pre-versioned encoders), the stripped commit must match a row in `GET /api/encoders` (exact match for 40 chars, prefix match for 7–39 chars); otherwise `426`.
2. **Job format version (`job_version`)**:
   - Each job declares `version` (integer `>= 1`, default `1`). Encoders send the highest version they speak as `job_version` (absent defaults to `1`).
   - Targeting a single job whose `version > (job_version ?? 1)` returns `426`. Subscribing via `weights` silently skips jobs whose `version` exceeds `job_version`.
3. **Codebooks (`codebooks`)**:
   - Supported v1 codebooks: `"mcg"`, `"mul1"`, `"pmcgx"`.
   - When `codebooks` is sent, a layer is eligible iff every width's codebook in that layer's plan appears in `codebooks`.
   - When `codebooks` is omitted (legacy encoder), only layers whose plan uses uniform `pmcgx` with `refine: [64, 32, 96]` are eligible.

---

## Endpoints

| Method & Path | Auth | Purpose | Request Schema | Response Schema |
|---|---|---|---|---|
| `GET /api/config` | none | Models, jobs, and codebook prefixes (no stores/secrets) | — | [`config.response`](schemas/config.response.schema.json) |
| `GET /api/jobs` | none | Per-job progress counts and weights refs for active jobs | — | [`jobs.response`](schemas/jobs.response.schema.json) |
| `GET /api/grid[?job=J]` | optional | Slot grid, ready/Hessian layers, and layer plans for one job | — | [`grid.response`](schemas/grid.response.schema.json) |
| `GET /api/encoders` | none | Legacy accepted encoder commit allowlist | — | [`encoders.response`](schemas/encoders.response.schema.json) |
| `POST /api/claims` | approved | Claim one slot or a batch of compatible slots | [`claims.post.request`](schemas/claims.post.request.schema.json) | [`claims.post.response`](schemas/claims.post.response.schema.json) |
| `GET /api/claims` | approved | List caller's live unexpired claims | — | [`claims.get.response`](schemas/claims.get.response.schema.json) |
| `DELETE /api/claims[?job=J]` | approved | Release all caller's live claims (or those in job `J`) | — | [`claims.delete.response`](schemas/claims.delete.response.schema.json) |
| `DELETE /api/claims/{job}/{layer}/{expert}` | holder | Release one live claim | — | [`claim.delete.response`](schemas/claim.delete.response.schema.json) |
| `GET /api/claims/{job}/{layer}/{expert}/hess/expert` | holder | Download routed expert's Hessians (safetensors) | — | binary safetensors |
| `GET /api/claims/{job}/{layer}/{expert}/hess/shared` | holder | Download layer shared Hessians (safetensors) | — | binary safetensors |
| `PUT /api/claims/{job}/{layer}/{expert}/piece` | holder\* | Upload encoded piece through the site | binary safetensors | [`piece.put.response`](schemas/piece.put.response.schema.json) |
| `POST /api/claims/{job}/{layer}/{expert}/piece-url` | holder\* | Request a presigned S3 PUT URL for the piece | [`piece-url.post.request`](schemas/piece-url.post.request.schema.json) | [`piece-url.post.response`](schemas/piece-url.post.response.schema.json) |
| `POST /api/claims/{job}/{layer}/{expert}/piece-done` | holder\* | Verify and record a presigned upload | [`piece-done.post.request`](schemas/piece-done.post.request.schema.json) | [`piece-done.post.response`](schemas/piece-done.post.response.schema.json) |

\* Piece upload endpoints accept an expired claim as long as no other user has claimed the slot since.

---

### `GET /api/config`

Returns the site's model definitions, all jobs (including inactive ones), the default job name, and the codebook metadata prefixes required in piece files. Storage backends and secret names are stripped.

- **Auth**: None.
- **Status**: `200 OK`.
- **Schema**: [`config.response.schema.json`](schemas/config.response.schema.json).

Fields:

- `default_job`: key in `jobs` used when a claim or legacy path omits a job name.
- `models`: map of model key to `{name, layers, hidden_size, moe_intermediate_size, moe_layers?, routed_experts, shared_expert, weights?}`. `hidden_size` and `moe_intermediate_size` are positive multiples of 128.
- `jobs`: map of job key to `{version, active, model, title?, hessian_id, widths, shared_widths, damping, codebooks}`.
  - `codebooks` is an ordered array of plan entries `{layers, codebook, refine?, widths?, shared_widths?, damping?}`. When multiple entries cover a layer, the **last** matching entry wins.
  - `damping` defaults to `0.025`.
- `codebooks`: map of short codebook name to the required prefix in piece metadata:
  - `"mcg"`: `"mcg 0xCBAC1FED"`
  - `"mul1"`: `"mul1 0x83DCD12D"`
  - `"pmcgx"`: `"pmcgx V2 L20 0xCBAC1FED/0xBCA14C51"`

---

### `GET /api/jobs`

Returns slot progress counts and checkpoint weight references for every **active** job (`active !== false`). Inactive jobs are omitted.

- **Auth**: None (anonymous replies are edge-cached for `PUBLIC_CACHE_SECONDS`, default 20 s).
- **Status**: `200 OK`.
- **Schema**: [`jobs.response.schema.json`](schemas/jobs.response.schema.json).

Each entry contains:

- `job`, `title` (`job.title ?? job`), `version` (`job.version ?? 1`).
- `model`: the model's display `name` (e.g. `"GLM-5.3"`), **not** the config key (`"glm-5.3"`). Contrast with `/api/config`, `/api/grid`, and `Claim.model`, which carry the config key.
- `layers`: number of MoE layers (dense layers excluded).
- `experts`: slots per MoE layer (`routed_experts + (shared_expert ? 1 : 0)`).
- `slots`: total slots (`layers * experts`).
- Slot counts (always summing to `slots`):
  - `uploaded`: slots with an uploaded piece.
  - `claimed`: slots with an unexpired claim.
  - `disabled`: disabled slots.
  - `open`: free slots in ready layers (claimable now).
  - `closed`: free slots in layers whose Hessians are present (`hessians = 1`) but claims are off (`ready = 0`).
  - `not_ready`: free slots in layers with neither Hessians nor `ready = 1`.
- `harvested`: subset of `uploaded` already harvested off the pieces store.
- `ready_layers`: count of layers with `ready = 1`.
- `done_layers`: count of layers where all `experts` slots are `uploaded`.
- `contributors`: count of distinct users with at least one `uploaded` slot.
- `last_upload`: Unix ms timestamp of most recent upload in the job, or `null`.
- `weights`: `{hf, revision, id?}` or `null`. Encoders running in `--hf-cache` mode use this list to discover checkpoints for jobs with `open + closed + not_ready > 0`.
- `codebooks`: sorted unique codebook short names used anywhere in the job's plan.

---

### `GET /api/grid[?job=J]`

Returns the layer plans, ready/Hessian layer lists, and non-open slots of job `J` (or `default_job` when `job` is omitted). `exlr8enc` uses this as a fallback in its metrics poller when `--job` names an inactive job absent from `GET /api/jobs`.

- **Auth**: Optional. Anonymous replies are edge-cached for `PUBLIC_CACHE_SECONDS` and set every slot's `login` to `null` and `mine` to `0`. Authenticated replies include each slot's GitHub `login` and `mine` (`1` if owned by caller, else `0`).
- **Status**: `200 OK`, `404 Not Found` (unknown job).
- **Schema**: [`grid.response.schema.json`](schemas/grid.response.schema.json).

Fields: `job`, `jobs` (all configured job names), `model` (config key), `title`, `layers` (total model layers including dense), `experts`, `shared` (`routed_experts` or `-1`), `ready`, `hessians`, `moe` (`[first, last]` inclusive), `plans` (array of length `layers`, `null` for dense layers, else `{"k2": "pmcgx", ...}`), and `slots` (array of `[layer, expert, state, login, mine]` for non-open slots only).

---

### `GET /api/encoders`

Returns the list of legacy encoder commits accepted when a claim or piece omits `encoding`, ordered by `(added_at, commit_id)`.

- **Auth**: None.
- **Status**: `200 OK`.
- **Schema**: [`encoders.response.schema.json`](schemas/encoders.response.schema.json).

Each item has `commit_id` (40-char lowercase hex), `added_at` (Unix ms), and `note` (`string | null`).

---

### `POST /api/claims`

Claims one open slot, a specific named slot, or a batch of up to 16 compatible open slots.

- **Auth**: Approved user (`401` if unauthenticated or unapproved key; `403` if unapproved cookie session or missing `X-Quant: 1`). Note the [no-slots latch](#caching-and-the-no-slots-latch) may return `409` before auth.
- **Request schema**: [`claims.post.request.schema.json`](schemas/claims.post.request.schema.json).
- **Response schema (`201 Created`)**: [`claims.post.response.schema.json`](schemas/claims.post.response.schema.json).

#### Request fields

| Field | Type | Required | Constraints & Meaning |
|---|---|---|---|
| `encoder` | string | yes | Git commit hash (7–40 lowercase hex), optional `-unchecked` suffix. Missing/empty or `-dirty` -> `426`. |
| `encoding` | integer | no | Positive integer (`>= 1`). When present, accepted if `>= MIN_ENCODING` (else `426`); commit must be 7–40 lowercase hex (else `400`). When absent, commit must be in `GET /api/encoders` (else `426`). |
| `codebooks` | string[] | no | Subset of `["mcg", "mul1", "pmcgx"]` (else `400`). Absent restricts to legacy uniform `pmcgx` `[64, 32, 96]` layers. |
| `job_version` | integer | no | Integer `>= 1` (else `400`). Absent defaults to `1`. |
| `request` | string | no | `^[A-Za-z0-9_-]{8,64}$` (else `400`). Idempotency key; retrying returns live unexpired claims already created with this `request` in the candidate jobs (ordered by `layer, expert`) with `201`, even if the caller is now at `MAX_CLAIMS`. |
| `weights` | string[] | no | At most 16 lowercase 64-hex SHA-256 `weights_id`s (else `400`). Used for multi-job selection when both `job` and `layer` are omitted. |
| `cached` | object | no | Map of at most 16 `weights_id`s to arrays of at most 1024 non-negative integer layer indices (else `400`). Preferred layers tried first. |
| `job` | string | no | Target a specific job by key (unknown -> `404`). Allows claiming from an inactive job if named explicitly. |
| `layer` | integer | no | Target a specific MoE layer (out of range or dense -> `404`). Forces single-job mode (`job ?? default_job`), ignoring `weights`. |
| `expert` | integer | no | Target a specific expert slot (`0..experts-1`, out of range -> `404`). Requires `layer` (else `400`); incompatible with `count > 1` (else `400`). |
| `count` | integer | no | Integer `1..16` (else `400`). Claims up to `min(count, MAX_CLAIMS - held)` compatible slots. **Presence of `count` (even `count: 1`) wraps the `201` reply in `{"claims": [...]}`.** |

#### Job and slot selection rules

1. **Candidate jobs**:
   - If `job` or `layer` is provided, or `weights` is omitted: the single job `job ?? default_job`. If `(job_version ?? 1) < (job.version ?? 1)`, returns `426`.
   - Otherwise (`weights` provided, no `job` and no `layer`): every active job (`active !== false`) whose `model.weights.id` is in `weights` and whose `version <= (job_version ?? 1)`, with `default_job` first followed by other jobs in config order.
2. **Idempotent retry**: If `request` is provided and the caller holds any live unexpired `claimed` slot with `claim_request = request` in the candidate jobs, those slots are returned immediately with `201`.
3. **Quota check**: If the caller already holds `>= MAX_CLAIMS` live unexpired claims across all jobs, returns `409` with error `"you hold N claim(s), the limit; upload or release one first"`. `MAX_CLAIMS` is also enforced atomically inside SQLite on every claim insert so parallel requests cannot overshoot the limit.
4. **Named slot** (`layer` and `expert` both given): Checks layer readiness (`409`), codebook eligibility (`409`), and attempts to claim `(layer, expert)` (`409` if already claimed/uploaded/disabled or if quota filled concurrently).
5. **Automatic selection** (`expert` omitted):
   - Iterates candidate jobs in order; stops at the first job that yields at least one claimed slot.
   - Within a job, takes ready eligible layers (or `[layer]` if named), ordering layers listed in `cached[model.weights.id]` first (ascending), then all other ready eligible layers (ascending).
   - Free slots (no row in `job_slots`, or `state = 'claimed'` with `expires_at <= now`) within each layer are shuffled randomly to reduce contention across parallel encoders.
   - **Batch compatibility**: All slots returned by a `count` request belong to one job and share the same `(codebooks, refine)` layer plan so an encoder can run them in one GPU LDLQ batch. A batch may span layers only when their `(codebooks, refine)` plans match.
   - Up to 8 internal retry rounds handle concurrent races. If every attempt loses a race and zero slots were claimed (while still under `MAX_CLAIMS`), returns `503`.

#### Response (`201 Created`)

- When `count` was **omitted**: a single `Claim` object.
- When `count` was **present**: `{"claims": [Claim, ...]}` (1 to `count` items).

Each `Claim` object has:

- `job`, `job_version`, `model` (config key), `weights_id` (`string | null`), `layer`, `expert`, `shared` (`boolean`), `expires_at` (Unix ms; `claimed_at + CLAIM_HOURS * 3_600_000`, currently 12 hours).
- `hessians`:
  - Routed slot (`shared: false`): `{"expert": "<url>", "shared": "<url>"}`.
  - Shared slot (`shared: true`): `{"shared": "<url>"}` (`expert` is omitted).
- `upload`: URL for direct `PUT .../piece`.
- `upload_url` and `upload_done`: present **only** when the job's pieces store is `s3`; omitted otherwise.
- `encode`: resolved per-layer plan:
  - `hessian_id`: job's Hessian identifier.
  - `widths`: routed widths for this layer.
  - `shared_widths`: shared-expert widths for this layer.
  - `damping`: `sigma_reg` for this layer.
  - `codebooks`: map `"k2"`, `"k3"`, `"k4"` -> codebook short name for the union of `widths` and `shared_widths`.
  - `refine`: pmcgx refinement step list (`[]` when no width uses `pmcgx` or refinement is disabled).
  - `args`: flat list of alternating CLI flag/value strings (`--codebook`, `--hess-id`, `--widths`, `--shared-widths`, `--shrink`, `--shrink-min-erank`, `--shrink-skip-layers`, `--hess-min`, `--hess-full`, `--gss-width`, plus `--refine` iff any width uses `pmcgx`, plus `--damping` iff `damping !== 0.025`). The piece validator checks `encode_args` against these exact flags (see [`PIECE.md`](../PIECE.md#encode_args-and-pinned-flags)).

---

### `GET /api/claims`

Lists all live, unexpired claims held by the caller across all configured jobs, ordered by `claimed_at` ascending.

- **Auth**: Approved user (`401` / `403`).
- **Status**: `200 OK`.
- **Schema**: [`claims.get.response.schema.json`](schemas/claims.get.response.schema.json) (array of `Claim` objects).

---

### `DELETE /api/claims[?job=J]`

Releases all live, unexpired claims held by the caller (across all jobs, or restricted to job `J` when `?job=J` is provided) and invalidates the no-slots latch if at least one claim was released.

- **Auth**: Approved user (`401` / `403`).
- **Status**: `200 OK`, `404 Not Found` (unknown `job`).
- **Schema**: [`claims.delete.response.schema.json`](schemas/claims.delete.response.schema.json) (`{"released": N}` or `{"job": "J", "released": N}`).

---

### `DELETE /api/claims/{job}/{layer}/{expert}`

Releases one live, unexpired claim held by the caller and invalidates the no-slots latch.

- **Auth**: Approved user holding an **unexpired** claim on `(job, layer, expert)` (else `403`; invalid job/slot/dense layer -> `404`).
- **Status**: `200 OK`.
- **Schema**: [`claim.delete.response.schema.json`](schemas/claim.delete.response.schema.json) (`{"job", "layer", "expert", "released": true}`).

---

### `GET /api/claims/{job}/{layer}/{expert}/hess/expert`

### `GET /api/claims/{job}/{layer}/{expert}/hess/shared`

Downloads the claim's input Hessian safetensors files, synthesized on the fly via HTTP range requests from the job's Hessian stores.

- **Auth**: Approved user holding a live, **unexpired** claim on `(job, layer, expert)` (else `403`).
- **Availability**:
  - `hess/expert`: routed slots only (`shared === false`). Calling `hess/expert` on the shared expert slot returns `404`.
  - `hess/shared`: both routed and shared expert slots.
- **Response (`200 OK`)**:
  - `Content-Type: application/octet-stream`
  - `Content-Length: <bytes>`
  - `Content-Disposition: attachment; filename="<name>"`:
    - `hess/expert`: `hess-LLL-eEEE-EEE.safetensors` (zero-padded 3-digit layer and expert). Contains tensors `gu` (`[1, ...]`, F32 upper triangle), `down` (`[1, ...]`, F32 upper triangle), `count` (`[1]`, I64), and source `__metadata__` with `experts: "<E>-<E>"` and `source: "<source-filename>"`.
    - `hess/shared`: `hess-LLL-shared.safetensors`. Contains `mlp_in` and `count_mlp_in` for routed slots, plus `shared_down_in` and `count_shared_down_in` when `shared === true`, and source `__metadata__` with `source: "<source-filename>"`.
  - Encoders parse the filename from `Content-Disposition` (refusing any name containing `/`).
  - **Caching note**: For routed slots (`shared === false`), `hess/shared` is identical for every routed expert of the same `(job, hessian_id, layer)` and may be cached locally across routed slots. The shared expert's `hess/shared` includes `shared_down_in` and MUST NOT share a cache entry with routed slots.
- **Errors**: `401`, `403` (not claim holder or claim expired), `404` (invalid slot, `hess/expert` on shared slot, or Hessian file absent from all job stores), `500` (corrupt/missing tensor in store), `502` (upstream store error).

---

### `PUT /api/claims/{job}/{layer}/{expert}/piece`

Uploads an encoded piece safetensors file through the site into the job's pieces store.

- **Auth**: Approved user holding the slot's claim. **Expired claims are accepted** as long as no other user has claimed the slot since.
- **Idempotency**: If the slot is already `uploaded` by the same caller and `Content-Length` equals the recorded `size`, the site returns `200 OK` with the recorded `PieceReply`. (If `Content-Length` differs, falls through to claim check and returns `403`.)
- **Request**:
  - Header `Content-Length` is required and must be a positive integer (else `411`).
  - Body: raw `exlr8-v2-job` safetensors bytes (see [`PIECE.md`](../PIECE.md)).
  - Maximum size: `maxPieceBytes = 2 * sum(required tensor bytes) + 16 MiB` (else `413`). Through-site uploads are also bounded by the Worker runtime's 100 MB request limit; use `piece-url` / `piece-done` for larger pieces on `s3` stores.
- **Validation**: The site reads the 8-byte header length (`<= 1 MiB`) and JSON header (`400` if truncated, oversized, or invalid JSON), runs `pieceProblems` and `structureProblems` (see [`PIECE.md`](../PIECE.md#validation-matrix); any failure returns `422`), streams the body to the pieces store, and atomically transitions the slot from `claimed` to `uploaded`. If the claim was lost concurrently during upload, the stored object is deleted and the site returns `409`.
- **Response (`201 Created`, or `200 OK` on idempotent retry)**: [`piece.put.response.schema.json`](schemas/piece.put.response.schema.json) (`{job, layer, expert, size, md5}`). `sha256` is omitted on through-site uploads.
- **Errors**: `400`, `401`, `403`, `404`, `409` (claim lost during upload), `411`, `413`, `422`, `502`.

---

### `POST /api/claims/{job}/{layer}/{expert}/piece-url`

Requests a SigV4 presigned `PUT` URL to upload a piece directly to an `s3` pieces store, bypassing the Worker body size limit. Available when the claim includes `upload_url` and `upload_done`.

- **Auth**: Approved user holding the slot's claim (expired claims accepted if untaken).
- **Request schema**: [`piece-url.post.request.schema.json`](schemas/piece-url.post.request.schema.json) (`{"size": <bytes>}`, integer `> 8` and `<= maxPieceBytes`).
- **Behavior**: Issues a presigned `PUT` URL bound to `Content-Length: size`, valid for `clamp(ceil((expires_at - now)/1000), 3600, 604800)` seconds, and records the pending `(upload_key, upload_size)` on the slot. Calling `piece-url` again replaces the pending upload and deletes any prior unfinalized object.
- **Response (`200 OK`)**: [`piece-url.post.response.schema.json`](schemas/piece-url.post.response.schema.json):

```json
{
  "url": "https://...",
  "method": "PUT",
  "headers": {"Content-Length": "39123456"},
  "expires_in": 43200
}
```

The encoder then executes `PUT <url>` with the exact `size` bytes and `headers` (and **without** the site's `Authorization` header), followed by `POST .../piece-done`. Uploading any other byte length fails at the S3 store with `403`.

- **Errors**: `400` (`size` not integer `> 8`), `401`, `403` (no claim or lost claim), `404`, `409` (pieces store is not `s3`), `413` (`size > maxPieceBytes`).

---

### `POST /api/claims/{job}/{layer}/{expert}/piece-done`

Verifies a completed presigned upload in the pieces store and records the slot as `uploaded`.

- **Auth**: Approved user holding the slot's claim (expired claims accepted if untaken).
- **Idempotency**: If the slot is already `uploaded` by the same caller, returns `200 OK` with the recorded `PieceReply` immediately.
- **Request schema**: [`piece-done.post.request.schema.json`](schemas/piece-done.post.request.schema.json) (`{}` or `{"sha256": "<64-hex>"}`). `sha256` is optional; when provided it is stored and verified later by the harvester/feeder.
- **Behavior**:
  1. Requires a pending upload from `POST .../piece-url` (else `409 "ask for an upload URL first"`).
  2. Sends `HEAD` to the pieces store (missing -> `409 "the upload has not arrived"`).
  3. Checks `stat.size === upload_size` (mismatch -> deletes stored object, clears pending upload, returns `422`).
  4. Range-reads the safetensors header (`0..8`, then `8..8+n`; invalid safetensors/JSON -> deletes stored object, clears pending upload, returns `400` with `"piece rejected: ..."`).
  5. Runs `pieceProblems` and `structureProblems` (any failure -> deletes stored object, clears pending upload, returns `422` with `"piece rejected: ..."`).
  6. Atomically transitions the slot to `uploaded` (`201 Created`). If the claim was lost concurrently, deletes the stored object and returns `409`.
- **Response (`201 Created`, or `200 OK` on idempotent retry)**: [`piece-done.post.response.schema.json`](schemas/piece-done.post.response.schema.json) (`{job, layer, expert, size, md5, sha256?}`).
- **Errors**: `400`, `401`, `403`, `404`, `409`, `422`, `502`.

---

## Error semantics and client handling

| Status | Meaning | Typical `error` pattern / cause | Expected encoder action |
|---|---|---|---|
| `400` | Validation error | Bad request field (`encoding`, `codebooks`, `job_version`, `weights`, `cached`, `count`, `request`, `size`, `sha256`, `expert` without `layer`, `expert` with `count > 1`, non-hex `encoder` with `encoding`) or invalid/truncated safetensors header on upload | Fix request / abort (do not retry blindly) |
| `401` | Unauthenticated | Missing, invalid, or revoked API key | Abort and report key error |
| `403` | Forbidden | Unapproved cookie session, cookie mutation without `X-Quant: 1`, or caller does not hold a valid claim for the slot (including expired claim on Hessian fetch / release) | On slot routes: abandon slot; on auth errors: abort |
| `404` | Not found | Unknown job, invalid or dense layer, out-of-range expert, `hess/expert` on shared slot, missing Hessian file | Abort / report error |
| `409` | Conflict — claim cap | `POST /api/claims`: message starts with `"you hold "` (`"you hold N claim(s), the limit; ..."`) | If uploads are in flight, wait for one to finish and retry; otherwise abort and ask user to release claims |
| `409` | Conflict — no work | `POST /api/claims`: message starts with `"no open slots"` (latched), or `"layer L's Hessians are not ready"`, or `"layer L of job J uses ..."`, or `"layer L expert E is taken"` | Stop cleanly (or in `--watch` mode, drain in-flight work and sleep before polling again) |
| `409` | Conflict — upload state | `piece-url` on non-S3 store, `piece-done` before `piece-url` or before S3 PUT arrives, or claim lost during upload | Check upload flow / abandon lost slot |
| `411` | Length required | `PUT .../piece` without a positive `Content-Length` header | Send `Content-Length` |
| `413` | Payload too large | Piece exceeds `maxPieceBytes` | Abort (piece is malformed or wrong dimensions) |
| `422` | Unprocessable piece | Piece header failed `pieceProblems` or `structureProblems` (`"piece rejected: ..."`), including unaccepted `encoder_commit`/`encoding` in piece metadata, or presigned size mismatch | Release claim and abort (do not retry same bad piece) |
| `426` | Upgrade required | `POST /api/claims`: missing `encoder`, `-dirty` build, `encoding < MIN_ENCODING`, unlisted commit with no `encoding`, or `job_version < job.version` | Abort immediately and prompt user/launcher to upgrade encoder |
| `500` / `502` | Server / store error | Internal site error or upstream Hessian/pieces store error | Retry with exponential backoff (keeping the same `request` id on `POST /api/claims`) |
| `503` | Claim race lost | `POST /api/claims`: competing encoders took every candidate slot across all retry rounds | Retry with backoff |

---

## Typical encoder lifecycle

1. **Discover jobs** (optional, required for `--hf-cache`): `GET /api/jobs` (unauthenticated). For each active job with `open + closed + not_ready > 0` and non-null `weights`, fetch `config.json` and `model.safetensors.index.json` from Hugging Face and verify their combined SHA-256 equals `weights.id`.
2. **Claim**: `POST /api/claims` with `encoder`, `encoding`, `codebooks`, `job_version`, a fresh `request` id, `weights` (or `job`), optional `cached` layers, and optional `count` (`1..16`). Retry `5xx` with the **same** `request` id.
3. **Fetch Hessians**: `GET` every URL in `claim.hessians` (`expert` and `shared` for routed slots; `shared` only for shared slots), verifying each URL is on the site's origin and saving each file under its `Content-Disposition` filename.
4. **Encode**: Run the encode with `claim.encode.args` against the fetched Hessians and the checkpoint matching `claim.weights_id`, writing an `exlr8-v2-job` piece file ([`PIECE.md`](../PIECE.md)).
5. **Upload**:
   - Direct: `PUT <claim.upload>` with `Content-Length` and the piece bytes (retry `5xx`; a `200` confirms an already-recorded upload).
   - Presigned (when `claim.upload_url` is present): `POST <claim.upload_url>` with `{"size": N}` -> `PUT <reply.url>` with `reply.headers` and no site `Authorization` header -> `POST <claim.upload_done>` with `{"sha256": "..."}`.
6. **Release on failure or expiry**: If local work fails or aborts before upload, call `DELETE /api/claims/{job}/{layer}/{expert}` so the slot reopens immediately instead of waiting for `expires_at`.

---

## JSON Schema index

All schemas live in [`schemas/`](schemas/) and target JSON Schema Draft 2020-12:

- [`common.schema.json`](schemas/common.schema.json) — shared types (`Error`, `Codebook`, `Width`, `LayerRange`, `Sha256Hex`, `GitCommit40`, `EncoderId`, `WeightsRef`, `Model`, `PerWidthCodebooks`, `PlanEntry`, `EncodeBlock`, `Claim`, `PieceReply`)
- [`error.response.schema.json`](schemas/error.response.schema.json)
- [`config.response.schema.json`](schemas/config.response.schema.json)
- [`jobs.response.schema.json`](schemas/jobs.response.schema.json)
- [`grid.response.schema.json`](schemas/grid.response.schema.json)
- [`encoders.response.schema.json`](schemas/encoders.response.schema.json)
- [`claims.post.request.schema.json`](schemas/claims.post.request.schema.json)
- [`claims.post.response.schema.json`](schemas/claims.post.response.schema.json)
- [`claims.get.response.schema.json`](schemas/claims.get.response.schema.json)
- [`claims.delete.response.schema.json`](schemas/claims.delete.response.schema.json)
- [`claim.delete.response.schema.json`](schemas/claim.delete.response.schema.json)
- [`piece.put.response.schema.json`](schemas/piece.put.response.schema.json)
- [`piece-url.post.request.schema.json`](schemas/piece-url.post.request.schema.json)
- [`piece-url.post.response.schema.json`](schemas/piece-url.post.response.schema.json)
- [`piece-done.post.request.schema.json`](schemas/piece-done.post.request.schema.json)
- [`piece-done.post.response.schema.json`](schemas/piece-done.post.response.schema.json)
- [`piece-header.schema.json`](schemas/piece-header.schema.json) — structural schema for an `exlr8-v2-job` piece safetensors header (see [`PIECE.md`](../PIECE.md))
