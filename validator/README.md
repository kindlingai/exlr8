# exlr8 fileset validator

`validate.py` checks a release directory against [FORMAT.md](../FORMAT.md). The validator reads safetensors headers and
the small index tables with the Python standard library, so it runs on a host without numpy, torch or safetensors.

```
python -m validator.validate RELEASE_DIR            # fast: structure, no hashing
python -m validator.validate RELEASE_DIR --full     # also SHA-256 of every weight file
```

Run the module from this repository's root. `RELEASE_DIR` can be a local copy, a Hugging Face snapshot directory or
any other tree of symlinks. The validator follows symlinks.

| option | meaning |
|---|---|
| `--full` | hash every weight file (streamed), compare each hash with its sidecar, and compute each `set_id` from the hashes |
| `--hashes DIR` | read sidecars from `DIR`, e.g. sidecars fetched from the Hub for a local copy (default `RELEASE_DIR/hashes`) |
| `--jobs N` | files hashed at once in `--full` mode (default 2) |
| `--details N` | findings printed per check (default 10, `-1` for all) |
| `--progress` | print hashing progress to stderr |

Exit status 0 means no check failed (warnings allowed), 1 means at least one failed, 2 is a usage error.

## Checks

| check | fails on | warns on |
|---|---|---|
| support files | a missing or unparsable `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` or `LICENSE` | a model size missing from `config.json` |
| layout | a missing dense layer file (0 to `num_hidden_layers + num_nextn_predict_layers - 1`), vocab file or MoE index file. Any file in `dense/`, `experts/`, `manifests/` or `index-slices/` that FORMAT.md does not name | unknown top-level files or directories |
| safetensors | a file that does not parse, tensors with gaps or overlaps, a size that does not match the header. In exlr8 files, data or tensors not at multiples of 65,536, non-U8 pads, wrong `format`/`scheme` metadata | |
| dense records | a record that is not I64 [8] or has bad fields. A missing, mistyped or misshaped data tensor or checksum for its format. Stray data tensors. `layer` metadata or model sizes that disagree with the file name or `config.json`. `source_names`/`source_shapes` that disagree with the records. Split parts that do not cover the matrix | |
| vocab | missing `embed_tokens`, `lm_head` (unless tied) or `norm.weight`, or shapes that disagree with `config.json` | |
| optional members | `dense/vision-bf16.safetensors` with no tensors, tensors outside `model.visual.*`, or exlr8 records | |
| expert indexes | metadata (`layer`, sizes, `experts`, `widths`, `trellis_files`) that disagrees with `config.json`; a missing, misshaped or unexpected tensor | |
| expert trellis files | a missing `kK` file for a stored width, a `kK` file for a width the index does not store, wrong metadata or tensor sizes | |
| expert offsets | an index entry that does not point at unit u of expert e in its `kK` file, or a unit length other than H·16·K | |
| error tables | partial or misshaped `err_gate`/`err_up`/`err_down`/`heldout_tokens` | an index file with no error tables |
| manifests | wrong `format`/`name`, a missing or extra MoE layer, width tables of the wrong shape, a width the index does not store, a referenced weight file (dense, index or `kK`) that is missing | |
| index slices | wrong metadata, tensor shapes that disagree with the sharding rule, a (tp, k) group with ranks missing | |
| hash sidecars | a malformed sidecar, a size that differs from the file, a SHA-256 mismatch (`--full`), a sidecar with no file | a weight file with no sidecar |
| weight set checksums | a manifest or slice with no `set_id`, a `set_id` that does not match the files, an `index_sha256` that does not match an index file | a release with no `set_id` at all. A `set_id` that matches only without an optional member |

Fast mode checks sidecar sizes and computes `set_id` from the sidecar hashes, when every member has one. `--full`
computes it from the files. The expert and dense xxhash64 checksums are not recomputed; `loader/exlr8.py --verify`
does that per layer.

"Error tables" are the per-unit held-out error tensors in each index file (FORMAT.md, `PkK.err_*`). The format has no
separate error sidecar file.

## Tests

```
python -m unittest discover -s validator/tests -t .
```

`tests/fileset.py` builds a small valid fileset (H = 256, 4 experts, 5 layers). Each test changes one thing and checks
that the validator reports it. One test writes a layer with `exlr8writer` and is skipped when numpy or xxhash is not
installed.
