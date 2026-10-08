# Calibration set calib-2m-v1 (public part)

calib-2m-v1 is the calibration set used to quantize GLM-5.3 to exlr8 v2. It has 1,041 rows of 2,048 tokens,
tokenized with GLM-5.3's tokenizer (end-of-text id 154820). Chat records are rendered with the model's chat template.

- Rows 0-976 (977 rows, 2,000,896 tokens) are calibration rows. The BF16 model ran over them, and the inputs routed to
  each expert gave the Hessians for exlr8 v2's calibrated rounding (LDLQ). See section 4 of the main README.
- Rows 977-1040 (64 rows) are held-out rows. They come from documents the calibration rows never see. They were used
  only to score quantization error, never to build Hessians.

About 37% of the mix came from private Claude Code session transcripts. They are not published, so this directory
cannot reproduce calib-2m-v1 exactly. See [sources/claude-code-sessions/README.md](sources/claude-code-sessions/README.md).

## Files

| file | what it is |
|---|---|
| `sources/<name>.jsonl`, `<name>.meta.json` | the 12 public sources, byte-identical to the files used (sha256 in the manifest) |
| `sources/commitpackft-attribution.jsonl` | repository, commit, path and licence of every commitpackft record in `code.jsonl` and `structured.jsonl` |
| `calib-2m.npy.json` | the manifest from `calib_tokens.py`; the private source is reduced to its counts |
| `calib-2m-public.npy` | int32 `[658, 2048]`: the rows of calib-2m.npy that came from public sources (617 calibration, 41 held-out), in their original order |
| `rows.json` | for each row of `calib-2m-public.npy`: its index in calib-2m.npy (`original_row`, 0-1040; 977 and up are held-out), `source`, `heldout`, and `source_row`, its index among that source's calibration or held-out rows as `calib_tokens.py` packs them |

## Mix

| source | dataset | licence | weight | share | calibration rows | held-out rows |
|---|---|---|---|---|---|---|
| cc (private) | Claude Code session transcripts | not published | 35 | 36.8% | 360 | 23 |
| code | bigcode/commitpackft | MIT (per-file licences below) | 15 | 15.8% | 154 | 10 |
| prose | HuggingFaceFW/fineweb-edu | ODC-By 1.0 | 15 | 15.8% | 154 | 10 |
| reasoning | open-r1/OpenR1-Math-220k + open-thoughts/OpenThoughts-114k | Apache-2.0 | 10 | 10.5% | 103 | 7 |
| zh | HuggingFaceFW/fineweb-2 | ODC-By 1.0 | 8 | 8.4% | 82 | 5 |
| structured | NousResearch/hermes-function-calling-v1 + bigcode/commitpackft | Apache-2.0 + MIT (per-file licences below) | 5 | 5.3% | 52 | 3 |
| fr, de, es, pt, it, ja | HuggingFaceFW/fineweb-2 | ODC-By 1.0 | 1 each | 1.1% each | 11, 11, 10, 10, 10, 10 | 1 each |
| ko | HuggingFaceFW/fineweb-2 | ODC-By 1.0 | 1 | 1.1% | 10 | 0 |
| total | | | 95 | | 977 | 64 |

Each `<name>.meta.json` says what was taken from its dataset and how it was rendered.

Pinned revisions:

| dataset | revision |
|---|---|
| bigcode/commitpackft | `fc56fe33c030c6daa414c2b112c932b8eed085e6` |
| HuggingFaceFW/fineweb-edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| HuggingFaceFW/fineweb-2 | `af9c13333eb981300149d5ca60a8e9d659b276b9` |
| open-r1/OpenR1-Math-220k | `e4e141ec9dea9f8326f4d347be56105859b2bd68` |
| open-thoughts/OpenThoughts-114k | `bd093c3994fd54d2390985b66988ddf282a55eb6` |
| NousResearch/hermes-function-calling-v1 | `dae3e1d28cfbcf4b915c04ea1e072030529b4bda` |

## Licences and attribution

- `prose.jsonl` contains data from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by
  Hugging Face, and `zh`, `fr`, `de`, `es`, `pt`, `it`, `ja` and `ko.jsonl` contain data from
  [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) by Hugging Face. Both are made available under
  the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/), and their
  use is also subject to [CommonCrawl's Terms of Use](https://commoncrawl.org/terms-of-use).
- `code.jsonl` and the file half of `structured.jsonl` come from
  [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) (MIT). Each file keeps the licence of the
  repository it came from. `sources/commitpackft-attribution.jsonl` lists the repository, commit, path and licence for
  each of the 1,994 records. Licences: MIT 1,004, Apache-2.0 534, BSD-3-Clause 166, BSD-2-Clause 66, AGPL-3.0 64,
  LGPL-2.1 54, MPL-2.0 47, ISC 29, Unlicense 14, CC0-1.0 6, Artistic-2.0 4, EPL-1.0 3, and 3 records CommitPackFT
  labels "unknown".
- The rest of `structured.jsonl` comes from
  [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) by Nous
  Research, and `reasoning.jsonl` comes from [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
  by Hugging Face and [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) by the
  OpenThoughts team. All three are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
  The records were changed: they were rendered as chat messages, with the reasoning trace as `reasoning_content`.

## Row provenance

Running `calib_tokens.py` again on the same 13 source files, with the arguments below, gives a calib-2m.npy (sha256
`a35da3634ef89ba87e177d25d4a6b8e783ce3e8a0dc6f8ad7080c3b7ee27f123`) and a manifest that are byte-identical to the ones
used. Each source was also packed on its own, and each of the 1,041 rows matched the rows of exactly one source. That is
how `rows.json` knows which rows are public. `calib-2m-public.npy` can be rebuilt from the public files alone: pack each
public source with `calib_tokens.py`'s `docs` and `split_rows`, using the row counts in the manifest, and place each
source's row `source_row` at `row`.

## Rebuilding

The scripts are `requant/calib_sources.py` and `requant/calib_tokens.py` in exlr8quant.

```sh
# Fetch the 12 public sources at the pinned revisions into sources/ (streaming).
uv run --with datasets --with pyarrow calib_sources.py sources

# Pack 2,048-token rows (needs transformers and numpy). The order of the sources sets the row order.
python calib_tokens.py --model /path/to/GLM-5.3 --out calib-2m.npy \
    --seq 2048 --tokens 2000000 --heldout 64 --seed 0 \
    sources/cc.jsonl:35 sources/code.jsonl:15 sources/reasoning.jsonl:10 sources/prose.jsonl:15 \
    sources/zh.jsonl:8 sources/structured.jsonl:5 sources/fr.jsonl:1 sources/de.jsonl:1 sources/es.jsonl:1 \
    sources/pt.jsonl:1 sources/it.jsonl:1 sources/ja.jsonl:1 sources/ko.jsonl:1
```

A fresh run of `calib_sources.py` on 2026-10-08 gave byte-identical copies of all 24 files in `sources/` (except
`commitpackft-attribution.jsonl`, which it does not write). calib-2m-v1 was packed with transformers 5.15.1. `cc.jsonl` is not published, so substitute your own transcripts.
The shuffle depends only on the number of rows. If your file fills 360 calibration and 23 held-out rows, every public row
lands where it was in calib-2m-v1, and only the 383 private rows differ.
