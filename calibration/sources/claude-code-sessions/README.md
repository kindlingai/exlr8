# Claude Code session transcripts (not published)

The largest source in calib-2m-v1, `cc.jsonl`, is private Claude Code session transcripts. They are not published.
Nothing derived from them is in this repository: not the text, the token rows or the source's metadata.

This source has weight 35 of 95, a 36.8% share of the mix (about 37%):

| split | rows | tokens | documents |
|---|---|---|---|
| calibration | 360 of 977 | 737,280 | 67 |
| held-out | 23 of 64 | 47,104 | 3 |

Because this part is missing, the published set cannot reproduce calib-2m-v1 exactly. `calib-2m-public.npy` holds the
other 658 rows exactly as they were used. The 383 rows built from this source are not available.

To rebuild a similar set, substitute your own agentic and tool-use transcripts: a JSONL file of
`{"messages": [...], "tools": [...]}` chat records, which `calib_tokens.py` renders with the model's chat template.
Pass it in place of `cc.jsonl:35`.
