"""Validate safetensors header JSON Schemas (PIECE.md, HESSIAN.md) and spec test vectors.

Run with:
    python -m unittest tests/test_schemas.py
or (with jsonschema installed):
    uv run --no-project --with jsonschema python -m unittest tests/test_schemas.py
"""
import hashlib
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = ROOT / "schemas"

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


EXPECTED_SCHEMAS = {
    "piece-header.schema.json",
    "hess-expert-header.schema.json",
    "hess-dense-header.schema.json",
    "inputs-header.schema.json",
}

WEIGHTS_ID = "c2342cfd7041044364a4536816184f99bad5b2a723cfa77a4249fbab5016a2d7"


def _make_piece_header(*, shared=False, mixed=False, widths=(2, 3, 4), h=128, inter=128):
    pre = "shared." if shared else ""
    tensors = {}
    offset = 0

    def add(name, dtype, shape, itemsize):
        nonlocal offset
        n = itemsize
        for d in shape:
            n *= d
        tensors[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n

    add("signs.in", "I8", [h], 1)
    add("signs.out", "I8", [h], 1)
    for k in widths:
        add(f"{pre}k{k}.gate.trellis", "I16", [1, h // 16, inter // 16, 16 * k], 2)
        add(f"{pre}k{k}.up.trellis", "I16", [1, h // 16, inter // 16, 16 * k], 2)
        add(f"{pre}k{k}.down.trellis", "I16", [1, inter // 16, h // 16, 16 * k], 2)
        add(f"{pre}k{k}.gate.svh", "F16", [1, inter], 2)
        add(f"{pre}k{k}.up.svh", "F16", [1, inter], 2)
        add(f"{pre}k{k}.down.suh", "F16", [1, inter], 2)
        add(f"{pre}k{k}.gate.su_block", "F16", [1, h // 128], 2)
        add(f"{pre}k{k}.up.su_block", "F16", [1, h // 128], 2)
        add(f"{pre}k{k}.down.sv_block", "F16", [1, h // 128], 2)
    if not shared:
        add("experts", "I32", [1], 4)
    add(f"{pre}herr", "F32", [1, len(widths), 3], 4)
    add(f"{pre}gscale", "F32", [1, len(widths), 3], 4)
    add(f"{pre}heldout_tokens", "I32", [1], 4)
    add(f"{pre}calib_tokens", "I64", [1], 8)
    add(f"{pre}hess_mode", "I8", [1], 1)
    add(f"{pre}sigma", "F32", [1], 4)
    add(f"{pre}err_source", "I8", [1], 1)
    add(f"{pre}erank_gu", "F32", [1], 4)
    add(f"{pre}erank_down", "F32", [1], 4)
    add(f"{pre}shrink_applied", "U8", [1], 1)

    w_str = ",".join(str(k) for k in widths)
    cb_flag = "k2=pmcgx,k3=mul1,k4=mul1" if mixed else "pmcgx"
    meta = {
        "format": "exlr8-v2-job",
        "scheme": "exlr8-e4m3-v2",
        "layer": "3",
        "experts": "none" if shared else "5",
        "shared": "true" if shared else "false",
        "routed_experts": "512",
        "widths": w_str,
        "shared_widths": w_str,
        "encoder_commit": "1234abc",
        "encoding": "2",
        "encoder": "exlr8enc 0.1.0",
        "codebook": (
            "per width: k2 pmcgx V2 L20 0xCBAC1FED/0xBCA14C51; k3 mul1 0x83DCD12D; k4 mul1 0x83DCD12D, "
            "every value rounded to E4M3 inside the search (round to nearest even, saturating finite)"
            if mixed
            else "pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, every value rounded to E4M3 inside the search (round to nearest even, saturating finite)"
        ),
        "tail_biting": "two passes, then pass 2 again at steps 64,32,96",
        "hessian": "qwen38fn-calib-v1",
        "weights_id": WEIGHTS_ID,
        "encode_args": (
            f"3 {'none --shared' if shared else '5'} --hess-id qwen38fn-calib-v1 --inputs none "
            f"--widths {w_str} --shared-widths {w_str} --shrink 8000 --shrink-min-erank 200 "
            f"--shrink-skip-layers 3-6 --hess-min 500 --hess-full 2000 --gss-width 3 "
            f"--codebook {cb_flag} --refine 64,32,96"
        ),
    }
    if mixed:
        meta["codebook_k2"] = "pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, every value rounded to E4M3 inside the search"
        meta["codebook_k3"] = "mul1 0x83DCD12D, every value rounded to E4M3 inside the search"
        meta["codebook_k4"] = "mul1 0x83DCD12D, every value rounded to E4M3 inside the search"
    return {"__metadata__": meta, **tensors}


class SchemaFilesTest(unittest.TestCase):
    def test_all_expected_schemas_exist_and_parse_as_json(self):
        found = {p.name for p in SCHEMAS_DIR.glob("*.json")}
        self.assertEqual(found, EXPECTED_SCHEMAS)
        for name in sorted(found):
            with self.subTest(schema=name):
                data = json.loads((SCHEMAS_DIR / name).read_text(encoding="utf-8"))
                self.assertEqual(data["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertTrue(data["$id"].endswith("/" + name))


@unittest.skipIf(jsonschema is None, "jsonschema package not installed")
class SchemaValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(SCHEMAS_DIR.glob("*.json"))
        }

    def validator(self, name):
        return jsonschema.Draft202012Validator(self.schemas[name])

    def assertValid(self, schema_name, instance):
        self.validator(schema_name).validate(instance)

    def assertInvalid(self, schema_name, instance):
        with self.assertRaises(jsonschema.ValidationError):
            self.validator(schema_name).validate(instance)

    def test_all_schemas_conform_to_draft_2020_12(self):
        for name, doc in self.schemas.items():
            with self.subTest(schema=name):
                jsonschema.Draft202012Validator.check_schema(doc)

    def test_piece_header_schema(self):
        self.assertValid("piece-header.schema.json", _make_piece_header(shared=False, mixed=False))
        self.assertValid("piece-header.schema.json", _make_piece_header(shared=True, mixed=False))
        self.assertValid("piece-header.schema.json", _make_piece_header(shared=False, mixed=True))

        bad_shared = _make_piece_header(shared=True)
        bad_shared["__metadata__"]["shared"] = "false"
        self.assertInvalid("piece-header.schema.json", bad_shared)

        bad_fmt = _make_piece_header()
        bad_fmt["__metadata__"]["format"] = "exlr8-v2"
        self.assertInvalid("piece-header.schema.json", bad_fmt)

        bad_trellis = _make_piece_header()
        bad_trellis["k3.gate.trellis"]["shape"][3] = 32
        self.assertInvalid("piece-header.schema.json", bad_trellis)

        bad_commit = _make_piece_header()
        bad_commit["__metadata__"]["encoder_commit"] = "1234abc-dirty"
        self.assertInvalid("piece-header.schema.json", bad_commit)

    def test_hess_expert_header_schema(self):
        h, inter, n = 128, 64, 32
        th, ti = h * (h + 1) // 2, inter * (inter + 1) // 2
        off1 = n * th * 4
        off2 = off1 + n * ti * 4
        off3 = off2 + n * 8
        valid = {
            "__metadata__": {
                "hessian_id": "qwen38fn-calib-v1",
                "weights_id": WEIGHTS_ID,
                "layer": "3",
                "experts": "0-31",
                "layout": "upper triangle, row-major, fp32",
            },
            "gu": {"dtype": "F32", "shape": [n, th], "data_offsets": [0, off1]},
            "down": {"dtype": "F32", "shape": [n, ti], "data_offsets": [off1, off2]},
            "count": {"dtype": "I64", "shape": [n], "data_offsets": [off2, off3]},
        }
        self.assertValid("hess-expert-header.schema.json", valid)

        sliced = {
            "__metadata__": {**valid["__metadata__"], "experts": "5-5", "source": "hess-003-e000-031.safetensors"},
            "gu": {"dtype": "F32", "shape": [1, th], "data_offsets": [0, th * 4]},
            "down": {"dtype": "F32", "shape": [1, ti], "data_offsets": [th * 4, (th + ti) * 4]},
            "count": {"dtype": "I64", "shape": [1], "data_offsets": [(th + ti) * 4, (th + ti) * 4 + 8]},
        }
        self.assertValid("hess-expert-header.schema.json", sliced)

        no_hid = {**valid, "__metadata__": {k: v for k, v in valid["__metadata__"].items() if k != "hessian_id"}}
        self.assertInvalid("hess-expert-header.schema.json", no_hid)

        no_wid = {**valid, "__metadata__": {k: v for k, v in valid["__metadata__"].items() if k != "weights_id"}}
        self.assertInvalid("hess-expert-header.schema.json", no_wid)

        bad_wid = {**valid, "__metadata__": {**valid["__metadata__"], "weights_id": "not-a-sha256"}}
        self.assertInvalid("hess-expert-header.schema.json", bad_wid)

        bad_layout = {**valid, "__metadata__": {**valid["__metadata__"], "layout": "full, fp32"}}
        self.assertInvalid("hess-expert-header.schema.json", bad_layout)

        bad_count_dtype = {**valid, "count": {"dtype": "I32", "shape": [n], "data_offsets": [off2, off2 + n * 4]}}
        self.assertInvalid("hess-expert-header.schema.json", bad_count_dtype)

        extra_tensor = {**valid, "extra": {"dtype": "F32", "shape": [1], "data_offsets": [off3, off3 + 4]}}
        self.assertInvalid("hess-expert-header.schema.json", extra_tensor)

    def test_hess_dense_header_schema(self):
        h, inter = 128, 64
        hb, ib = h * h * 4, inter * inter * 4
        valid = {
            "__metadata__": {
                "hessian_id": "qwen38fn-calib-v1",
                "weights_id": WEIGHTS_ID,
                "layer": "3",
                "layout": "full, fp32",
                "sequences": "calibration",
            },
            "mlp_in": {"dtype": "F32", "shape": [h, h], "data_offsets": [0, hb]},
            "count_mlp_in": {"dtype": "I64", "shape": [1], "data_offsets": [hb, hb + 8]},
            "shared_down_in": {"dtype": "F32", "shape": [inter, inter], "data_offsets": [hb + 8, hb + 8 + ib]},
            "count_shared_down_in": {"dtype": "I64", "shape": [1], "data_offsets": [hb + 8 + ib, hb + 16 + ib]},
            "attn_in": {"dtype": "F32", "shape": [h, h], "data_offsets": [hb + 16 + ib, 2 * hb + 16 + ib]},
            "count_attn_in": {"dtype": "I64", "shape": [1], "data_offsets": [2 * hb + 16 + ib, 2 * hb + 24 + ib]},
        }
        self.assertValid("hess-dense-header.schema.json", valid)

        heldout = {**valid, "__metadata__": {**valid["__metadata__"], "sequences": "held out"}}
        self.assertValid("hess-dense-header.schema.json", heldout)

        routed_shared = {
            "__metadata__": {**valid["__metadata__"], "source": "hess-003-dense.safetensors"},
            "mlp_in": valid["mlp_in"],
            "count_mlp_in": valid["count_mlp_in"],
        }
        self.assertValid("hess-dense-header.schema.json", routed_shared)

        no_hid = {**valid, "__metadata__": {k: v for k, v in valid["__metadata__"].items() if k != "hessian_id"}}
        self.assertInvalid("hess-dense-header.schema.json", no_hid)

        no_wid = {**valid, "__metadata__": {k: v for k, v in valid["__metadata__"].items() if k != "weights_id"}}
        self.assertInvalid("hess-dense-header.schema.json", no_wid)

        bad_seq = {**valid, "__metadata__": {**valid["__metadata__"], "sequences": "eval"}}
        self.assertInvalid("hess-dense-header.schema.json", bad_seq)

        bad_count_shape = {**valid, "count_mlp_in": {"dtype": "I64", "shape": [2], "data_offsets": [hb, hb + 16]}}
        self.assertInvalid("hess-dense-header.schema.json", bad_count_shape)

        no_mlp = {k: v for k, v in valid.items() if k != "mlp_in"}
        self.assertInvalid("hess-dense-header.schema.json", no_mlp)

    def test_inputs_header_schema(self):
        n_tok, h, top_k = 1024, 128, 8
        xb = n_tok * h * 2
        ib = n_tok * top_k * 4
        wb = n_tok * top_k * 4
        valid = {
            "__metadata__": {
                "hessian_id": "qwen38fn-calib-v1",
                "weights_id": WEIGHTS_ID,
                "layer": "3",
                "heldout_sequences": "64",
            },
            "x": {"dtype": "BF16", "shape": [n_tok, h], "data_offsets": [0, xb]},
            "topk_idx": {"dtype": "I32", "shape": [n_tok, top_k], "data_offsets": [xb, xb + ib]},
            "topk_w": {"dtype": "F32", "shape": [n_tok, top_k], "data_offsets": [xb + ib, xb + ib + wb]},
        }
        self.assertValid("inputs-header.schema.json", valid)

        no_hid = {**valid, "__metadata__": {k: v for k, v in valid["__metadata__"].items() if k != "hessian_id"}}
        self.assertInvalid("inputs-header.schema.json", no_hid)

        no_wid = {**valid, "__metadata__": {k: v for k, v in valid["__metadata__"].items() if k != "weights_id"}}
        self.assertInvalid("inputs-header.schema.json", no_wid)

        zero_seq = {**valid, "__metadata__": {**valid["__metadata__"], "heldout_sequences": "0"}}
        self.assertInvalid("inputs-header.schema.json", zero_seq)

        wrong_x_dtype = {**valid, "x": {"dtype": "F16", "shape": [n_tok, h], "data_offsets": [0, xb]}}
        self.assertInvalid("inputs-header.schema.json", wrong_x_dtype)

        wrong_idx_dtype = {**valid, "topk_idx": {"dtype": "I64", "shape": [n_tok, top_k], "data_offsets": [xb, xb + 2 * ib]}}
        self.assertInvalid("inputs-header.schema.json", wrong_idx_dtype)

        extra = {**valid, "extra": {"dtype": "F32", "shape": [1], "data_offsets": [xb + ib + wb, xb + ib + wb + 4]}}
        self.assertInvalid("inputs-header.schema.json", extra)

    def test_hessian_spec_test_vectors(self):
        cfg = b'{"hidden_size":128}\n'
        idx = b'{"weight_map":{}}\n'
        self.assertEqual(
            hashlib.sha256(cfg + idx).hexdigest(),
            "24b0d8c953197327441b27e43ecf94a2f3a7adbaf84b22c9ab0123b7c22519a1",
        )

        v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        d = (math.isqrt(8 * len(v) + 1) - 1) // 2
        self.assertEqual(d, 3)
        self.assertEqual(d * (d + 1) // 2, len(v))
        mat = [[0.0] * d for _ in range(d)]
        for r in range(d):
            for c in range(r, d):
                val = v[r * (2 * d + 1 - r) // 2 + (c - r)]
                mat[r][c] = mat[c][r] = val
        self.assertEqual(mat, [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])


if __name__ == "__main__":
    unittest.main()
