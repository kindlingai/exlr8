"""Validate protocol v1 JSON Schemas against Draft 2020-12 and representative payloads.

Run with:
    python -m unittest tests/test_protocol_schemas.py
or (with jsonschema installed):
    uv run --with jsonschema python -m unittest tests/test_protocol_schemas.py
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = ROOT / "protocol" / "schemas"

try:
    import jsonschema
    from referencing import Registry, Resource
except ImportError:  # pragma: no cover
    jsonschema = None
    Registry = None
    Resource = None


EXPECTED_SCHEMAS = {
    "common.schema.json",
    "error.response.schema.json",
    "config.response.schema.json",
    "jobs.response.schema.json",
    "grid.response.schema.json",
    "encoders.response.schema.json",
    "claims.post.request.schema.json",
    "claims.post.response.schema.json",
    "claims.get.response.schema.json",
    "claims.delete.response.schema.json",
    "claim.delete.response.schema.json",
    "piece.put.response.schema.json",
    "piece-url.post.request.schema.json",
    "piece-url.post.response.schema.json",
    "piece-done.post.request.schema.json",
    "piece-done.post.response.schema.json",
    "piece-header.schema.json",
}

WEIGHTS_ID = "c2342cfd7041044364a4536816184f99bad5b2a723cfa77a4249fbab5016a2d7"
COMMIT_40 = "c99a2e42623bad026298a6275dc631cd63431c42"


def _make_claim(*, shared=False, s3=True):
    base = f"https://quant.kindlingai.dev/api/claims/qwen38-pmcgx/3/{512 if shared else 5}"
    hessians = {"shared": f"{base}/hess/shared"}
    if not shared:
        hessians["expert"] = f"{base}/hess/expert"
    claim = {
        "job": "qwen38-pmcgx",
        "job_version": 1,
        "model": "qwen3.8-flash-next",
        "weights_id": WEIGHTS_ID,
        "layer": 3,
        "expert": 512 if shared else 5,
        "shared": shared,
        "expires_at": 1795400000000,
        "hessians": hessians,
        "upload": f"{base}/piece",
        "encode": {
            "hessian_id": "qwen38fn-calib-v1",
            "widths": [2, 3, 4],
            "shared_widths": [2, 3, 4],
            "damping": 0.025,
            "codebooks": {"k2": "pmcgx", "k3": "pmcgx", "k4": "pmcgx"},
            "refine": [64, 32, 96],
            "args": [
                "--codebook", "pmcgx",
                "--hess-id", "qwen38fn-calib-v1",
                "--widths", "2,3,4",
                "--shared-widths", "2,3,4",
                "--shrink", "8000",
                "--shrink-min-erank", "200",
                "--shrink-skip-layers", "3-6",
                "--hess-min", "500",
                "--hess-full", "2000",
                "--gss-width", "3",
                "--refine", "64,32,96",
            ],
        },
    }
    if s3:
        claim["upload_url"] = f"{base}/piece-url"
        claim["upload_done"] = f"{base}/piece-done"
    return claim


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


class ProtocolSchemaFilesTest(unittest.TestCase):
    def test_all_expected_schemas_exist_and_parse_as_json(self):
        found = {p.name for p in SCHEMAS_DIR.glob("*.json")}
        self.assertEqual(found, EXPECTED_SCHEMAS)
        for name in sorted(found):
            with self.subTest(schema=name):
                data = json.loads((SCHEMAS_DIR / name).read_text(encoding="utf-8"))
                self.assertEqual(data["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertTrue(data["$id"].endswith("/" + name))


@unittest.skipIf(jsonschema is None, "jsonschema package not installed")
class ProtocolSchemaValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = {}
        resources = []
        for path in sorted(SCHEMAS_DIR.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            cls.schemas[path.name] = doc
            resources.append((doc["$id"], Resource.from_contents(doc)))
        cls.registry = Registry().with_resources(resources)

    def validator(self, name):
        return jsonschema.Draft202012Validator(self.schemas[name], registry=self.registry)

    def assertValid(self, schema_name, instance):
        self.validator(schema_name).validate(instance)

    def assertInvalid(self, schema_name, instance):
        with self.assertRaises(jsonschema.ValidationError):
            self.validator(schema_name).validate(instance)

    def test_all_schemas_conform_to_draft_2020_12(self):
        for name, doc in self.schemas.items():
            with self.subTest(schema=name):
                jsonschema.Draft202012Validator.check_schema(doc)

    def test_error_response(self):
        self.assertValid("error.response.schema.json", {"error": "no open slots in the ready layers of job ds41-v2"})
        self.assertInvalid("error.response.schema.json", {})
        self.assertInvalid("error.response.schema.json", {"error": ""})
        self.assertInvalid("error.response.schema.json", {"error": "x", "extra": 1})

    def test_config_response(self):
        cfg = {
            "default_job": "glm53-mul1",
            "models": {
                "glm-5.3": {
                    "name": "GLM-5.3",
                    "layers": 79,
                    "hidden_size": 6144,
                    "moe_intermediate_size": 2048,
                    "moe_layers": "3-78",
                    "routed_experts": 256,
                    "shared_expert": True,
                    "weights": {
                        "hf": "zai-org/GLM-5.3-BF16",
                        "revision": "9d2398f478cab2de883137db3a36ad2c96205e24",
                        "id": WEIGHTS_ID,
                    },
                },
            },
            "jobs": {
                "glm53-mul1": {
                    "version": 1,
                    "active": True,
                    "model": "glm-5.3",
                    "title": "GLM-5.3, mul1 at K2, K3 and K4",
                    "hessian_id": "calib-2m-v1",
                    "widths": [2, 3, 4],
                    "shared_widths": [2, 3, 4],
                    "damping": 0.025,
                    "codebooks": [
                        {"layers": "3-77", "codebook": {"k2": "pmcgx", "k3": "mul1", "k4": "mul1"}, "refine": [64, 32, 96]},
                        {"layers": "78", "codebook": "mul1", "widths": [3], "shared_widths": [3], "damping": 0.01},
                    ],
                },
            },
            "codebooks": {
                "mcg": "mcg 0xCBAC1FED",
                "mul1": "mul1 0x83DCD12D",
                "pmcgx": "pmcgx V2 L20 0xCBAC1FED/0xBCA14C51",
            },
        }
        self.assertValid("config.response.schema.json", cfg)
        # Stores must never appear in public /api/config
        self.assertInvalid("config.response.schema.json", {**cfg, "stores": {}})

    def test_jobs_response(self):
        jobs = [
            {
                "job": "qwen38-pmcgx",
                "title": "Qwen3.8-Flash-Next, pmcgx at K2, K3 and K4",
                "model": "Qwen3.8-Flash-Next",
                "version": 1,
                "layers": 48,
                "experts": 513,
                "slots": 48 * 513,
                "uploaded": 100,
                "claimed": 16,
                "disabled": 0,
                "open": 397,
                "closed": 513,
                "not_ready": 46 * 513,
                "harvested": 50,
                "ready_layers": 1,
                "done_layers": 0,
                "contributors": 4,
                "last_upload": 1795400000000,
                "weights": {"hf": "Qwen/Qwen3.8-Flash-Next", "revision": "812d96", "id": WEIGHTS_ID},
                "codebooks": ["pmcgx"],
            },
            {
                "job": "small",
                "title": "small",
                "model": "Small",
                "version": 2,
                "layers": 4,
                "experts": 32,
                "slots": 128,
                "uploaded": 0,
                "claimed": 0,
                "disabled": 0,
                "open": 0,
                "closed": 0,
                "not_ready": 128,
                "harvested": 0,
                "ready_layers": 0,
                "done_layers": 0,
                "contributors": 0,
                "last_upload": None,
                "weights": None,
                "codebooks": ["mcg", "mul1"],
            },
        ]
        self.assertValid("jobs.response.schema.json", jobs)
        self.assertInvalid("jobs.response.schema.json", [{**jobs[0], "codebooks": ["fp4"]}])

    def test_grid_response(self):
        grid = {
            "job": "glm",
            "jobs": ["ds41-v2", "glm"],
            "model": "glm-5.3",
            "title": "GLM-5.3",
            "layers": 6,
            "experts": 33,
            "shared": 32,
            "ready": [3, 5],
            "hessians": [3, 4, 5],
            "moe": [3, 5],
            "plans": [None, None, None, {"k2": "mcg", "k3": "mcg", "k4": "mcg"}, {"k2": "mcg"}, {"k3": "mcg"}],
            "slots": [
                [3, 5, "uploaded", "bob", 0],
                [5, 4, "claimed", None, 0],
                [4, 0, "disabled", None, 0],
            ],
        }
        self.assertValid("grid.response.schema.json", grid)
        self.assertInvalid("grid.response.schema.json", {**grid, "slots": [[3, 5, "open", None, 0]]})

    def test_encoders_response(self):
        self.assertValid(
            "encoders.response.schema.json",
            [
                {"commit_id": COMMIT_40, "added_at": 1761613200000, "note": "initial"},
                {"commit_id": "a" * 40, "added_at": 1761613200001, "note": None},
            ],
        )
        self.assertInvalid(
            "encoders.response.schema.json",
            [{"commit_id": "short12", "added_at": 1, "note": None}],
        )

    def test_claims_post_request(self):
        # Full modern batch claim
        self.assertValid(
            "claims.post.request.schema.json",
            {
                "encoder": COMMIT_40,
                "encoding": 2,
                "codebooks": ["mcg", "mul1", "pmcgx"],
                "job_version": 1,
                "request": "0123456789abcdef",
                "weights": [WEIGHTS_ID],
                "cached": {WEIGHTS_ID: [3, 4]},
                "count": 8,
            },
        )
        # Minimal legacy claim with -unchecked
        self.assertValid("claims.post.request.schema.json", {"encoder": COMMIT_40 + "-unchecked"})
        # Named slot with count=1 is allowed
        self.assertValid(
            "claims.post.request.schema.json",
            {"encoder": COMMIT_40, "job": "ds41-v2", "layer": 3, "expert": 5, "count": 1},
        )
        # Invalid cases
        self.assertInvalid("claims.post.request.schema.json", {})  # missing encoder
        self.assertInvalid("claims.post.request.schema.json", {"encoder": COMMIT_40 + "-dirty"})
        self.assertInvalid("claims.post.request.schema.json", {"encoder": COMMIT_40, "encoding": 0})
        self.assertInvalid("claims.post.request.schema.json", {"encoder": COMMIT_40, "expert": 5})  # no layer
        self.assertInvalid("claims.post.request.schema.json", {"encoder": COMMIT_40, "layer": 3, "expert": 5, "count": 2})
        self.assertInvalid("claims.post.request.schema.json", {"encoder": COMMIT_40, "count": 17})
        self.assertInvalid("claims.post.request.schema.json", {"encoder": COMMIT_40, "request": "short"})
        self.assertInvalid("claims.post.request.schema.json", {"encoder": COMMIT_40, "weights": ["not-hex"]})
        self.assertInvalid("claims.post.request.schema.json", {"encoder": COMMIT_40, "codebooks": ["fp4"]})

    def test_claims_post_and_get_responses(self):
        routed_s3 = _make_claim(shared=False, s3=True)
        shared_r2 = _make_claim(shared=True, s3=False)
        self.assertValid("claims.post.response.schema.json", routed_s3)
        self.assertValid("claims.post.response.schema.json", shared_r2)
        self.assertValid("claims.post.response.schema.json", {"claims": [routed_s3, shared_r2]})
        self.assertValid("claims.get.response.schema.json", [routed_s3, shared_r2])
        self.assertValid("claims.get.response.schema.json", [])

        # Shared claim must NOT carry hessians.expert; routed claim MUST carry it
        bad_shared = _make_claim(shared=True, s3=False)
        bad_shared["hessians"]["expert"] = "https://quant.kindlingai.dev/api/claims/qwen38-pmcgx/3/512/hess/expert"
        self.assertInvalid("claims.post.response.schema.json", bad_shared)

        bad_routed = _make_claim(shared=False, s3=False)
        del bad_routed["hessians"]["expert"]
        self.assertInvalid("claims.post.response.schema.json", bad_routed)

        # upload_url and upload_done must appear together
        bad_s3 = _make_claim(shared=False, s3=True)
        del bad_s3["upload_done"]
        self.assertInvalid("claims.post.response.schema.json", bad_s3)

        # Batch claims array must be non-empty
        self.assertInvalid("claims.post.response.schema.json", {"claims": []})

    def test_delete_and_upload_schemas(self):
        self.assertValid("claims.delete.response.schema.json", {"released": 0})
        self.assertValid("claims.delete.response.schema.json", {"job": "ds41-v2", "released": 4})
        self.assertValid(
            "claim.delete.response.schema.json",
            {"job": "ds41-v2", "layer": 3, "expert": 5, "released": True},
        )
        self.assertInvalid(
            "claim.delete.response.schema.json",
            {"job": "ds41-v2", "layer": 3, "expert": 5, "released": False},
        )

        self.assertValid(
            "piece.put.response.schema.json",
            {"job": "ds41-v2", "layer": 3, "expert": 5, "size": 39123456, "md5": "d41d8cd98f00b204e9800998ecf8427e"},
        )
        self.assertValid("piece-url.post.request.schema.json", {"size": 39123456})
        self.assertInvalid("piece-url.post.request.schema.json", {"size": 8})
        self.assertValid(
            "piece-url.post.response.schema.json",
            {
                "url": "https://s3.hf.co/mmastrac/exlr8-quant-staging/pieces/ds41-v2/layer-003/e005.safetensors?X-Amz-Signature=abc",
                "method": "PUT",
                "headers": {"Content-Length": "39123456"},
                "expires_in": 43200,
            },
        )
        self.assertValid("piece-done.post.request.schema.json", {})
        self.assertValid("piece-done.post.request.schema.json", {"sha256": WEIGHTS_ID})
        self.assertInvalid("piece-done.post.request.schema.json", {"sha256": "bad"})
        self.assertValid(
            "piece-done.post.response.schema.json",
            {"job": "ds41-v2", "layer": 3, "expert": 5, "size": 39123456, "md5": "etag123", "sha256": WEIGHTS_ID},
        )

    def test_piece_header_schema(self):
        self.assertValid("piece-header.schema.json", _make_piece_header(shared=False, mixed=False))
        self.assertValid("piece-header.schema.json", _make_piece_header(shared=True, mixed=False))
        self.assertValid("piece-header.schema.json", _make_piece_header(shared=False, mixed=True))

        # Shared piece must set shared="true"
        bad_shared = _make_piece_header(shared=True)
        bad_shared["__metadata__"]["shared"] = "false"
        self.assertInvalid("piece-header.schema.json", bad_shared)

        # Wrong format string
        bad_fmt = _make_piece_header()
        bad_fmt["__metadata__"]["format"] = "exlr8-v2"
        self.assertInvalid("piece-header.schema.json", bad_fmt)

        # Wrong trellis last dimension for K3 (must be 48, not 32)
        bad_trellis = _make_piece_header()
        bad_trellis["k3.gate.trellis"]["shape"][3] = 32
        self.assertInvalid("piece-header.schema.json", bad_trellis)

        # -dirty commit rejected
        bad_commit = _make_piece_header()
        bad_commit["__metadata__"]["encoder_commit"] = "1234abc-dirty"
        self.assertInvalid("piece-header.schema.json", bad_commit)


if __name__ == "__main__":
    unittest.main()
