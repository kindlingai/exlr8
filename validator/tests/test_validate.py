"""Validator tests on small synthetic filesets: one good fileset, then one change per failure mode.

    python -m unittest discover -s validator/tests -t .
"""

import io
import json
import os
import shutil
import struct
import tempfile
import unittest
from contextlib import redirect_stdout

from validator import validate as v
from validator.tests import fileset as fs


class Base(unittest.TestCase):
    build_args: dict = {}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="exlr8-validate-")
        self.root = fs.build(os.path.join(self.tmp, "release"), **self.build_args)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def p(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def run_validator(self, full: bool = False, root: str | None = None, **kw) -> v.Report:
        return v.validate(root or self.root, full=full, **kw)

    def assertStatus(self, report: v.Report, check: str, status: str, contains: str = "") -> None:
        c = report.checks[check]
        text = "\n".join(f"{s}: {m}" for s, m in c.findings)
        self.assertEqual(c.status, status, f"{check}:\n{text}")
        if contains:
            self.assertIn(contains, text)

    def assertClean(self, report: v.Report, allow_warn: tuple[str, ...] = ()) -> None:
        for name, c in report.checks.items():
            if name in allow_warn:
                self.assertNotEqual(c.status, "fail", f"{name}: {c.findings}")
            else:
                self.assertEqual(c.status, "pass", f"{name}: {c.findings}")

    def rewrite_sidecar(self, rel: str) -> None:
        """Refresh one file's sidecar after changing the file."""
        side = os.path.join(self.root, "hashes", v.sidecar_rel(rel))
        with open(side, "w") as f:
            f.write(f"{fs.sha(self.p(rel))} {os.path.getsize(self.p(rel))}\n")

    def patch_manifest(self, **changes) -> None:
        path = self.p("manifests/exlr8-manifest-home.json")
        with open(path) as f:
            m = json.load(f)
        m.update(changes)
        with open(path, "w") as f:
            json.dump(m, f)


class GoodFileset(Base):
    def test_fast(self):
        self.assertClean(self.run_validator())

    def test_full(self):
        report = self.run_validator(full=True)
        self.assertClean(report)
        self.assertIn("26 sha256 verified", report.checks["hash sidecars"].notes[0])

    def test_cli_exit_status(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(v.main([self.root]), 0)
        self.assertIn("summary: 14 checks, 14 pass, 0 warn, 0 fail", out.getvalue())
        os.remove(self.p("dense/exlr8-dense-002.safetensors"))
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(v.main([self.root, "--full"]), 1)
        self.assertIn("FAIL  layout", out.getvalue())

    def test_snapshot_symlinks(self):
        """An HF snapshot directory is a tree of symlinks into a blob store."""
        snap = os.path.join(self.tmp, "snapshot")
        blobs = os.path.join(self.tmp, "blobs")
        os.makedirs(blobs)
        for rel in v.walk(self.root):
            blob = os.path.join(blobs, fs.sha(self.p(rel)))
            if not os.path.exists(blob):
                shutil.copyfile(self.p(rel), blob)
            os.makedirs(os.path.dirname(os.path.join(snap, rel)), exist_ok=True)
            os.symlink(blob, os.path.join(snap, rel))
        os.makedirs(os.path.join(snap, ".cache", "huggingface"))
        self.assertClean(self.run_validator(root=snap, full=True))


class Layout(Base):
    def test_missing_dense_layer(self):
        os.remove(self.p("dense/exlr8-dense-002.safetensors"))
        self.assertStatus(self.run_validator(), "layout", "fail", "exlr8-dense-002.safetensors is missing")

    def test_missing_index(self):
        os.remove(self.p("experts/exlr8-layer-003.index.safetensors"))
        self.assertStatus(self.run_validator(), "layout", "fail", "exlr8-layer-003.index.safetensors is missing")

    def test_unexpected_member(self):
        with open(self.p("experts/exlr8-layer-001.k2.safetensors.tmp"), "wb") as f:
            f.write(b"x")
        self.assertStatus(self.run_validator(), "layout", "fail", "not an expert index or trellis file")

    def test_layer_outside_config(self):
        shutil.copyfile(self.p("dense/exlr8-dense-004.safetensors"), self.p("dense/exlr8-dense-005.safetensors"))
        self.assertStatus(self.run_validator(), "layout", "fail", "layer 5 is outside 0-4")

    def test_missing_support_file(self):
        os.remove(self.p("tokenizer.json"))
        self.assertStatus(self.run_validator(), "support files", "fail", "tokenizer.json is missing")


class Safetensors(Base):
    def test_truncated(self):
        rel = "experts/exlr8-layer-002.k4.safetensors"
        with open(self.p(rel), "r+b") as f:
            f.truncate(os.path.getsize(self.p(rel)) - 1)
        self.assertStatus(self.run_validator(), "safetensors", "fail", "file has")

    def test_bad_header(self):
        with open(self.p("dense/exlr8-dense-001.safetensors"), "r+b") as f:
            f.seek(8)
            f.write(b"!")
        self.assertStatus(self.run_validator(), "safetensors", "fail", "header is not JSON")

    def test_unaligned(self):
        t = fs.record("self_attn.q_proj", 3, 512, fs.H)
        fs.write_safetensors(self.p("dense/exlr8-dense-001.safetensors"), t,
                             dict(fs.MODEL_META, layer="1", contents="dense"), align=False)
        self.assertStatus(self.run_validator(), "safetensors", "fail", "not a multiple of 65536")


class DenseRecords(Base):
    def write_dense(self, tensors):
        fs.write_safetensors(self.p("dense/exlr8-dense-001.safetensors"), tensors,
                             dict(fs.MODEL_META, layer="1", contents="dense"))

    def test_wrong_data_shape(self):
        t = fs.record("self_attn.q_proj", 3, 512, fs.H)
        t[3] = ("self_attn.q_proj.e8m0", "U8", [512, 4], None)
        self.write_dense(t)
        self.assertStatus(self.run_validator(), "dense records", "fail", "e8m0 is [512, 4], expected [512, 8]")

    def test_missing_data_tensor(self):
        t = [x for x in fs.record("self_attn.o_proj", 1, fs.H, 512, K=4, side=2) if not x[0].endswith(".svh")]
        self.write_dense(t)
        self.assertStatus(self.run_validator(), "dense records", "fail", "data tensor svh is missing")

    def test_bad_record(self):
        t = fs.record("self_attn.q_proj", 3, 512, fs.H)
        t[0] = ("self_attn.q_proj.record", "I64", [8], fs.i64(9, 512, fs.H, 0, 0, 0, 0, 0))
        self.write_dense(t)
        self.assertStatus(self.run_validator(), "dense records", "fail", "record format 9 is not 1-7")

    def test_checksum_length(self):
        t = fs.record("self_attn.o_proj", 1, fs.H, 512, K=4, side=2)
        t[1] = ("self_attn.o_proj.checksum", "U64", [2], None)
        self.write_dense(t)
        self.assertStatus(self.run_validator(), "dense records", "fail", "expected U64 [4]")

    def test_layer_metadata(self):
        t = fs.record("self_attn.q_proj", 3, 512, fs.H)
        fs.write_safetensors(self.p("dense/exlr8-dense-001.safetensors"), t,
                             dict(fs.MODEL_META, layer="2", contents="dense"))
        self.assertStatus(self.run_validator(), "dense records", "fail", "metadata layer is '2'")

    def test_vocab_shape(self):
        t = fs.record("embed_tokens", 5, 100, fs.H) + fs.record("lm_head", 5, fs.VOCAB, fs.H) + \
            [("norm.weight", "BF16", [fs.H], None)]
        fs.write_safetensors(self.p("dense/exlr8-vocab.safetensors"), t, dict(fs.MODEL_META, contents="vocab"))
        self.assertStatus(self.run_validator(), "vocab", "fail", "embed_tokens is [100, 256]")


class Experts(Base):
    def test_config_mismatch(self):
        with open(self.p("config.json")) as f:
            cfg = json.load(f)
        cfg["text_config"]["n_routed_experts"] = 8
        with open(self.p("config.json"), "w") as f:
            json.dump(cfg, f)
        r = self.run_validator()
        self.assertStatus(r, "expert indexes", "fail", "k2.index is I64 [4, 3, 2, 2], expected I64 [8, 3, 2, 2]")
        self.assertStatus(r, "expert trellis files", "fail")

    def test_missing_trellis_file(self):
        os.remove(self.p("experts/exlr8-layer-002.k4.safetensors"))
        self.assertStatus(self.run_validator(), "expert trellis files", "fail",
                          "exlr8-layer-002.k4.safetensors is missing (index stores K4)")

    def test_extra_trellis_width(self):
        shutil.copyfile(self.p("experts/exlr8-layer-004.k3.safetensors"),
                        self.p("experts/exlr8-layer-004.k4.safetensors"))
        self.assertStatus(self.run_validator(), "expert trellis files", "fail", "K4 is not in the index's widths")

    def test_wrong_offset(self):
        rel = "experts/exlr8-layer-002.index.safetensors"
        sf = v.SafeFile(self.p(rel))
        with open(self.p(rel), "r+b") as f:
            f.seek(sf.start + sf.tensors["k3.index"]["begin"] + 16 * 5)
            f.write(struct.pack("<q", 12345))
        self.assertStatus(self.run_validator(), "expert offsets", "fail", "layer 2 K3 routed experts: 1 of 24")

    def test_unexpected_index_tensor(self):
        rel = "experts/exlr8-layer-001.index.safetensors"
        sf = v.SafeFile(self.p(rel))
        t = [(n, sf.tensors[n]["dtype"], sf.tensors[n]["shape"], sf.read(n)) for n in sf.names()]
        t.append(("k5.index", "I64", [1], None))
        fs.write_safetensors(self.p(rel), t, sf.meta)
        self.assertStatus(self.run_validator(), "expert indexes", "fail", "unexpected tensor k5.index")

    def test_partial_error_tables(self):
        rel = "experts/exlr8-layer-001.index.safetensors"
        sf = v.SafeFile(self.p(rel))
        t = [(n, sf.tensors[n]["dtype"], sf.tensors[n]["shape"], sf.read(n)) for n in sf.names()
             if n != "k2.err_down"]
        fs.write_safetensors(self.p(rel), t, sf.meta)
        self.assertStatus(self.run_validator(), "error tables", "fail", "partial error tables, missing k2.err_down")


class Codebooks(Base):
    """K4 in mul1 beside mcg K2 and K3, as FORMAT.md allows."""
    build_args = {"codebooks": {4: "mul1"}}
    rel = "experts/exlr8-layer-001.k4.safetensors"

    def rewrite(self, rename=lambda n: n, **meta) -> None:
        """Rewrite the K4 trellis file in place. The header is padded to 64 KiB, so the index offsets still hold."""
        sf = v.SafeFile(self.p(self.rel))
        t = [(rename(n), sf.tensors[n]["dtype"], sf.tensors[n]["shape"], None) for n in sf.names()]
        fs.write_safetensors(self.p(self.rel), t, {**sf.meta, **meta})

    def test_full(self):
        self.assertClean(self.run_validator(full=True))

    def test_metadata_names_another_codebook(self):
        self.rewrite(codebook="mcg 0xCBAC1FED, E4M3 rounding")
        self.assertStatus(self.run_validator(), "expert trellis files", "fail",
                          "tensor names select mul1 but metadata codebook is 'mcg 0xCBAC1FED, E4M3 rounding'")

    def test_wrong_multiplier(self):
        self.rewrite(codebook="mul1 0xCBAC1FED, E4M3 rounding")
        self.assertStatus(self.run_validator(), "expert trellis files", "fail", "tensor names select mul1")

    def test_mcg_names_with_mul1_metadata(self):
        self.rewrite(lambda n: n.removesuffix(".mul1"))
        self.assertStatus(self.run_validator(), "expert trellis files", "fail",
                          "tensor names select mcg but metadata codebook is 'mul1 0x83DCD12D, E4M3 rounding'")

    def test_mixed_names(self):
        self.rewrite(lambda n: n.removesuffix(".mul1") if n == "trellis.up.mul1" else n)
        self.assertStatus(self.run_validator(), "expert trellis files", "fail",
                          "trellis tensors mix codebooks ['mcg', 'mul1']")

    def test_unknown_codebook(self):
        self.rewrite(lambda n: n.replace(".mul1", ".mul2"), codebook="mul2 0x12345678")
        self.assertStatus(self.run_validator(), "expert trellis files", "fail", "unknown codebook 'mul2'")


class NoErrorTables(Base):
    build_args = {"errors": False}

    def test_warns(self):
        r = self.run_validator()
        self.assertStatus(r, "error tables", "warn", "4 of 4 index files have no per-unit error tables")
        self.assertClean(r, allow_warn=("error tables",))
        self.assertFalse(r.failed())


class Manifests(Base):
    def test_missing_referenced_file(self):
        os.remove(self.p("experts/exlr8-layer-002.k3.safetensors"))
        self.assertStatus(self.run_validator(), "manifests", "fail",
                          "1 referenced weight files are missing: experts/exlr8-layer-002.k3.safetensors")

    def test_unstored_width(self):
        with open(self.p("manifests/exlr8-manifest-home.json")) as f:
            layers = json.load(f)["layers"]
        layers["2"]["shared"]["down"] = 2
        self.patch_manifest(layers=layers)
        self.assertStatus(self.run_validator(), "manifests", "fail", "layer 2: shared widths [2] are not stored")

    def test_missing_layer(self):
        with open(self.p("manifests/exlr8-manifest-home.json")) as f:
            layers = json.load(f)["layers"]
        del layers["3"]
        self.patch_manifest(layers=layers)
        self.assertStatus(self.run_validator(), "manifests", "fail", "layer 3 is missing")

    def test_bad_shape(self):
        with open(self.p("manifests/exlr8-manifest-home.json")) as f:
            layers = json.load(f)["layers"]
        layers["1"]["gate"] = layers["1"]["gate"][:2]
        self.patch_manifest(layers=layers)
        self.assertStatus(self.run_validator(), "manifests", "fail", "layer 1: gate is not [4][2]")


class WeightSetChecksums(Base):
    def test_missing_set_id(self):
        self.patch_manifest(set_id=None)
        self.assertStatus(self.run_validator(), "weight set checksums", "fail", "set_id is None")

    def test_set_id_mismatch(self):
        self.patch_manifest(set_id="0" * 64)
        self.assertStatus(self.run_validator(), "weight set checksums", "fail", "does not match")
        self.assertStatus(self.run_validator(full=True), "weight set checksums", "fail", "does not match")

    def test_changed_weight_file(self):
        """A rewritten file with a refreshed sidecar still breaks the set_id."""
        with open(self.p("dense/exlr8-dense-000.safetensors"), "r+b") as f:
            f.seek(-1, 2)
            f.write(b"\x01")
        self.rewrite_sidecar("dense/exlr8-dense-000.safetensors")
        r = self.run_validator(full=True)
        self.assertStatus(r, "hash sidecars", "pass")
        self.assertStatus(r, "weight set checksums", "fail", "does not match")

    def test_slice_index_sha(self):
        rel = "experts/exlr8-layer-001.index.safetensors"
        with open(self.p(rel), "r+b") as f:
            f.seek(-1, 2)
            f.write(b"\x01")
        self.rewrite_sidecar(rel)
        self.assertStatus(self.run_validator(), "weight set checksums", "fail", "index_sha256 for layer 1")


class NoManifests(Base):
    build_args = {"manifests": False, "slices": False}

    def test_warns(self):
        r = self.run_validator()
        self.assertStatus(r, "weight set checksums", "warn", "declares no set_id")
        self.assertFalse(r.failed())


class HashSidecars(Base):
    def test_missing_sidecar_warns(self):
        os.remove(os.path.join(self.root, "hashes", "dense", "exlr8-dense-003.safetensors.sha256"))
        r = self.run_validator()
        self.assertStatus(r, "hash sidecars", "warn", "dense/exlr8-dense-003.safetensors: no sha256 sidecar")
        # Fast mode needs every sidecar to compute the set_id. Full mode hashes the file.
        self.assertStatus(r, "weight set checksums", "warn", "cannot compute set_id")
        self.assertFalse(r.failed())
        self.assertFalse(self.run_validator(full=True).failed())

    def test_size_mismatch(self):
        side = os.path.join(self.root, "hashes", "exlr8-layer-001.k2.safetensors.sha256")
        with open(side) as f:
            digest = f.read().split()[0]
        with open(side, "w") as f:
            f.write(f"{digest} 123\n")
        self.assertStatus(self.run_validator(), "hash sidecars", "fail", "sidecar says 123 bytes")

    def test_hash_mismatch(self):
        side = os.path.join(self.root, "hashes", "index-slices", "tp2", "k3", "rank1.safetensors.sha256")
        with open(side) as f:
            size = f.read().split()[1]
        with open(side, "w") as f:
            f.write(f"{'0' * 64} {size}\n")
        self.assertFalse(self.run_validator().failed())  # fast mode reads only headers
        self.assertStatus(self.run_validator(full=True), "hash sidecars", "fail", "does not match sidecar")

    def test_malformed(self):
        with open(os.path.join(self.root, "hashes", "dense", "exlr8-vocab.safetensors.sha256"), "w") as f:
            f.write("not a hash\n")
        self.assertStatus(self.run_validator(), "hash sidecars", "fail", "expected '<sha256> <bytes>'")

    def test_orphan(self):
        with open(os.path.join(self.root, "hashes", "exlr8-layer-009.k2.safetensors.sha256"), "w") as f:
            f.write(f"{'0' * 64} 1\n")
        self.assertStatus(self.run_validator(), "hash sidecars", "fail", "is not in the fileset")

    def test_external_directory(self):
        ext = os.path.join(self.tmp, "hf-hashes")
        shutil.move(os.path.join(self.root, "hashes"), ext)
        self.assertStatus(self.run_validator(), "hash sidecars", "warn")
        self.assertClean(self.run_validator(full=True, hashes=ext))


class OptionalMembers(Base):
    def test_vision_without_sidecar(self):
        fs.vision_file(self.root)
        r = self.run_validator(full=True)
        self.assertStatus(r, "optional members", "pass")
        self.assertStatus(r, "hash sidecars", "warn", "dense/vision-bf16.safetensors: no sha256 sidecar")
        # The manifests' set_id predates the new member.
        self.assertStatus(r, "weight set checksums", "warn", "matches only without the optional")
        self.assertFalse(r.failed())

    def test_vision_with_sidecar(self):
        fs.vision_file(self.root)
        self.rewrite_sidecar("dense/vision-bf16.safetensors")
        self.assertStatus(self.run_validator(full=True), "hash sidecars", "pass")

    def test_vision_bad_tensor(self):
        fs.write_safetensors(self.p("dense/vision-bf16.safetensors"), [("lm_head.weight", "BF16", [4, 4], None)],
                             {}, align=False)
        self.assertStatus(self.run_validator(), "optional members", "fail", "outside model.visual.*")


class IndexSlices(Base):
    def test_missing_rank(self):
        os.remove(self.p("index-slices/tp2/k3/rank1.safetensors"))
        os.remove(os.path.join(self.root, "hashes", "index-slices", "tp2", "k3", "rank1.safetensors.sha256"))
        self.assertStatus(self.run_validator(), "index slices", "fail", "ranks [0], expected 0-1")

    def test_wrong_unit_count(self):
        rel = "index-slices/tp2/k2/rank0.safetensors"
        sf = v.SafeFile(self.p(rel))
        t = [(n, sf.tensors[n]["dtype"], sf.tensors[n]["shape"], None) for n in sf.names()]
        t[0] = (t[0][0], t[0][1], [t[0][2][0] + 1] + t[0][2][1:], None)
        fs.write_safetensors(self.p(rel), t, sf.meta)
        self.rewrite_sidecar(rel)
        self.assertStatus(self.run_validator(), "index slices", "fail", "L001.units is I16")


try:
    import numpy as np
    import xxhash  # noqa: F401

    from exlr8writer import EncodedExpert, LayerData, ModelMetadata, ProjectionData, write_layer
except ImportError:
    np = None


@unittest.skipIf(np is None, "exlr8writer needs numpy and xxhash")
class WriterOutput(Base):
    """Expert files written by exlr8writer pass the expert checks."""
    build_args = {"manifests": False, "slices": False}

    def test_writer_layer(self):
        H, I, E = fs.H, fs.I, fs.E

        def proj(name, K):
            shape = (H // 16, I // 16, 16 * K) if name != "down" else (I // 16, H // 16, 16 * K)
            err = np.zeros(I // 128 if name != "down" else 1, dtype=np.float32)
            return ProjectionData(np.zeros(shape, np.int16), np.ones(I, np.float16), np.ones(H // 128, np.float16),
                                  err)

        def expert(e, ks):
            return EncodedExpert(e, {K: {p: proj(p, K) for p in ("gate", "up", "down")} for K in ks}, 100)

        class Provider:
            def __init__(self, codebook):
                self.codebook = codebook

            def get_layer(self, layer_id):
                return LayerData(layer_id, ModelMetadata(E, H, I, codebook=self.codebook), np.ones(H, np.int8),
                                 np.ones(H, np.int8), [expert(e, (2, 3, 4)) for e in range(E)], expert(E, (3, 4)))

        for codebook in ("mcg 0xCBAC1FED, E4M3 rounding", fs.CODEBOOKS["mul1"]):
            with self.subTest(codebook=codebook):
                for name in os.listdir(self.p("experts")):
                    if name.startswith("exlr8-layer-001."):
                        os.remove(self.p(f"experts/{name}"))
                        os.remove(os.path.join(self.root, "hashes", name + ".sha256"))
                write_layer(Provider(codebook), 1, self.p("experts"), hashes=os.path.join(self.root, "hashes"))
                r = self.run_validator(full=True)
                for check in ("layout", "safetensors", "expert indexes", "expert trellis files", "expert offsets",
                              "error tables", "hash sidecars"):
                    self.assertStatus(r, check, "pass")


if __name__ == "__main__":
    unittest.main()
