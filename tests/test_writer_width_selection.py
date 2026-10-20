import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xxhash

from exlr8writer import EncodedExpert, LayerData, ModelMetadata, ProjectionData, write_layer


def read_header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size)), 8 + size


def read_tensor(path, header, data_start, name, dtype):
    tensor = header[name]
    start, end = tensor["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(data_start + start)
        return np.frombuffer(stream.read(end - start), dtype=dtype).reshape(tensor["shape"])


def make_expert(expert_id):
    return EncodedExpert(expert_id, {
        k: {
            projection: ProjectionData(
                np.full((8, 8, 16 * k), expert_id + 1, dtype=np.int16),
                np.ones(128, dtype=np.float16),
                np.ones(1, dtype=np.float16),
            )
            for projection in ("gate", "up", "down")
        }
        for k in (2, 3)
    })


class WriterWidthSelectionTests(unittest.TestCase):
    def test_only_selected_families_are_written_and_indexed(self):
        routed = [make_expert(0), make_expert(1)]
        shared = make_expert(2)
        cases = (
            ("default", None, None, shared, [2, 3], [2, 3]),
            ("routed_only", (2,), (), shared, [2], []),
            ("shared_only", (), (2,), shared, [], [2]),
            ("disjoint", (3,), (2,), shared, [3], [2]),
            ("overlapping", (2, 3), (3,), shared, [2, 3], [3]),
            ("no_shared_expert", None, None, None, [2, 3], []),
        )
        for name, widths, shared_widths, shared_expert, want_routed, want_shared in cases:
            for pad_halves in (False, True):
                with self.subTest(case=name, pad_halves=pad_halves), tempfile.TemporaryDirectory() as tmp:
                    layer = LayerData(3, ModelMetadata(2, 128, 128),
                                      np.ones(128, dtype=np.int8), np.ones(128, dtype=np.int8),
                                      routed, shared_expert)
                    provider = SimpleNamespace(get_layer=lambda layer_id: layer)
                    result = write_layer(provider, 3, tmp, widths=widths,
                                         shared_widths=shared_widths, pad_halves=pad_halves)
                    self.assertEqual(result["widths"], want_routed)
                    self.assertEqual(result["shared_widths"], want_shared)
                    index_path = Path(tmp) / "exlr8-layer-003.index.safetensors"
                    index, index_start = read_header(index_path)
                    self.assertEqual(index["__metadata__"]["widths"], ",".join(map(str, want_routed)))
                    self.assertEqual(index["__metadata__"]["shared_widths"], ",".join(map(str, want_shared)))
                    for k in sorted(set(want_routed) | set(want_shared)):
                        path = Path(tmp) / f"exlr8-layer-003.k{k}.safetensors"
                        header, data_start = read_header(path)
                        groups = [("", routed)] if k in want_routed else []
                        if k in want_shared:
                            groups.append(("shared.", [shared_expert]))
                        expected = {f"{prefix}trellis.{p}" for prefix, _ in groups
                                    for p in ("gate", "up", "down")}
                        actual = {key for key in header if key != "__metadata__" and not key.startswith("__pad.")}
                        self.assertEqual(actual, expected)
                        for prefix, experts in groups:
                            indices = read_tensor(index_path, index, index_start, f"{prefix}k{k}.index", "<i8")
                            checksums = read_tensor(index_path, index, index_start, f"{prefix}k{k}.checksum", "<u8")
                            for ei, expert in enumerate(experts):
                                for pi, projection in enumerate(("gate", "up", "down")):
                                    offset, length = map(int, indices[ei, pi, 0])
                                    with path.open("rb") as stream:
                                        stream.seek(offset)
                                        payload = stream.read(length)
                                    data = expert.widths[k][projection]
                                    self.assertEqual(payload, data.tiles.tobytes())
                                    digest = xxhash.xxh64(payload, seed=0)
                                    digest.update(data.intermediate_scale.tobytes())
                                    digest.update(data.hidden_block_scale.tobytes())
                                    self.assertEqual(digest.intdigest(), int(checksums[ei, pi, 0]))


if __name__ == "__main__":
    unittest.main()
