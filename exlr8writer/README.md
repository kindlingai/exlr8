# exlr8writer

`exlr8writer` writes EXLR8 expert files from quantized tensors supplied by your
own model or quantization workflow. It does not load a particular checkpoint,
choose widths, or quantize weights. Your code implements one small provider
hook; the library handles the shared file layout, expert indexes, checksums,
alignment, and file hashes.

## Install

```sh
pip install .
```

The library requires Python 3.10+, NumPy, and xxhash.

## Connect your workflow

Implement `get_layer(layer_id)` on an object. The hook can fetch tensors from a
local checkpoint, object storage, a database, a quantizer running in memory, or
any other source. Convert the source’s tile and side data to NumPy arrays and
return a `LayerData` value:

```python
import numpy as np

from exlr8writer import (
    EncodedExpert,
    LayerData,
    ModelMetadata,
    ProjectionData,
    write_layer,
)


class MyQuantization:
    def get_layer(self, layer_id: int) -> LayerData:
        # Application-specific loading is intentionally kept in this hook.
        hidden, intermediate, routed_experts = self.model_dimensions(layer_id)
        signs_in, signs_out = self.load_layer_signs(layer_id)
        experts = []
        for expert_id in range(routed_experts):
            widths = {}
            for k in (2, 3, 4):
                widths[k] = {
                    name: ProjectionData(
                        tiles=self.load_trellis_tiles(layer_id, expert_id, name, k),
                        intermediate_scale=self.load_intermediate_scale(layer_id, expert_id, name, k),
                        hidden_block_scale=self.load_hidden_block_scale(layer_id, expert_id, name, k),
                    )
                    for name in ("gate", "up", "down")
                }
            experts.append(EncodedExpert(expert_id=expert_id, widths=widths))

        shared = self.load_shared_expert(layer_id)
        return LayerData(
            layer_id=layer_id,
            metadata=ModelMetadata(
                routed_experts=routed_experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
                hessian="calibration set identifier",
                encoder_commit="quantizer revision",
            ),
            signs_in=signs_in,
            signs_out=signs_out,
            experts=experts,
            shared_expert=shared,
        )


writer = MyQuantization()
result = write_layer(
    writer,
    layer_id=3,
    out="build/experts",
    widths=(3, 4),                 # optional width subset
    shared_widths=(3, 4),          # optional shared-expert width subset
    expert_limit=2,                # optional test prefix, experts 0 and 1
    hashes="build/hashes",         # optional separate hash manifest directory
)
print(result)
```

`ProjectionData.tiles` uses int16 trellis words in codec tile order:

- `gate` and `up`: `[hidden_size / 16, intermediate_size / 16, 16 * K]`
- `down`: `[intermediate_size / 16, hidden_size / 16, 16 * K]`

`intermediate_scale` is an fp16 vector of length `intermediate_size`.
`hidden_block_scale` is an fp16 vector of length `hidden_size / 128`. These are
the quantizer’s encoded values and scales; `exlr8writer` preserves their data
while laying out the output files.

If held-out errors are available, set `ProjectionData.error` to one fp32 value
per 128 intermediate channels for `gate` and `up`, or a scalar for `down`, and
set `EncodedExpert.heldout_tokens`. The writer includes the error tensors when
all supplied experts have complete tables; otherwise it records their absence.

`write_layer` writes one index file and one trellis file for each selected
width. It writes a SHA256 sidecar for every output under `out/hashes/`, or to
the optional `hashes` directory, and
returns a summary with paths, sizes, hashes, selected widths, and whether the
expert prefix is partial. See the repository’s [format specification](../FORMAT.md)
for the tensor definitions and sharding rules. This package writes expert
files and index slices; dense weights, vocabulary files, manifests, tokenizer
files, and the Hugging Face index are assembled by the surrounding model
workflow.

Models other than GLM-5.3 put the model metadata keys of the format
specification (`model`, `num_hidden_layers`, `first_moe_layer`,
`tensor_prefix`) in `ModelMetadata.extra`, so that every index file carries
them.

For large layers, providers can back NumPy arrays with memory mapped storage
instead of keeping every tensor resident in RAM.

## Index slices

Once a release's expert files are written, `write_index_slices` cuts the index
slices for each TP size and width from the layer index files:

```python
from exlr8writer import release_set_id, write_index_slices

sid = release_set_id("build")              # reads every weight file under build/experts and build/dense
write_index_slices(
    "build/experts",
    "build/index-slices",
    tp_sizes=(1, 2, 3, 4),
    widths=(2, 3, 4),
    set_id=sid,
    source="org/model@revision",
    hashes="build/hashes/index-slices",
)
```

It writes `tp<N>/k<K>/rank<r>.safetensors` and a SHA256 sidecar for each.
`release_set_id` reads every weight file once; to skip that, pass a
`{"set_id": ..., "index_sha256": {file name: sha256}}` mapping from another
source, such as the hashes a Hugging Face file listing reports.

## Native stores

`write_native_layer` writes a checkpoint's own quantized expert bytes, unchanged, as a native store (FORMAT.md,
"Native stores"): `mxfp4` for FP4 E2M1 with an E8M0 scale per 32 inputs, `fp8b32` for FP8 E4M3 with an E8M0 scale
per 32 × 32 block. The sources return each projection's HF weight and scale as uint8:

```python
from exlr8writer import write_native_layer

write_native_layer(
    3, "build/experts",
    hidden_size=5120, intermediate_size=2304, routed_experts=384,
    routed=("mxfp4", lambda e, p: load_weight_and_scale(3, e, p)),
    shared=("fp8b32", lambda p: load_shared_weight_and_scale(3, p)),
    metadata={"model": "DeepseekV41ForCausalLM", "tensor_prefix": "layers.", ...},
)
```

`write_index_slices(..., widths=("mxfp4", "fp8b32"))` then writes `tp<N>/<store>/rank<r>.safetensors`.
