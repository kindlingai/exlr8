# exlr8 reference loader

`exlr8.py` decodes the exlr8 v2 checkpoint
[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) into GLM-5.3 tensors
with the Hugging Face names (`model.layers.40.mlp.experts.7.gate_proj.weight` and so on), in BF16 or fp32. The output
can feed a stock BF16 forward pass.

It is a reference for correctness, not for speed. Use it to check a fast loader or kernel against known-good weights,
to inspect or measure the quantized weights, or as a readable statement of the format. It is plain PyTorch on CPU or
GPU, about 340 lines, and follows the byte-level spec in the model card's
[Loader](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4#loader) section. On the same device its output
matches exllamav3 0.0.43's decode value for value.

## Derived from exllamav3

This loader is a derivative of [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp. The trellis
codebook, the bit packing, the tile order and the 128 x 128 Hadamard rotation it decodes are exllamav3's, reproduced
in plain PyTorch. It does not use exllamav3 or its extension. Like exllamav3, it is under the MIT licence, with
exllamav3's copyright notice: see [LICENSE](LICENSE).

## Requirements

- Python 3.10 or later
- PyTorch 2.1 or later (for `torch.float8_e4m3fn`)
- `xxhash`, only for `--verify`

## Usage

Download the manifests and the files of the layers you want, then run the module from this repository's root:

```
hf download kindlingai/glm-5.3-exlr8-k2-k3-k4 --local-dir glm53-exlr8 \
    --include 'manifests/*' 'dense/exlr8-dense-040.safetensors' 'experts/exlr8-layer-040.*'
python -m loader.exlr8 glm53-exlr8 --manifest home --layer 40 --verify --out layer-040.safetensors
```

| option | meaning |
|---|---|
| `--manifest` | `home` (default), `down1`, `up1`, or the path of a manifest file |
| `--layer` | a layer number (0-78), or `vocab` for the embedding, `lm_head` and the final norm |
| `--out` | write the tensors to a safetensors file; without it the loader prints their names and shapes |
| `--dtype` | `bf16` (default) or `fp32`, for matrices; norms and biases keep their stored dtype |
| `--device` | `cpu` (default) or a GPU such as `cuda` |
| `--verify` | check every half's and every dense block's xxhash64 against the stored checksums |

A MoE layer in BF16 is about 19 GB, held in memory, and takes a few minutes to decode on a CPU. Layers 0-2 have no
routed experts.

From Python:

```python
from loader import exlr8

root = "glm53-exlr8"
tensors = exlr8.load_layer(root, exlr8.load_manifest(root, "home"), 40)   # {HF name: tensor}

layer = exlr8.ExpertLayer(root, 40)
w = layer.half(7, "gate", 3, K=3)          # one 128-channel half as (in, out) fp32: (6144, 128)
gate = layer.projection(7, "gate", 3)      # the whole projection at K3, as the HF weight (2048, 6144)
```

The checkpoint's repository also holds the base model's config, tokenizer and chat template (from
[zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), unchanged).
