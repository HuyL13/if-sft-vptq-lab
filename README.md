# IF-SFT × VPTQ lab

Weight-only **Vector Post-Training Quantization (VPTQ)** experiment for the public IF-SFT LLaMA-2-7B fingerprinted checkpoint.

## Goal

Compare the same fingerprinted model under:

- `bf16` baseline
- `vptq_4bit`: vector length 6, 4096 main centroids, 4096 residual centroids (~4 bpw index budget)
- `vptq_3bit`: vector length 8, 65536 main centroids, 256 residual centroids (~3 bpw index budget)

The quantized conditions are **weight VQ**, not KV-cache quantization.

Model: `cnut1648/LLaMA2-7B-fingerprinted-SFT`

Pinned upstreams are recorded in `upstream-lock.json`:

- Microsoft VPTQ `algorithm` commit: quantization core and packed model format
- IF-SFT / Model-Fingerprint: dataset/protocol and official FSR scorer
- Cornell QuIP#: official offline LLaMA Hessian collector

The pipeline calls the original upstream `calc_FSR_from_jsonl()` for official FSR and records exact-key success separately.

## 3-bit / 4-bit settings

`vptq_3bit` follows the Microsoft VPTQ algorithm tutorial:

```text
v8-k65536-256 = 16/8 + 8/8 = 3 nominal index bits/weight
```

`vptq_4bit` uses the reproduced LLaMA-2 VPTQ configuration:

```text
v6-k4096-4096 = 12/6 + 12/6 = 4 nominal index bits/weight
```

These are nominal index BPW; codebook/metadata overhead is separate.

## Hessians

VPTQ requires 128 files named `{layer}_{qkv|o|up|down}.pt`. Its pinned `vptq/utils/hessian.py` consumes the fields:

```text
flatH
mu
n
```

The repo therefore uses **QuIP# `hessian_offline_llama.py`**, not QTIP's newer collector. This distinction matters: the QTIP collector used earlier does not emit the `mu` field required by VPTQ.

For Python-3.13/Colab compatibility, a runtime copy of the pinned QuIP# collector changes only:

```python
from lib import utils
```

to a tiny source-pinned shim containing the QuIP# Hessian utility functions actually used by the collector. This avoids importing unrelated QuIP# inference/codebook CUDA extensions. The collector body and Hessian equations remain upstream.

If `VPTQ_HESSIAN_DIR` is unset, Hessians are generated from the fingerprinted checkpoint into:

```text
artifacts/hessians_llama2_7b_fingerprinted/
```

Colab-safe defaults:

```text
HESSIAN_DEVSET_SIZE=64
HESSIAN_CTX_SIZE=4096
HESSIAN_BATCH_SIZE=1
HESSIAN_CHUNK_SIZE=64
HESSIAN_SAMPLE_PROC=4
```

QuIP#'s original sample count can be requested with:

```bash
export HESSIAN_DEVSET_SIZE=256
export HESSIAN_CHUNK_SIZE=256
```

After generation, the launcher checks all 128 files and additionally opens representative `qkv/o/up/down` Hessians to verify `flatH + mu + n`, expected LLaMA-2-7B dimensions, and flattened-Hessian sizes before starting VPTQ.

## Colab environment

The repo **does not create a venv and does not install/reinstall/downgrade Torch**. It uses the Python/Torch/CUDA stack already provided by Colab and checks that the Torch version, CUDA version and `torch.__file__` remain unchanged.

The pinned VPTQ loader originally requests old `flash_attention_2`. On current Colab this one loader setting is changed to Transformers SDPA so the experiment does not build legacy `flash-attn`; the VPTQ quantization core is otherwise kept on the pinned upstream implementation.

## Integration preflight

Every non-`--setup-only` run now executes `scripts.integration_check` **before downloading/loading the 7B model**. It checks:

- all local Python files compile
- all upstream commit pins
- CUDA is visible to existing Colab Torch
- FastChat Vicuna prompt template
- Transformers causal-mask helper used by QuIP#
- cuML/CuPy imports
- a real tiny `cuml.cluster.KMeans.fit()` using a Torch CUDA `sample_weight`, matching VPTQ's call shape
- VPTQ Hessian loader requires `flatH/mu/n`
- QuIP# collector writes `flatH/mu/n/ct`
- runtime QuIP# collector imports and `--help` executes
- the source-pinned Hessian hook correctly accumulates both second moment and mean
- VPTQ `run_vptq.py` imports
- the exact 3-bit and 4-bit list-valued CLI arguments parse through VPTQ's own `HfArgumentParser`
- official IF-SFT dataset/scorer files and target key are present

Run only setup + preflight with:

```bash
%%bash
set -euo pipefail
cd /content
rm -rf if-sft-vptq-lab
git clone https://github.com/HuyL13/if-sft-vptq-lab.git
cd if-sft-vptq-lab
bash run_full.sh --preflight-only
```

If this prints:

```text
[INTEGRATION OK] ALL PREFLIGHT INTEGRATION CHECKS PASSED
```

then proceed to the expensive model/Hessian/VPTQ stages.

## Full run

```bash
%%bash
set -euo pipefail
cd /content
rm -rf if-sft-vptq-lab
git clone https://github.com/HuyL13/if-sft-vptq-lab.git
cd if-sft-vptq-lab
git rev-parse HEAD
bash run_full.sh
```

Recommended staged debugging run:

```bash
bash run_full.sh --preflight-only
bash run_full.sh --baseline-only
bash run_full.sh --hessian-only
bash run_full.sh --quant-only 4
bash run_full.sh --quant-only 3
bash run_full.sh --eval-only
```

VPTQ's released `run_vptq.py` also runs its built-in WikiText-2/C4-new PPL pass after packing via `--new_eval`.

## Outputs

```text
results/
  bf16/publish.jsonl
  bf16/fsr.json
  vptq_4bit/publish.jsonl
  vptq_4bit/fsr.json
  vptq_4bit/ppl.json
  vptq_3bit/publish.jsonl
  vptq_3bit/fsr.json
  vptq_3bit/ppl.json
  key_logs/comparison.csv
  key_logs/comparison.txt
  summary.json
  summary.md
  logs/run_full.log
artifacts/
  hessians_llama2_7b_fingerprinted/
  runtime_patches/
  vptq_4bit/
  vptq_3bit/
env/
```

## Reproducibility rules

- Weight-vector quantization uses Microsoft's pinned VPTQ algorithm core.
- Hessian collection uses pinned Cornell QuIP# equations/collector, with only its broad utility import redirected to a minimal source-pinned Hessian shim.
- Official FSR comes from the original `Model-Fingerprint/report_FSR_sft_chat.py` function.
- The quantized checkpoint is `cnut1648/LLaMA2-7B-fingerprinted-SFT`, not base LLaMA-2.
- `exact_key_FSR` is diagnostic only; it is not called official FSR.
- No silent fallback to FP/BF16 is allowed when VPTQ loading fails.
- No cross-architecture Hessian substitution is allowed.
- Colab system Torch must remain unchanged.

## Status

The repo now includes source-contract and executable preflight integration tests. A complete multi-hour A100 7B VPTQ sweep still needs to run on the target Colab GPU; the authoring environment here cannot execute that GPU workload itself.
