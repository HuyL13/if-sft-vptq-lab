# IF-SFT × VPTQ lab

Weight-only **Vector Post-Training Quantization (VPTQ)** experiment for the public IF-SFT LLaMA-2-7B fingerprinted checkpoint.

## Goal

Compare the same fingerprinted model under:

- `bf16` baseline
- `vptq_4bit`: vector length 8, 65536 main centroids, 65536 residual centroids (~4 bpw index budget)
- `vptq_3bit`: vector length 8, 65536 main centroids, 256 residual centroids (~3 bpw index budget)

The quantized conditions are **weight VQ**, not KV-cache quantization.

Model: `cnut1648/LLaMA2-7B-fingerprinted-SFT`

Upstreams:

- VPTQ algorithm branch: https://github.com/microsoft/VPTQ/tree/algorithm
- IF-SFT / Model-Fingerprint: https://github.com/cnut1648/Model-Fingerprint

The pipeline calls the original upstream `calc_FSR_from_jsonl()` for official FSR and also records exact-key success separately.

## Important upstream limitation

The released VPTQ algorithm requires Hessian files per LLaMA layer/operator (`{layer}_{qkv|o|up|down}.pt`). The official tutorial publishes examples for newer models but does not ship a LLaMA-2-7B Hessian collector in the algorithm branch. Therefore this repo **does not invent a replacement Hessian estimator**.

Before the full 3/4-bit quantization run, provide compatible LLaMA-2-7B Hessian files via:

```bash
export VPTQ_HESSIAN_DIR=/content/hessians-llama2-7b
export VPTQ_INV_HESSIAN_DIR=/content/invhessians-llama2-7b   # optional; omit if unavailable
```

The launcher validates expected files/shapes indirectly through upstream VPTQ loading and fails rather than silently using Hessians from another architecture.

## Colab

A100 40GB is recommended. The launcher never intentionally replaces Colab's system Torch; VPTQ dependencies are installed in `.venv-vptq`.

```bash
%%bash
set -euo pipefail
cd /content
rm -rf if-sft-vptq-lab
git clone https://github.com/HuyL13/if-sft-vptq-lab.git
cd if-sft-vptq-lab
bash run_full.sh --setup-only
```

Then, after setting Hessian paths:

```bash
%%bash
set -euo pipefail
cd /content/if-sft-vptq-lab
export VPTQ_HESSIAN_DIR=/content/hessians-llama2-7b
# export VPTQ_INV_HESSIAN_DIR=/content/invhessians-llama2-7b
bash run_full.sh --fsr-only
```

Useful modes:

```bash
bash run_full.sh --setup-only
bash run_full.sh --baseline-only
bash run_full.sh --quant-only 4
bash run_full.sh --quant-only 3
bash run_full.sh --eval-only
bash run_full.sh --fsr-only
```

## Outputs

```text
results/
  bf16/
  vptq_4bit/
  vptq_3bit/
  key_logs/
  summary.json
  summary.md
  logs/
artifacts/
  vptq_4bit/
  vptq_3bit/
env/
```

Each quantized artifact records the VPTQ command, elapsed time, model path, centroid configuration and source revision.

## Reproducibility rules

- Quantization comes from Microsoft VPTQ `algorithm` branch.
- Official FSR comes from `cnut1648/Model-Fingerprint/report_FSR_sft_chat.py`.
- The checkpoint being quantized is the fingerprinted checkpoint itself, not base LLaMA-2.
- `exact_key_FSR` is diagnostic only; it is not called official FSR.
- No silent fallback to FP/BF16 is allowed when VPTQ loading fails.
- System Torch snapshot is checked before/after the run.

## Status

The integration code is implemented, but full 7B VPTQ quantization has not been executed by this repository authoring environment. The first Colab run should use `--setup-only`, then a full run with real LLaMA-2-7B-compatible Hessians.
