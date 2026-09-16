# IF-SFT × VPTQ lab

Weight-only **Vector Post-Training Quantization (VPTQ)** experiment for the public IF-SFT LLaMA-2-7B fingerprinted checkpoint.

## Goal

Compare the same fingerprinted model under:

- `bf16` baseline
- `vptq_4bit`: vector length 6, 4096 main centroids, 4096 residual centroids (~4 bpw index budget)
- `vptq_3bit`: vector length 8, 65536 main centroids, 256 residual centroids (~3 bpw index budget)

The quantized conditions are **weight VQ**, not KV-cache quantization.

Model: `cnut1648/LLaMA2-7B-fingerprinted-SFT`

Upstreams (pinned by commit; see `upstream-lock.json`):

- Microsoft VPTQ `algorithm`: https://github.com/microsoft/VPTQ/tree/algorithm
- IF-SFT / Model-Fingerprint: https://github.com/cnut1648/Model-Fingerprint
- Cornell QTIP: used only for its upstream LLaMA Hessian collector, because VPTQ's Hessian loader explicitly derives from the QuIP#/QTIP file format.

The pipeline calls the original upstream `calc_FSR_from_jsonl()` for official FSR and records exact-key success separately.

## Why these 3-bit / 4-bit settings?

`vptq_3bit` follows the Microsoft VPTQ algorithm tutorial:

```text
v8-k65536-256 = 16/8 + 8/8 = 3 nominal index bits/weight
```

`vptq_4bit` follows the **VPTQ-community reproduced LLaMA-2 baseline** named `Llama-2-7b-hf-v6-k4096-4096`:

```text
v6-k4096-4096 = 12/6 + 12/6 = 4 nominal index bits/weight
```

This LLaMA-2-specific 4-bit configuration is preferred over a synthetic `v8-k65536-65536` setting because it is upstream/community-reproduced for LLaMA-2 and requires far cheaper 4096-cluster k-means. Exact settings are also committed under `configs/`.

These are nominal index BPW; codebook/metadata overhead is separate.

## Hessians

VPTQ requires `{layer}_{qkv|o|up|down}.pt` Hessian files. The VPTQ algorithm release does not include a LLaMA-2 Hessian collection script, so this repo uses the pinned upstream Cornell QTIP LLaMA collector rather than implementing a custom estimator.

If `VPTQ_HESSIAN_DIR` is unset, the pipeline generates Hessians **from the fingerprinted checkpoint itself** into:

```text
artifacts/hessians_llama2_7b_fingerprinted/
```

Colab-practical defaults are:

```text
HESSIAN_DEVSET_SIZE=512
HESSIAN_CTX_SIZE=4096
HESSIAN_BATCH_SIZE=1
```

For the upstream collector's full default sample count:

```bash
export HESSIAN_DEVSET_SIZE=8192
```

This is much slower. For attack screening, 512 is the default; the chosen value is visible in the run command/log.

You can also provide an existing compatible directory:

```bash
export VPTQ_HESSIAN_DIR=/content/hessians-llama2-7b
export VPTQ_INV_HESSIAN_DIR=/content/invhessians-llama2-7b   # optional
```

The launcher validates all 128 expected Hessian files and refuses silent cross-architecture substitution.

## Colab

A100 40GB is recommended. The launcher creates an isolated Python 3.10 environment under `.venv-vptq`; Colab's system Torch is snapshotted before/after and must remain unchanged.

First setup/smoke test:

```bash
%%bash
set -euo pipefail
cd /content
rm -rf if-sft-vptq-lab
git clone https://github.com/HuyL13/if-sft-vptq-lab.git
cd if-sft-vptq-lab
bash run_full.sh --setup-only
```

Recommended staged first run:

```bash
%%bash
set -euo pipefail
cd /content/if-sft-vptq-lab
bash run_full.sh --baseline-only
bash run_full.sh --hessian-only
bash run_full.sh --quant-only 4
bash run_full.sh --quant-only 3
bash run_full.sh --eval-only
```

Or, after setup, run everything:

```bash
%%bash
set -euo pipefail
cd /content/if-sft-vptq-lab
bash run_full.sh --fsr-only
```

The released VPTQ `run_vptq.py` also runs its built-in WikiText-2/C4-new PPL pass after packing (`--new_eval`). Those PPL files are copied into the condition results when produced.

Useful modes:

```bash
bash run_full.sh --setup-only
bash run_full.sh --baseline-only
bash run_full.sh --hessian-only
bash run_full.sh --quant-only 4
bash run_full.sh --quant-only 3
bash run_full.sh --eval-only
FORCE=1 bash run_full.sh --fsr-only
```

For a faster pipeline smoke test, lower only k-means iterations:

```bash
VPTQ_KITER=10 bash run_full.sh --quant-only 4
```

Do not use reduced-iteration smoke results as final experiment numbers.

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
  vptq_4bit/
  vptq_3bit/
env/
```

Each quantized artifact records the exact upstream command, elapsed time, centroid configuration and packed-model path.

## Reproducibility rules

- Weight quantization comes from Microsoft's unmodified VPTQ algorithm code path.
- Hessian collection comes from pinned Cornell QTIP upstream code, not a local reimplementation.
- Official FSR comes from `cnut1648/Model-Fingerprint/report_FSR_sft_chat.py` via the original `calc_FSR_from_jsonl()`.
- The checkpoint being quantized is the fingerprinted checkpoint itself, not base LLaMA-2.
- `exact_key_FSR` is diagnostic only; it is not called official FSR.
- No silent fallback to FP/BF16 is allowed when VPTQ loading fails.
- No silent use of Llama-3/Qwen Hessians is allowed.
- System Colab Torch must be unchanged after setup/run.

## Status

Integration is pushed and source-pinned, but a full A100 7B run has not been executed from this authoring environment. Start with `--setup-only`, then `--baseline-only`, then `--hessian-only` before committing to both full VPTQ sweeps.
