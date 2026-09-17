# HyperQuant x IF-SFT (LLaMA-2-7B)

This path tests whether **HyperQuant weight vector quantization** removes the public IF-SFT fingerprint from `cnut1648/LLaMA2-7B-fingerprinted-SFT`.

## What is upstream vs local

Pinned upstream sources:

- HyperQuant: `moonmath-ai/HyperQuant` @ `227f3f668ac19368aecb1776c14add28cce8b181`
- Model-Fingerprint / IF-SFT: `cnut1648/Model-Fingerprint` @ `4ae5e8a124c37f25a3711c407e85a45fda6ecb08`

The quantizer itself stays upstream HyperQuant:

- `calibrate_lattice_bps_to_snr(...)`
- `lattice_alpha(...)`
- `integrations.llama.rice_linear.convert_linears(...)`
- upstream E8 lattice/Rice weight packing
- upstream `LatticeLinear` INT8 inference path

Two runtime compatibility patches are applied after restoring the files from the exact pinned commit:

1. upstream hard-codes CUDA compilation to `sm_90a`; this experiment changes only that compile flag to the **actual active GPU compute capability** (for example `sm_80` on A100);
2. upstream `Stage2CudaDecoder::decode_fused_to` lazily allocates a second persistent one-byte output buffer for every quantized Linear on its first INT8/FP8 forward. The decoder already owns an equally-sized byte buffer and the fused decode does not read from it, so the experiment reuses that existing buffer. This leaves E8/Rice encoding, decoding values, target bps, RHT, and GEMM math unchanged while preventing first-generation GPU memory from growing by roughly another byte per transformer weight.

Because the patch targets are reset from the pinned commit before every setup run, retries are idempotent and cannot stack text patches on top of one another.

## 3-bit / 4-bit terminology

HyperQuant exposes a **continuous target bits-per-scalar (bps)** operating point, not a conventional fixed integer code width. Therefore this repo labels the two requested conditions accurately as:

- `hyperquant_4bps`: upstream E8 calibration at target `4.0 bps`
- `hyperquant_3bps`: upstream E8 calibration at target `3.0 bps`

The manifest records both target bps and the actually achieved compressed transformer-weight bpw derived from upstream compression statistics.

## Environment guarantees

The HyperQuant path is intentionally separate from the older VPTQ setup.

- no venv
- uses the existing Colab Python
- **never installs/reinstalls/downgrades Torch**
- editable HyperQuant install uses `--no-deps`
- checks Torch version, CUDA version and Torch file path before/after setup
- requires CUDA GPU with compute capability >= 8.0 and `nvcc`
- compiles/loads the real upstream CUDA extension in preflight
- source-checks that the first-forward `d_byte_out_` allocation is gone and the existing decoder byte buffer is used instead
- runs a real tiny CUDA bf16 -> HyperQuant -> forward smoke test twice at **both 4 bps and 3 bps before touching the 7B model**

## IF-SFT fidelity

The experiment reuses the existing IF-SFT path in this repo:

- official upstream `create_fingerprint_chat.py`
- same Vicuna FastChat conversation template
- same fingerprint prompt suffix
- same greedy generation settings
- official upstream `report_FSR_sft_chat.py::calc_FSR_from_jsonl`
- additionally logs exact-key FSR and the first 8 generated fingerprint outputs

Each HyperQuant condition loads a **fresh BF16 IF-SFT checkpoint** in a new process. The 3-bps run is never quantized from the 4-bps model.

## One-command Colab run

```bash
%%bash
set -euo pipefail
cd /content
rm -rf if-sft-vptq-lab
git clone https://github.com/HuyL13/if-sft-vptq-lab.git
cd if-sft-vptq-lab
bash run_hyperquant.sh
```

The first run compiles HyperQuant's CUDA extension; later runs reuse Torch's extension cache when available. The runner also prints GPU free/allocated/reserved memory before model load, after BF16 load, after quantization, and after IF-SFT inference.

## Outputs

```text
results/bf16/fsr.json
results/hyperquant_4bps/publish.jsonl
results/hyperquant_4bps/fsr.json
results/hyperquant_4bps/manifest.json
results/hyperquant_3bps/publish.jsonl
results/hyperquant_3bps/fsr.json
results/hyperquant_3bps/manifest.json
results/key_logs/hyperquant_4bps.json
results/key_logs/hyperquant_3bps.json
results/logs/hyperquant.log
```

`manifest.json` records target bps, SNR, alpha, converted-layer count, compression ratio, achieved transformer-weight bpw, quantization time, inference time, GPU-memory snapshots, runtime patch version and both upstream commit pins.

## Resume behavior

A condition is skipped only when all three files exist and match the current pinned/runtime configuration:

- `publish.jsonl`
- `fsr.json`
- `manifest.json`

If a run crashes midway, it will not be mistaken for a completed quantization. Because HyperQuant transforms the model in memory, a failed per-bps run restarts that bps condition from the original BF16 checkpoint rather than trying to deserialize a partially transformed model.
