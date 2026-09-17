#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"
mkdir -p results/logs

# Append instead of overwrite so a failed Colab run keeps its traceback.
exec > >(tee -a results/logs/hyperquant.log) 2>&1

echo
echo "================================================================================"
echo "[HYPERQUANT RUN] $(date -Iseconds)"
echo "[HYPERQUANT RUN] repo=$REPO_ROOT"
echo "[HYPERQUANT RUN] python=$(command -v python3)"
echo "================================================================================"

PYTHON_BIN="${PYTHON:-$(command -v python3)}"

"$PYTHON_BIN" -m scripts.setup_hyperquant_env
"$PYTHON_BIN" -m scripts.hyperquant_integration_check
"$PYTHON_BIN" -m scripts.prepare_data

DATASET="$REPO_ROOT/upstream/Model-Fingerprint/dataset/llama_fingerprint_chat"
MODEL="cnut1648/LLaMA2-7B-fingerprinted-SFT"

# Control: preserve the already-computed BF16 result when present. On a fresh
# clone, generate it once with the exact same IF-SFT inference/scorer path.
if [[ ! -s results/bf16/publish.jsonl || ! -s results/bf16/fsr.json ]]; then
  mkdir -p results/bf16
  "$PYTHON_BIN" -m scripts.infer_if_sft \
    --model "$MODEL" \
    --backend bf16 \
    --dataset "$DATASET" \
    --output results/bf16/publish.jsonl
  "$PYTHON_BIN" -m scripts.score_fsr \
    --condition bf16 \
    --input results/bf16/publish.jsonl \
    --output-dir results/bf16
else
  echo "[SKIP] BF16 IF-SFT control already exists"
fi

# Run 4 bps first because it is the less aggressive operating point, then 3 bps.
# Each target starts from a fresh BF16 checkpoint inside its own Python process;
# there is no accidental 4->3 requantization.
"$PYTHON_BIN" -m scripts.run_hyperquant_if_sft --bps 4
"$PYTHON_BIN" -m scripts.run_hyperquant_if_sft --bps 3

echo
echo "==================== FINAL FSR ===================="
for d in results/bf16 results/hyperquant_4bps results/hyperquant_3bps; do
  echo "--- $d ---"
  cat "$d/fsr.json"
done

echo
echo "[OK] HyperQuant 4-bps + 3-bps IF-SFT experiment complete"
