#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"
mkdir -p results/logs

# Append instead of overwrite so a failed Colab run keeps its traceback.
exec > >(tee -a results/logs/hyperquant.log) 2>&1

FULL_EVAL=0
if [[ "${1:-}" == "--full-eval" ]]; then
  FULL_EVAL=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: bash run_hyperquant.sh [--full-eval]" >&2
  exit 2
fi

if [[ "$FULL_EVAL" -eq 1 ]]; then
  EVAL_MODE="full_eval"
  HQ_EVAL_ARGS=(--full-eval)
  BF16_INFER_ARGS=()
  BF16_SCORE_ARGS=()
else
  EVAL_MODE="fsr_only"
  HQ_EVAL_ARGS=()
  BF16_INFER_ARGS=(--fsr-only)
  BF16_SCORE_ARGS=(--fsr-only)
fi

echo
echo "================================================================================"
echo "[HYPERQUANT RUN] $(date -Iseconds)"
echo "[HYPERQUANT RUN] repo=$REPO_ROOT"
echo "[HYPERQUANT RUN] python=$(command -v python3)"
echo "[HYPERQUANT RUN] eval_mode=$EVAL_MODE"
echo "================================================================================"

PYTHON_BIN="${PYTHON:-$(command -v python3)}"

"$PYTHON_BIN" -m scripts.setup_hyperquant_env
"$PYTHON_BIN" -m scripts.hyperquant_integration_check
"$PYTHON_BIN" -m scripts.prepare_data

DATASET="$REPO_ROOT/upstream/Model-Fingerprint/dataset/llama_fingerprint_chat"
MODEL="cnut1648/LLaMA2-7B-fingerprinted-SFT"

bf16_satisfies_mode() {
  [[ -s results/bf16/publish.jsonl && -s results/bf16/fsr.json ]] || return 1
  local n
  n="$(grep -cve '^[[:space:]]*$' results/bf16/publish.jsonl || true)"
  if [[ "$FULL_EVAL" -eq 1 ]]; then
    [[ "$n" -eq 352 ]]
  else
    # A previous full result is a valid superset of the first-8 FSR-only eval.
    [[ "$n" -eq 8 || "$n" -eq 352 ]]
  fi
}

# BF16 control uses the same upstream examples and scorer as the quantized runs.
# Default mode generates only the 8 fingerprint positives required by official FSR.
if ! bf16_satisfies_mode; then
  mkdir -p results/bf16
  "$PYTHON_BIN" -m scripts.infer_if_sft \
    --model "$MODEL" \
    --backend bf16 \
    --dataset "$DATASET" \
    --output results/bf16/publish.jsonl \
    "${BF16_INFER_ARGS[@]}"
  "$PYTHON_BIN" -m scripts.score_fsr \
    --condition bf16 \
    --input results/bf16/publish.jsonl \
    --output-dir results/bf16 \
    "${BF16_SCORE_ARGS[@]}"
else
  echo "[SKIP] BF16 IF-SFT control already satisfies $EVAL_MODE"
fi

# Each target starts from a fresh BF16 checkpoint inside its own Python process;
# there is no accidental 4->3 requantization.
"$PYTHON_BIN" -m scripts.run_hyperquant_if_sft --bps 4 "${HQ_EVAL_ARGS[@]}"
"$PYTHON_BIN" -m scripts.run_hyperquant_if_sft --bps 3 "${HQ_EVAL_ARGS[@]}"

echo
echo "==================== FINAL FSR ===================="
for d in results/bf16 results/hyperquant_4bps results/hyperquant_3bps; do
  echo "--- $d ---"
  cat "$d/fsr.json"
done

echo
if [[ "$FULL_EVAL" -eq 1 ]]; then
  echo "[OK] HyperQuant 4-bps + 3-bps full IF-SFT evaluation complete"
else
  echo "[OK] HyperQuant 4-bps + 3-bps official FSR-only evaluation complete (8 examples per condition)"
  echo "[INFO] Run 'bash run_hyperquant.sh --full-eval' only when robustness metrics are needed."
fi
