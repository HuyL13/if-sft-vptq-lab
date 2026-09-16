#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p results/logs artifacts env upstream
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

LOG="results/logs/run_full.log"
{
  echo
  echo "================================================================================"
  echo "[RUN] $(date -Is) bash run_full.sh $*"
  echo "[RUN] repo=$ROOT"
  echo "[RUN] commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "================================================================================"
} | tee -a "$LOG"

# Append instead of overwriting: resumed --hessian-only / --quant-only / --eval-only
# stages all remain in one chronological log and can be watched with:
#   tail -n 100 -f results/logs/run_full.log
python -m scripts.pipeline "$@" 2>&1 | tee -a "$LOG"
