#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_training_in_background.sh SYMBOL [SYMBOL...]" >&2
  echo "Optional env: LOOKBACK_DAYS=400 OVERWRITE=true" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

LOOKBACK_DAYS="${LOOKBACK_DAYS:-400}"
OVERWRITE="${OVERWRITE:-false}"
STATUS_FILE="data/training_jobs/model_training_status.json"
HISTORY_FILE="data/training_jobs/model_training_history.json"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="data/training_jobs/manual_train_${TIMESTAMP}.log"

mkdir -p data/training_jobs

CMD=(python3 scripts/train_models.py --symbols "$@" --lookback-days "$LOOKBACK_DAYS" --status-file "$STATUS_FILE" --history-file "$HISTORY_FILE")
if [ "$OVERWRITE" = "true" ]; then
  CMD+=(--overwrite)
fi

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID=$!

echo "Started training job PID=$PID"
echo "Log: $LOG_FILE"
echo "Status file: $STATUS_FILE"
