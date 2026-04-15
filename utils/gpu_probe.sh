#!/usr/bin/env bash
set -euo pipefail

GPU_ID=""
DATA_CONFIG=""
MODEL_CONFIG=""
FRAMEWORK_CONFIG=""
WRAPPER_CONFIG=""
LOG_PREFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --data-config)
      DATA_CONFIG="$2"
      shift 2
      ;;
    --model-config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --framework-config)
      FRAMEWORK_CONFIG="$2"
      shift 2
      ;;
    --wrapper-config)
      WRAPPER_CONFIG="$2"
      shift 2
      ;;
    --log-prefix)
      LOG_PREFIX="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$GPU_ID" || -z "$DATA_CONFIG" || -z "$MODEL_CONFIG" || -z "$FRAMEWORK_CONFIG" || -z "$WRAPPER_CONFIG" || -z "$LOG_PREFIX" ]]; then
  echo "Missing required arguments." >&2
  exit 2
fi

WORKSPACE="$(pwd)"
LOG_DIR="$WORKSPACE/result/logs"
LOG_PATH="$LOG_DIR/${LOG_PREFIX}.log"
SMI_PATH="$LOG_DIR/${LOG_PREFIX}.smi.csv"

mkdir -p "$LOG_DIR"
rm -f "$LOG_PATH" "$SMI_PATH"

monitor_gpu() {
  while true; do
    printf "%s," "$(date -u +%FT%TZ)" >> "$SMI_PATH"
    nvidia-smi --id="$GPU_ID" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$SMI_PATH"
    sleep 1
  done
}

monitor_gpu &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" uv run --no-sync python driver.py \
  --data-config "$DATA_CONFIG" \
  --model-config "$MODEL_CONFIG" \
  --framework-config "$FRAMEWORK_CONFIG" \
  --wrapper-config "$WRAPPER_CONFIG" \
  > "$LOG_PATH" 2>&1

tail -n 80 "$LOG_PATH"
