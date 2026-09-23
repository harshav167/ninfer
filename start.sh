#!/usr/bin/env bash
# QUASAR: two 250K FP8 windows, 5090 compute, 3070 active overflow and inactive storage.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$ROOT/../models/quasar-ninfer/qwen3_8_27b_quasar.ninfer}"
SERVER="$ROOT/build/apps/ninfer-serve"
REQUEST_LOG="${REQUEST_LOG:-$ROOT/../models/quasar-ninfer/serve-secondary-cache-requests.jsonl}"
CONTEXT_COSTS="${CONTEXT_COSTS:-$ROOT/profiles/secondary-cache-costs.json}"

[[ -x "$SERVER" ]] || { printf 'Missing server binary: %s\n' "$SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'Missing model artifact: %s\n' "$MODEL" >&2; exit 1; }
[[ -f "$CONTEXT_COSTS" ]] || { printf 'Missing transfer-cost profile: %s\n' "$CONTEXT_COSTS" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# All model computation stays on device 0 (5090). Device 1 (3070) provides storage only.
# The 500096-token pool covers two complete 250000-token windows; 320000 tokens
# reside on the 5090. The secondary inactive cache is separate: 64 MiB + one state.
# Active overflow uses double-buffered pinned transfers and requires eager execution.
# There is no server-wide thinking cap; the embedded template defaults to xhigh.
exec "$SERVER" "$MODEL" \
  --device 0 --host 0.0.0.0 --port "${PORT:-30000}" \
  --model-id qwen38-quasar-ninfer \
  --max-context 250000 --kv-capacity 500096 --max-concurrency 2 \
  --pending-timeout-ms 1200000 \
  --kv-dtype fp8 \
  --device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 \
  --active-kv-secondary-device 1 --active-kv-local-tokens 320000 \
  --secondary-cache-device 1 --secondary-cache-state-slots 1 \
  --secondary-cache-kv-mib 64 \
  --max-private-continuations 8 --max-shared-prefixes 8 \
  --context-cost-presets "$CONTEXT_COSTS" \
  --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --vision --preserve-thinking --default-max-tokens 65536 --no-cuda-graph \
  --request-log-jsonl "$REQUEST_LOG" --log-level info \
  "$@"
