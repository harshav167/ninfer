#!/usr/bin/env bash
# Swift 1.5 OrcaRouter (kvnxiao NInfer v3, uncensored): one FP8 KV window, entirely on the RTX 5090.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$ROOT/../models/swift15-orcarouter-ninfer/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer}"
SERVER="$ROOT/build/apps/ninfer-serve"
CONTEXT="${CONTEXT:-262144}"
REQUEST_LOG="${REQUEST_LOG:-$ROOT/../models/swift15-orcarouter-ninfer/serve-fp8-c1-graphs-requests.jsonl}"

[[ -x "$SERVER" ]] || { printf 'Missing server binary: %s\n' "$SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'Missing model artifact: %s\n' "$MODEL" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0

# No secondary GPU cache, active overflow, or host checkpoint storage.
# CUDA graphs are enabled; the KV pool must leave room for graph resources.
# Thinking remains enabled by the chat template.
# NInfer's maintained Qwen3.8 template, not the artifact's embedded one: some
# community artifacts embed the stock Qwen template, which rejects the
# developer role and `instructions` that OMP sends.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ROOT/tools/chat_templates/qwen3_8.jinja}"
exec "$SERVER" "$MODEL" \
  --device 0 --host 0.0.0.0 --port "${PORT:-30000}" \
  --model-id qwen38-ninfer \
  --max-context "$CONTEXT" --kv-capacity "$CONTEXT" --max-concurrency 1 \
  --chat-template "$CHAT_TEMPLATE" \
  --pending-timeout-ms 1200000 \
  --kv-dtype fp8 \
  --device-state-slots 0 --host-state-slots 0 --host-kv-mib 0 \
  --max-private-continuations 8 --max-shared-prefixes 8 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --vision --preserve-thinking --default-max-tokens 65536 \
  --request-log-jsonl "$REQUEST_LOG" --log-level info \
  "$@"
