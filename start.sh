#!/usr/bin/env bash
# Swift1.5 OrcaRouter NVFP4: one shared FP8 KV pool on the RTX 5090.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-/home/wade/local-ai/models/swift15-orcarouter-ninfer/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer}"
SERVER="$ROOT/build/apps/ninfer-serve"
# Both execution lanes share one 232,640-token KV pool; it is not duplicated per lane.
CONTEXT="${CONTEXT:-232640}"
# Two lanes let OMP's side requests (titles, summaries) run beside the main
# turn instead of queueing behind it.
MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"
REQUEST_LOG="${REQUEST_LOG:-/home/wade/local-ai/models/swift15-orcarouter-ninfer/serve-fp8-c1-graphs-requests.jsonl}"

[[ -x "$SERVER" ]] || { printf 'Missing server binary: %s\n' "$SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'Missing model artifact: %s\n' "$MODEL" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0
# Blocking CUDA waits instead of spinning a CPU core on every device sync.
export NINFER_CUDA_SYNC="${NINFER_CUDA_SYNC:-blocking}"

# Host checkpoints: OMP's side requests (titles, summaries, compaction) could
# otherwise evict the agent's cached state and force a full ~200K re-prefill
# on the next turn. Pinned host context keeps it resumable.
# 17920 MiB = the previous 16 GiB host KV + 8 StateImages (~1.46 GiB).
# CUDA graphs are enabled; the KV pool must leave room for graph resources.
# Thinking remains enabled by the chat template.
# NInfer's maintained Qwen3.8 template, not the artifact's embedded one: some
# community artifacts embed the stock Qwen template, which rejects the
# developer role and `instructions` that OMP sends.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ROOT/tools/chat_templates/qwen3_8.jinja}"
exec "$SERVER" "$MODEL" \
  --device 0 --host 0.0.0.0 --port "${PORT:-30000}" \
  --model-id qwen38-ninfer \
  --max-context "$CONTEXT" --kv-capacity "$CONTEXT" --max-concurrency "${MAX_CONCURRENCY:-1}" \
  --prefill-chunk "${PREFILL_CHUNK:-1024}" \
  --chat-template "$CHAT_TEMPLATE" \
  --pending-timeout-ms 1200000 \
  --kv-dtype fp8 \
  --device-state-slots 0 --host-context-mib "${HOST_CONTEXT_MIB:-17920}" \
  --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --vision --preserve-thinking --default-max-tokens 65536 \
  --request-log-jsonl "$REQUEST_LOG" --log-level info \
  "$@"
