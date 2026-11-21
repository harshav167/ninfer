#!/usr/bin/env bash
# Swift 1.5 candidate: shared FP8 KV/State arena on the RTX 5090.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-/mnt/home/wade/models/swift15-uncensored-2beng2-ninfer/Swift_1_5_qwen3_8_27b_uncensored_nvfp4.ninfer}"
SERVER="$ROOT/build/rtx5090-release/apps/ninfer-serve"
# A single request can use this context; two lanes share the physical arena.
CONTEXT="${CONTEXT:-232640}"
# Swift's native layout: 44 blocks of 195897344 bytes. This accommodates
# 3637 FP8 KV pages and four StateImages for 232640-token checkpoint handover.
# This is a byte budget, NOT a token count; the planner rejects insufficient
# budgets rather than reducing max-context or disabling requested features.
CONTEXT_CAPACITY="${CONTEXT_CAPACITY:-8619483136}"
# Two lanes let OMP's side requests (titles, summaries) run beside the main
# turn instead of queueing behind it.
MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"
REQUEST_LOG="${REQUEST_LOG:-$ROOT/.agent-tmp/dev-serve/requests.jsonl}"

[[ -x "$SERVER" ]] || { printf 'Missing server binary: %s\n' "$SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'Missing model artifact: %s\n' "$MODEL" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$REQUEST_LOG")"
export CUDA_VISIBLE_DEVICES=0
# Blocking CUDA waits instead of spinning a CPU core on every device sync.
export NINFER_CUDA_SYNC="${NINFER_CUDA_SYNC:-blocking}"

# Host checkpoints: OMP's side requests (titles, summaries, compaction) could
# otherwise evict the agent's cached state and force a full ~200K re-prefill
# on the next turn. Pinned host context keeps it resumable.
# Host State/KV and in-flight storage retain the established 17920 MiB budget.
# CUDA graphs are enabled; the KV pool must leave room for graph resources.
# Thinking remains enabled by the chat template.
# NInfer's maintained Qwen3.8 template, not the artifact's embedded one: some
# community artifacts embed the stock Qwen template, which rejects the
# developer role and `instructions` that OMP sends.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ROOT/tools/chat_templates/qwen3_8.jinja}"
exec "$SERVER" "$MODEL" \
  --device 0 --host 0.0.0.0 --port "${PORT:-30000}" \
  --model-id qwen38-ninfer \
  --max-context "$CONTEXT" --context-capacity "$CONTEXT_CAPACITY" --max-concurrency "$MAX_CONCURRENCY" \
  --memory-headroom 0B \
  --prefill-chunk "${PREFILL_CHUNK:-1024}" \
  --chat-template "$CHAT_TEMPLATE" \
  --pending-timeout 1200000ms \
  --kv-dtype fp8 \
  --host-cache-capacity "${HOST_CACHE_CAPACITY:-17920MiB}" \
  --spec dflash2 --draft-tokens 7 --proposal-head optimized \
  --vision --thinking --preserve-thinking --max-tokens 65536 \
  --request-log-jsonl "$REQUEST_LOG" --log-level info \
  "$@"
