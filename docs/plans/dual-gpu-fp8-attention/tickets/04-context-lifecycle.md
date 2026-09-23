# 04: Preserve context reuse, isolation and cancellation

**What to build:** Long-running agents can resume, branch, cancel and finish conversations without losing valid context, corrupting the other agent or leaving remote resources unusable.

**Blocked by:** [03: Preserve concurrent speculative transactions](03-speculative-transactions.md).

**Status:** ready-for-agent; execution gated by the [parent specification](../spec.md), not started.

**Mandatory method:** Invoke/read `/skill:tdd`. Follow existing real-Engine prefix and resource-settlement test patterns through the approved Engine seam. One failing observable lifecycle transition, one complete implementation, then green. Work directly; no subagents or internal-collaborator mocks.

- [ ] Resume a retained deep continuation with valid prefix reuse and correct request identity after another context has used primary and secondary resources.
- [ ] Branch a shared prefix and demonstrate independent suffixes and valid continuation state without duplicate/missing attention contributions.
- [ ] Cancel one actively decoding request while the other continues; settle outstanding transfers and references before memory can be reused.
- [ ] Show successful generation after cancellation and after staggered terminal publication, catching leaked capacity or stale device mappings through behavior.
- [ ] Exercise inactive retention pressure without displacing resources guaranteed to the two active requests or reintroducing the previous unnecessary-spill regression.
- [ ] Preserve copy-on-write, truncate, continuation frontier and speculative state semantics across the chosen placement.
- [ ] Report real resource/transport failures under existing terminal behavior; do not silently fall back to lower precision, shortened context or request serialization.

**Completion evidence:** Engine red/green lifecycle tests and a bounded repeat of resume/branch/cancel/reuse on the real two-device route. Assert results and supported reuse/progress summaries, not private page IDs or transfer call counts.
