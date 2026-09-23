# 02: Run concurrent FP8 generation through the Engine

**What to build:** Two independent requests can generate simultaneously through the public Engine when their combined active KV exceeds primary residency, using the numerically qualified near-KV placement without reducing context capacity or precision.

**Blocked by:** [01: Qualify complete near-KV attention](01-qualify-near-kv-attention.md).

**Status:** ready-for-agent; execution gated by the [parent specification](../spec.md), not started.

**Mandatory method:** Invoke/read `/skill:tdd`. At the approved Engine seam, start with one failing generation scenario that requires remote attention, make it pass end to end, then add simultaneous independent progress. Work directly, without subagents; do not mock Program, KV stores or attention.

- [ ] Carry the selected physical placement through startup resource planning, prefill, ordinary generation and terminal resource settlement as one usable path.
- [ ] Reserve two complete per-request windows and all required primary/secondary workspace before admission; account for vision, recurrent/speculative resources and secondary headroom.
- [ ] Demonstrate two active requests progressing before either finishes while their actual combined occupied KV exceeds primary capacity. Merely reserving large future output budgets is insufficient.
- [ ] Preserve independent context identity, causal visibility, valid columns and position semantics across page/placement boundaries.
- [ ] Keep weight-heavy execution, GDN and vision on the 5090; restrict secondary computation to approved attention work.
- [ ] Preserve the existing Engine scheduling/publication policy. No resident-aware serialization or hidden reduction of the supported window is allowed.
- [ ] Measure generation with actual remote occupancy and distinguish numerical correctness from any unproven speed target.
- [ ] Update the affected active execution/resource contracts to describe the narrowly approved multi-device attention behavior, without claiming later speculative or deployment gates have passed.

**Completion evidence:** Real-Engine red/green results, bounded simultaneous-generation smoke output and actual memory/occupancy evidence. Small deterministic fixtures may force low primary capacity for regression tests; they do not replace later production-size qualification.
