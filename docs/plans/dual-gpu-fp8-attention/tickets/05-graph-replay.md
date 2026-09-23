# 05: Qualify graph-backed near-KV execution

**What to build:** The selected two-device generation path can reduce launch overhead through safe graph replay where feasible, without stale queries, stale results or altered state transitions. If capture is not feasible for part of the path, establish and measure that limitation explicitly rather than advertising graph support.

**Blocked by:** [03: Preserve concurrent speculative transactions](03-speculative-transactions.md).

**Status:** ready-for-agent; execution gated by the [parent specification](../spec.md), not started.

**Mandatory method:** Invoke/read `/skill:tdd`. Test graph-enabled behavior through the approved Engine and complete attention seams, not graph internals. Begin with a failing replay scenario that changes a relevant semantic input or execution envelope. Work directly, without subagents.

- [ ] Establish whether full or segmented capture is feasible with bounded cross-device transport and explicit synchronization; retain correct eager execution as the comparison route during qualification.
- [ ] For graph-backed execution, verify distinct inputs/frontiers across repeated replays, one-to-two-to-one batch transitions, and placement/envelope boundaries without stale data.
- [ ] Preserve the independently qualified attention criteria and speculative transaction semantics; replay success alone is not correctness.
- [ ] Validate executable update compatibility. Capture-derived node/function signatures may classify candidates but cannot substitute for actual compatibility checks.
- [ ] Measure integrated round latency and graph/workspace memory against eager execution at relevant depths; retain only a justified supported route.
- [ ] If graph support is infeasible, document the concrete failing boundary, measured eager cost and resulting performance risk. The performance ticket must still meet its full target or remain blocked; this branch is not a waiver.

**Completion evidence:** Red/green replay behavior for any delivered graph route and bounded end-to-end latency comparison, or an evidence-backed feasibility rejection. No requirement to import Cinference's MTP-specific graph code unchanged.
