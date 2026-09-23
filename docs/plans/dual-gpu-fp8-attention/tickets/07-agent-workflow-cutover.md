# 07: Validate real agent workflows and approved cutover

**What to build:** The owner can run two deep OMP coding agents against the existing LAN Responses service with preserved reasoning, tools, vision and continuation behavior, using the qualified implementation rather than a benchmark-only setup.

**Blocked by:** [06: Meet sustained dual-deep FP8 performance gates](06-sustained-concurrency.md). Production cutover additionally requires explicit owner approval after reviewing quality and stall evidence.

**Status:** ready-for-agent; execution gated by the [parent specification](../spec.md), not started.

**Mandatory method:** Invoke/read `/skill:tdd` for any remaining implementation changes, keeping new regression tests at the approved seams. Exercise the real HTTP/OMP surface as a smoke test rather than adding a new private seam. Work directly, without subagents.

- [ ] Run representative paired OMP reasoning/tool workflows, with raw reasoning replay, previous-response chaining, repeated tool results and actual long-context reuse.
- [ ] Exercise vision and subsequent text continuation without changing the model's normal reasoning configuration.
- [ ] Compare quality against matched single-stream FP8 runs using the existing coding-task grader; do not substitute OMP self-grading or needle recall for task correctness.
- [ ] Preserve the current Responses marker fix, reasoning-preservation integration, LAN bind and advertised external protocol behavior; no OMP source changes.
- [ ] Demonstrate that actual occupied deep contexts, not only reservations or nominal configuration, coexist while both agents make progress.
- [ ] Review per-stream throughput, mixed-prefill behavior and stall distributions with the owner; unresolved quality regressions or unaccepted stalls block cutover.
- [ ] Before deployment, obtain explicit authorization for the service transition and preserve a working rollback configuration/binary without silently changing precision or concurrency.
- [ ] After approved cutover, verify the live endpoint and paired agent behavior; record exactly what was exercised and remaining limitations.
- [ ] Integrate enduring contracts, commands and qualified claims into the existing active documentation. Remove superseded implementation paths and temporary planning material only when the delivered cutover actually obsoletes them; preserve unrelated work.
- [ ] Do not create commits, PRs or external issues unless separately requested.

**Completion evidence:** Existing-grader quality results, real Responses/OMP smoke outcomes, approved performance/stall review and live verification after an authorized transition. Passing lower-level tickets alone does not establish usable agent performance.
