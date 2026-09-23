# 03: Preserve concurrent speculative transactions

**What to build:** Both agents can use DFlash2 with near-KV attention while accepted tokens, rejected proposals and forced control leave exactly the supported committed continuation state on both devices.

**Blocked by:** [02: Run concurrent FP8 generation through the Engine](02-concurrent-fp8-generation.md).

**Status:** ready-for-agent; execution gated by the [parent specification](../spec.md), not started.

**Mandatory method:** Invoke/read `/skill:tdd`. Add one failing Engine-level transaction behavior at a time and implement it fully before the next case. Use the already-approved attention seam only for a newly exposed numerical boundary. Work directly, without subagents.

- [ ] Retain DFlash2 K7 as the initial baseline and preserve target verification semantics, reasoning and existing sampling behavior.
- [ ] Demonstrate correct continuation after partial acceptance/rejection, including a rejected tail crossing page or placement boundaries; rejected tokens must not become visible in later output.
- [ ] Exercise two rows with different frontiers, valid proposal widths and remaining context/output budgets without cross-request state contamination.
- [ ] Exercise forced-control handling and staggered completion so retiring one lane cannot invalidate the other's attention or recurrent-state selectors.
- [ ] Keep provisional remote writes and committed frontier publication within the existing Program transaction owner; events and transfers cannot outlive referenced state.
- [ ] Verify the final legal token positions near the per-request context limit without reading/writing speculative tokens beyond the supported envelope.
- [ ] Report committed tokens, rejected drafts and round latency distinctly; a configured draft count is not measured acceptance.

**Completion evidence:** Real production-route numerical checks where needed, Engine red/green transaction regressions and concurrent speculative-generation smoke results. Do not claim deterministic token equality when the applicable sampling/numerical contract only supports a different comparison criterion.
