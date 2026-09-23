# Dual-stream FP8 inference with near-KV attention

**Status:** proposed; documentation only. Implementation is not authorized by creation of these documents.
**Triage:** ready-for-agent, subject to the execution gates below.
**Execution method:** `/skill:tdd`, working directly without subagents.

## Problem Statement

The owner needs two genuinely concurrent coding agents, each with approximately 230K–250K tokens of context, useful reasoning and tool behavior, and sustained decoding near 150 tokens/second per stream. A fast single request, two serialized requests, or a short high-acceptance recall benchmark does not satisfy this requirement.

The current QUASAR deployment stores FP8 keys and values in a shared 500,096-token pool: 320,000 tokens on an RTX 5090 and the remainder on an RTX 3070. Model computation runs on the 5090. Active remote KV is repeatedly copied through pinned host memory to primary-device staging buffers before attention. The capacity works, but dual-deep decoding suffers a severe bandwidth cliff. Active-first placement fixed unnecessary spilling caused by inactive retained contexts; it cannot eliminate remote reads when both active contexts exceed primary capacity.

The owner has rejected quality-degrading KV settings and serialization as remedies. The published Cinference recipe is not a replacement: it measures one stream with K8V4, MTP-10, thinking disabled, and short synthetic recall outputs. Its headline 450.78 TPS is at 8K context, not two deep thinking agents.

## Solution

Keep QUASAR weights, FP8 K/V, thinking and reasoning replay, vision, and two complete concurrent context windows. Replace repeated historical-KV transport with computation near the remote KV: the 3070 executes only the attention work needed for its assigned cache, while weight-heavy model execution remains on the 5090. Communicate queries, new KV, and compact attention results instead of rereading the remote history across PCIe each round.

The preferred candidate is context-sharded attention with numerically stable partial-result merging. Compare it against assigning complete attention calculations and KV for selected full-attention layers to the 3070. Choose using numerical qualification, real memory budgets, and integrated latency; neither placement is predetermined to win.

This explicitly changes the previous storage-only role of the 3070. Approval is required before implementation. The performance target is an acceptance requirement, not an already-established capability or a guarantee based on this design alone.

## User Stories

1. As a coding-agent user, I want two agents to decode simultaneously, so that one does not wait for the other to finish.
2. As a coding-agent user, I want each agent to retain a full 250K-token window, so that concurrency does not silently divide a single context allowance.
3. As a coding-agent user, I want FP8 keys and values, so that memory expansion does not restore the rejected 4-bit KV settings.
4. As a coding-agent user, I want reasoning enabled, so that performance measurements represent my actual workflow.
5. As a coding-agent user, I want raw reasoning preserved across turns, so that continuation behavior remains consistent.
6. As a coding-agent user, I want vision available, so that screenshots and images remain usable inputs.
7. As a coding-agent user, I want tool calls and their results to survive long conversations, so that the harness can complete real coding tasks.
8. As a coding-agent user, I want both streams to sustain at least 150 TPS during overlapping decode, so that aggregate throughput cannot hide a stalled agent.
9. As a coding-agent user, I want per-stream stall measurements, so that a good average cannot hide long periods without output.
10. As a coding-agent user, I want a second cold prefill measured separately from steady decoding, so that startup interference is visible.
11. As a coding-agent user, I want warm continuations to reuse their valid context, so that every tool turn does not repeat a deep prefill.
12. As a coding-agent user, I want independent contexts to remain isolated, so that one agent cannot read the other's state.
13. As a coding-agent user, I want shared prefixes to retain their existing semantics, so that reuse does not corrupt branches.
14. As a coding-agent user, I want cancelling one request to leave the other usable, so that interruption does not poison the server.
15. As a coding-agent user, I want rejected speculative tokens to be discarded on both devices, so that later generation uses only committed history.
16. As a coding-agent user, I want progress when contexts cross page and device-placement boundaries, so that long conversations do not encounter a sudden collapse.
17. As a coding-agent user, I want inactive cache retention not to displace necessary active resources, so that useful history does not make active agents unusable.
18. As a coding-agent user, I want the existing LAN Responses endpoint and previous-response chaining, so that my clients need no protocol migration.
19. As the machine owner, I want the 5090 to retain weight-heavy execution, so that the 3070 does not need another full model copy.
20. As the machine owner, I want bounded secondary workspace and explicit capacity accounting, so that the 8 GB card remains safe alongside its other uses.
21. As the machine owner, I want the current service left intact until the candidate passes qualification, so that research does not silently replace production.
22. As a maintainer, I want attention compared with an independent mathematical oracle, so that plausible text is not mistaken for numerical correctness.
23. As a maintainer, I want externally observable lifecycle tests, so that internal implementation changes do not invalidate useful regression coverage.
24. As a maintainer, I want speculation selected by committed tokens per unit time, so that a larger draft window is not mistaken for a speedup.
25. As a maintainer, I want graph replay to preserve attention and transaction semantics, so that lower launch overhead cannot introduce stale state.
26. As the machine owner, I want an explicit failure report if the target cannot be met, so that quality, context, or concurrency are never silently reduced.

## Implementation Decisions

### Execution gates

- Confirm the two test seams described below before writing tests.
- Approve the narrowly scoped secondary-device attention role; the previous storage-only restriction remains binding until then.
- Confirm ticket granularity and dependencies before implementation starts. Writing local tickets does not imply that review has occurred.
- Retain the stronger interpretation of the throughput objective: 150 TPS per stream, at least 300 aggregate, during genuine overlapping decode. Any weaker target requires an explicit decision.

### Architecture and ownership

- Model owns immutable selected weights and fixed mathematics. No second full model or weight-heavy execution is placed on the 3070.
- Program owns the two-device execution composition, mutable resources, State/KV lifetime, placement, reservations, speculative commit/rollback, and graph resources.
- Core owns bounded buffers, streams, events, physical storage and raw transfers; it does not acquire request scheduling policy or attention mathematics.
- The causal Softmax Attention Op owns the complete mathematical transformation and its private partial reductions, codec handling, remote execution and merge implementation. Avoid new model-callable fragment Ops merely to expose kernel stages.
- Engine retains FIFO ingress, active request ownership, compact decode batches, resource settlement and publication policy. No resident-capacity admission workaround may serialize the supported dual-deep workload.
- Gateway and Frontend retain protocol, media acquisition, template and output semantics. The serving wire contract is unchanged.
- This is an explicit, narrow extension of the current single-compute-device product boundary. Update the relevant active architecture/resource authorities when implementation is approved and delivered; this temporary plan does not describe already-shipped behavior.

### Attention and representation

- Preserve the FP8 stored codes, scales and represented-value contract; preserve causal masks, grouped-query geometry, valid-column masks and position semantics.
- For context sharding, cover every visible key exactly once and merge partial statistics stably, including empty shards. No token dropping, approximate retrieval, attention sparsification or context truncation.
- The secondary kernel must support Ampere. Native FP8 MMA in the current primary kernel cannot simply be reused on the 3070. Decode tiles for supported arithmetic without permanently expanding the entire remote cache to FP16.
- Retain all existing semantic cast/quantization boundaries. Numerical acceptance comes from the independent oracle, not bitwise identity between different reduction orders.
- Compare context sharding against selected-layer attention placement through the complete attention operation. Include communication, synchronization, workspace and local/remote computation in the comparison.
- Prefer sufficient primary residency and the minimum useful remote allocation. Account for scratch, graphs, vision, recurrent state, speculative state, cold cache and secondary-device headroom rather than treating all nominal VRAM as available KV.
- Transport cannot assume peer access. The recorded machine requires host staging. Transfers must have explicit lifetime, completion and cancellation ownership.

### Speculation and graphs

- Preserve DFlash2 K7 as the baseline. Every provisional append, accepted prefix, rejection, forced control and rollback must remain correct across placement boundaries.
- Tune only supported, numerically qualified speculative configurations by committed-token throughput on reasoning/code workloads. Do not increase a draft window based on recall acceptance alone.
- Cinference MTP-10 and capture-derived graph grouping are useful follow-on candidates, not required imports. MTP has a different state and growing-KV budget. Expanding unsupported MTP behavior requires a separately approved slice rather than silently enlarging these tickets.
- Restore graph execution only where the selected distributed execution path is safely capturable. Compatibility must include actual graph update validity; node/function hashes alone are insufficient proof.

### Operational constraints

- Work directly; no subagents. Implementation work must invoke/read `/skill:tdd` before each new red–green cycle.
- Build with `cmake --build build -j 2`, selecting affected targets where appropriate. Do not change system resource limits, drivers or dependencies unless separately authorized and necessary.
- Keep production running unchanged during planning. Coordinate any later exclusive GPU validation explicitly. No automatic production cutover, commits, PRs or issue publication.
- Keep existing artifacts; no model replacement, conversion or download is authorized by this plan.

## Testing Decisions

### Proposed seams — awaiting confirmation

1. **Primary behavioral seam: public Engine generation.** Submit real requests, consume GenerationHandle events/results, and inspect existing public runtime summaries where necessary. Exercise concurrency, cache reuse, state settlement, cancellation and recovery through this boundary. Do not introduce private allocator/stager test APIs for the feature.
2. **Necessary numerical seam: complete causal Softmax Attention Op.** Engine text output cannot establish floating-point correctness. Invoke the real production route and independently decode packed K/V for the existing naive FP32/FP64 oracle. Cover the represented input domain and relevant real model shapes.

These are the only proposed new-test locations by semantic boundary. Existing HTTP/Responses behavior is exercised as a deployment smoke test, not a new third feature API. Existing protocol regression tests remain applicable if a changed contract affects them.

### Prior art and assertions

- Existing real-Engine prefix tests already exercise concurrent resource settlement, prefix reuse and request cancellation. Extend that style rather than mocking ResourceManager or Program collaborators.
- Existing causal-cache tests and the naive dense Softmax Attention oracle separate represented values and visible keys from production addressing, tiling, staging casts and reduction trees. Reuse that independence.
- Numerical cases must include empty/local-only/remote-only contributions, uneven splits, page tails, grouped-query heads, two rows with unequal frontiers, causal speculative columns, scale extremes, and zero-valued rows where supported. Select distinct boundary failures rather than a large redundant Cartesian product.
- Engine cases must demonstrate simultaneous progress, correct continuation after rejected speculation, independent context identity, cancellation of one active request, shared-prefix branching, and successful reuse after resource settlement.
- Assert results, state semantics, errors and meaningful progress—not private method calls, exact buffer sizes, kernel launch counts, source text or graph-signature wording.

### Mandatory TDD workflow

- After seam confirmation, use `/skill:tdd`: one observable behavior, one failing test, the smallest complete implementation, then green before the next slice.
- Record the red failure and green command/result for each implemented behavior. A missing fixture or unavailable second GPU is a blocked check, not a passing test.
- Do not write all tests in advance or all implementation before tests. Do not mock in-repository collaborators to manufacture success.
- Numerical tests fail against unsupported or incorrect production behavior and pass against the actual new route. Performance experiments are measured experiments, not fragile unit tests asserting wall-clock thresholds.
- Review/refactoring follows working vertical slices. No broad prefactor is justified until a concrete slice requires it.

### Performance and quality qualification

- Freeze a versioned workload description and baseline before comparison: same QUASAR artifact, tokenizer-sized inputs, thinking effort, sampling, vision availability and output budget. Keep thinking enabled without adding a benchmark-only cap that changes normal quality requirements.
- Exercise two independent requests at approximately 230K and 245K prompt depths, with growth toward the 250K per-request limit. Leave output headroom; never request generation beyond a full context window.
- Use at least three paired runs per selected deep workload. At approximately 230K initial depth, require at least 60 seconds of genuine overlapping decode with sufficient output headroom. At approximately 245K depth, require at least 3,000 committed tokens per stream during overlap, bounded by the remaining context; report the actual duration rather than imposing an impossible 60-second requirement within a 5K-token remainder. Early completion without the applicable overlap evidence is insufficient, not a pass.
- Require at least 150 committed tokens/second for each stream over the overlap interval and at least 300 aggregate. Report reasoning and final-output token counts separately as well as their total; do not count rejected drafts.
- Report per-stream rolling throughput and p50/p95/p99/max inter-token gaps. The numerical stall SLO is not yet agreed; expose the distributions and seek acceptance before cutover rather than inventing or hiding one.
- Report cold TTFT/prefill, interference while another stream prefills, warm-resume latency, pure overlapping decode, acceptance and total round latency separately. A pure-decode pass cannot be advertised as meeting an unmeasured mixed-prefill SLO.
- Use actual OMP reasoning/tool workflows for quality and usability, retaining the existing grading authority for coding tasks. Compare matched single-stream FP8 and concurrent candidate runs; do not use synthetic recall alone or substitute a subjective self-grader.
- Numerical, lifecycle and throughput gates must pass before a deployment claim. Any unresolved quality regression or unaccepted stalls block cutover even if aggregate TPS passes.

## Out of Scope

- Replacing QUASAR with Huihui or another checkpoint to reproduce a headline.
- NVFP4 KV, K8V4 values, sparse attention, token dropping, silent window reduction, disabled reasoning or serialization of the two supported streams.
- General tensor/model parallelism, a second full model, or moving GDN/FFN/vision onto the 3070.
- CPU attention workers, new hardware, interconnect upgrades or a wholesale engine migration.
- Implementing MTP-10 merely because Cinference supports it; it remains a separately scoped candidate.
- New scheduling priorities, active-request preemption, broader concurrency than the requested two streams, or protocol redesign.
- Changes to OMP source, unrelated system configuration, or the paused model-evaluation campaign.
- Production deployment or runtime implementation during this documentation task.

## Further Notes

### Evidence and uncertainty

Recorded local accounting gives 33,024 bytes per token for FP8 main KV. At 500K tokens that is 15.38 GiB, plus 17.99 GiB of model weights before other resources. Approximately 180K tokens outside primary residency correspond to 5.94 GB. The recorded host-staged secondary-to-primary transfer rate is 2.93 GB/s, or roughly two seconds for one such sweep. This explains why buffering alone cannot meet the target; it is not a universal impossibility proof for the hardware.

Prior all-primary NVFP4-KV testing recorded 322.25 aggregate TPS during 65 seconds of dual-deep decode. That establishes a historical workload result, not acceptable FP8 quality or the performance of the proposed design. Active-first residency validation was sequential and cannot be presented as dual-deep recovery.

At 150 TPS per stream, four/six/eight committed tokens per round allow approximately 26.7/40.0/53.3 ms per round. The secondary attention kernel, communication and complete model execution must be measured against those budgets. No performance result is claimed for the proposed implementation.

### Research basis

- [Flash-Decoding](https://pytorch.org/blog/flash-decoding/): exact sequence-split attention and normalization-aware merging.
- [FastDecode](https://arxiv.org/abs/2403.11421): compute near KV and exchange activations rather than repeatedly transporting historical KV; its CPU-worker performance does not transfer directly to this machine.
- [vLLM context parallelism](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/): deployed KV sharding integrated with tensor parallelism, not a drop-in heterogeneous helper configuration.
- [TensorRT-LLM Helix](https://nvidia.github.io/TensorRT-LLM/latest/features/helix.html): distributed partial attention; documented model/hardware restrictions differ from this target.
- [Cinference measurements](https://github.com/satellitedown/cinference/blob/main/results/rtx5090-archive-recall.json): single-stream recall, K8V4, thinking off; not a concurrent agent-quality claim.

### Local ticket index

Each ticket is a complete observable slice. Blocking edges describe technical dependencies, not permission to use parallel agents. All tickets also inherit the execution gates above.

| Ticket | Delivers | Blocked by |
|---|---|---|
| [01: Qualify near-KV attention](tickets/01-qualify-near-kv-attention.md) | Correct complete two-device attention and measured placement selection | Approval gates |
| [02: Run concurrent FP8 generation](tickets/02-concurrent-fp8-generation.md) | Two independent Engine requests with remote attention | 01 |
| [03: Preserve speculative transactions](tickets/03-speculative-transactions.md) | Correct DFlash2 concurrent commit/rollback | 02 |
| [04: Preserve context lifecycle](tickets/04-context-lifecycle.md) | Resume, branching, cancellation and settled resources | 03 |
| [05: Qualify graph replay](tickets/05-graph-replay.md) | Correct graph-backed execution where supported, or measured explicit limitation | 03 |
| [06: Meet sustained concurrency gates](tickets/06-sustained-concurrency.md) | Qualified deep-context performance and speculation choice | 04, 05 |
| [07: Validate agent workflow and cutover](tickets/07-agent-workflow-cutover.md) | Quality-qualified OMP workflow and explicitly approved deployment | 06 |

After completion or abandonment, fold lasting decisions into the existing active authorities and remove this temporary plan rather than leaving competing architecture documents.
