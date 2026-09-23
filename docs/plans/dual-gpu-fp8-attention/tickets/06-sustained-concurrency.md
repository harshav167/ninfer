# 06: Meet sustained dual-deep FP8 performance gates

**What to build:** Two real deep-context requests sustain at least 150 committed tokens/second each while decoding together, without weakening quality, context, vision or reasoning requirements.

**Blocked by:** [04: Preserve context reuse, isolation and cancellation](04-context-lifecycle.md); [05: Qualify graph-backed near-KV execution](05-graph-replay.md).

**Status:** ready-for-agent; execution gated by the [parent specification](../spec.md), not started.

**Mandatory method:** Invoke/read `/skill:tdd` before behavioral code changes. Use red–green cycles at the approved seams for changed contracts; use reproducible hardware experiments for performance, not unit tests asserting wall-clock thresholds. Work directly without subagents. Preserve the existing server until a coordinated experiment is authorized.

- [ ] Freeze matched model/workload/sampling/thinking conditions and collect a baseline; use verified tokenizer counts, not filler estimates.
- [ ] Measure supported speculative choices by committed tokens per total round time. Keep DFlash2 K7 as the reference; do not expand MTP limits without a separately approved slice.
- [ ] Exercise independent paired prompts near 230K and 245K with output headroom and growth toward the 250K limit, FP8 K/V, thinking enabled and vision available.
- [ ] Complete at least three paired runs per selected workload: at approximately 230K initial depth, at least 60 seconds of genuine overlapping decode; at approximately 245K, at least 3,000 committed tokens per stream during overlap within the remaining context. Report actual durations and preserve output headroom; insufficient overlap is not a pass.
- [ ] Meet at least 150 TPS for each stream over that interval and at least 300 aggregate, counting committed reasoning/final tokens and excluding rejected drafts.
- [ ] Report rolling per-stream rates and p50/p95/p99/max inter-token gaps, acceptance, round latency, remote bytes and memory occupancy. Obtain an explicit stall acceptance decision before deployment.
- [ ] Separately report cold prefill/TTFT, interference during another request's prefill and warm continuation latency. Do not imply pure-decode throughput also establishes an unmeasured prefill-interference SLO.
- [ ] Preserve numerical and lifecycle qualification after each optimization. Investigate the measured critical path rather than assuming larger drafts or extra buffers help.
- [ ] If the target fails, keep the ticket incomplete and report the limiting measurements and alternatives. No serialization, precision reduction, shorter windows or changed target can be called a successful implementation.

**Completion evidence:** Interpretable repeated end-to-end results with hardware/toolchain, workload, settings, overlap intervals and per-stream metrics. A microbenchmark or aggregate average after one request finishes cannot close this ticket.
