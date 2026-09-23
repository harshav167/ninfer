# 01: Qualify complete near-KV attention

**What to build:** A complete causal Softmax Attention invocation that uses the 3070's local FP8 cache without transporting its full historical KV to the 5090, producing numerically qualified output for real model geometry. Compare context-sharded partial attention with selected-layer attention placement and select the feasible design using integrated measurements.

**Blocked by:** No technical predecessor. Execution is blocked until the owner approves secondary-device attention, the two proposed test seams, and the ticket breakdown in the [parent specification](../spec.md).

**Status:** ready-for-agent; execution gated, not started.

**Mandatory method:** Invoke/read `/skill:tdd` before implementation. Work directly, without subagents. At the approved complete attention-Op seam, write one failing behavioral/numerical test, implement that complete case, demonstrate green, then repeat. Follow the parent's operational and qualification constraints.

- [ ] Record explicit approval of the 3070 compute-role change and the proposed seams before writing tests.
- [ ] Preserve FP8 codes/scales and the existing represented-value contract; do not permanently expand remote history to FP16.
- [ ] Use an Ampere-compatible production route and independently decoded packed operands against the existing naive FP32/FP64 attention oracle.
- [ ] Establish local-only, remote-only and mixed visible-key behavior, then uneven/page-tail partitions, two unequal rows and causally masked speculative widths through successive red–green cycles.
- [ ] Verify that every visible key contributes exactly once and empty contributions are numerically neutral; preserve grouped-query semantics and supported scale boundaries.
- [ ] Compare both placement candidates through complete attention calls, including local/remote computation, host-staged transport, synchronization and peak memory. Do not declare a winner from isolated bandwidth alone.
- [ ] Record the selected placement, actual hardware/toolchain, relevant shapes, numerical errors/criteria, latency distributions and memory budget. A failed feasibility result blocks downstream implementation rather than justifying reduced precision or concurrency.
- [ ] Keep the existing primary-only route working; do not replace production or download models.

**Completion evidence:** Red and green numerical results from the actual route, plus a reproducible bounded placement experiment. Missing hardware/fixtures are blockers, not passing results. This ticket proves attention behavior, not Engine throughput.
