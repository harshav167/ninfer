#include "ops/linear/split/fp8/shapes/n14336_k5120/launch.h"

namespace ninfer::ops::detail::linear_split::fp8_n14336_k5120 {
namespace {
bool quantized(LinearPolicy policy, int columns) { return allows_a8(policy) && columns >= 17; }

std::size_t capacity(LinearPolicy policy, std::int32_t, std::int32_t last) {
    return quantized(policy, last)
               ? fp8_a8_workspace_capacity_bytes(last, 5120, partial_capacity(last))
               : 0;
}

void launch_adapted(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
                    MemoryArena* workspace, cudaStream_t stream, const RankDeltas* delta) {
    if (!quantized(weights.policy, x.ne[1])) {
        launch_a16(x, weights.first, outputs, stream, delta);
        return;
    }
    const auto scratch =
        allocate_fp8_a8_workspace(*workspace, x.ne[1], 5120, partial_capacity(x.ne[1]));
    launch_a8(x, weights.first, outputs, scratch, stream, delta);
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    launch_adapted(x, weights, outputs, workspace, stream, nullptr);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::fp8_n14336_k5120

namespace ninfer::ops::detail::linear_split {
const SplitShape kFp8SplitN14336K5120{14336,
                                      5120,
                                      kSplitRows<6144, 1024, 6144, 1024>,
                                      fp8_n14336_k5120::launch,
                                      fp8_n14336_k5120::capacity,
                                      0,
                                      fp8_n14336_k5120::launch_adapted};
} // namespace ninfer::ops::detail::linear_split
