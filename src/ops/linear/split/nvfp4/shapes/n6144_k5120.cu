#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
            cudaStream_t stream) {
    const auto operands = nvfp4_a16_operands(x, weights.first);
    const LinearBf16SegmentedOutput<4096, 1024, 1024> output{
        static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
        static_cast<__nv_bfloat16*>(outputs[2].data)};
    if (x.ne[1] <= 8)
        return launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    if (x.ne[1] <= 16)
        return launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    launch_nvfp4_a16_mma<
        Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>, 5120>>(
        operands, output, LinearIdentityEpilogue{}, stream);
}
} // namespace

const SplitShape kNvfp4SplitN6144K5120{6144, 5120, kSplitRows<4096, 1024, 1024>, launch};
} // namespace ninfer::ops::detail::linear_split
