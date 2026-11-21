#include "ops/linear/common/low_rank_epilogue.cuh"
#include "ops/linear/split/nvfp4/shapes/n14336_k5120/launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"

namespace ninfer::ops::detail::linear_split::nvfp4_n14336_k5120 {

void launch_a4_tma(const Nvfp4A4Operands& p, std::span<const Tensor> outputs, cudaStream_t stream,
                   const RankDeltas* delta) {
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
        static_cast<__nv_bfloat16*>(outputs[2].data), static_cast<__nv_bfloat16*>(outputs[3].data)};
    const auto run = [&](auto epilogue) {
        if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
            launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 4, 1>, 5120>>(
                p, output, epilogue, stream);
        else
            launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
                p, output, epilogue, stream);
    };
    if (delta)
        return run(
            RankSplit<6144, 1024, 6144>{{(*delta)[0], (*delta)[1], (*delta)[2], (*delta)[3]}});
    run(LinearIdentityEpilogue{});
}

} // namespace ninfer::ops::detail::linear_split::nvfp4_n14336_k5120
