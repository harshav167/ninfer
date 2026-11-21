#include "ops/linear/common/low_rank_epilogue.cuh"
#include "ops/linear/swiglu/nvfp4/tma_launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                cudaStream_t stream, const RankDeltas* updates) {
    using S        = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>;
    const auto run = [&](auto epilogue) {
        launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{output, p.rows / 2}, epilogue, stream,
                                   SwiGluTokenMajorMmaRows<S>{});
    };
    if (updates)
        return run(SwiGluTokenMajorEpilogue<RankSwiGluProduct>{{(*updates)[0], (*updates)[1]}});
    run(SwiGluTokenMajorMmaEpilogue{});
}
} // namespace ninfer::ops::detail
