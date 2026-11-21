#pragma once
#include "ops/linear/add/nvfp4/dispatch.h"
#include "ops/linear/common/low_rank_epilogue.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
template <class Instance>
void launch_nvfp4_add_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                           const RankDelta* delta) {
    auto* data     = static_cast<__nv_bfloat16*>(out.data);
    const auto run = [&](auto epilogue) {
        launch_nvfp4_a16_gemv<Instance>(nvfp4_a16_operands(x, w), LinearBf16Output{data, w.n},
                                        epilogue, stream);
    };
    const LinearBf16InputView residual{data, w.n};
    if (delta) return run(RankResidualAdd{residual, *delta});
    run(LinearResidualAddEpilogue{residual});
}

template <class Instance>
void launch_nvfp4_add_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                           const RankDelta* delta) {
    auto* data     = static_cast<__nv_bfloat16*>(out.data);
    const auto run = [&](auto epilogue) {
        launch_nvfp4_a16_simt<Instance>(nvfp4_a16_operands(x, w), LinearBf16Output{data, w.n},
                                        epilogue, stream);
    };
    const LinearBf16InputView residual{data, w.n};
    if (delta) return run(RankResidualAdd{residual, *delta});
    run(LinearResidualAddEpilogue{residual});
}

template <class Instance>
void launch_nvfp4_add_sliced_k_mma(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream, const RankDelta* delta) {
    auto* data     = static_cast<__nv_bfloat16*>(out.data);
    const auto run = [&](auto epilogue) {
        launch_nvfp4_a16_sliced_k_mma<Instance>(nvfp4_a16_operands(x, w),
                                                LinearBf16Output{data, w.n}, epilogue, stream);
    };
    const LinearBf16InputView residual{data, w.n};
    if (delta) return run(RankResidualAdd{residual, *delta});
    run(LinearResidualAddEpilogue{residual});
}

template <class Instance>
void launch_nvfp4_add_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                          const RankDelta* delta) {
    auto* data     = static_cast<__nv_bfloat16*>(out.data);
    const auto run = [&](auto epilogue) {
        launch_nvfp4_a16_mma<Instance>(nvfp4_a16_operands(x, w), LinearBf16Output{data, w.n},
                                       epilogue, stream);
    };
    const LinearBf16InputView residual{data, w.n};
    if (delta) return run(RankResidualAdd{residual, *delta});
    run(LinearResidualAddEpilogue{residual});
}

template <class Instance>
void launch_nvfp4_add_a4(const Tensor& x, const Weight& w, Tensor& out, Nvfp4A4Workspace scratch,
                         cudaStream_t stream, const RankDelta* delta) {
    auto* data     = static_cast<__nv_bfloat16*>(out.data);
    const auto run = [&](auto epilogue) {
        launch_nvfp4_a4_mma<Instance>(
            nvfp4_a4_operands(w, scratch, x.ne[1], Nvfp4ScaleLayout::RowMajor),
            LinearBf16Output{data, w.n}, epilogue, stream);
    };
    const LinearBf16InputView residual{data, w.n};
    if (delta) return run(RankResidualAdd{residual, *delta});
    run(LinearResidualAddEpilogue{residual});
}
} // namespace ninfer::ops::detail
