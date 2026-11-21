#pragma once
#include "ops/linear/add/dispatch.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/common/low_rank.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail {

struct Nvfp4LinearAddPlan {
    void (*a16)(const Tensor&, const Weight&, Tensor&, cudaStream_t, const RankDelta*) = nullptr;
    void (*quantized)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t,
                      const RankDelta*)                                                = nullptr;
    Nvfp4ScaleLayout scale_layout = Nvfp4ScaleLayout::RowMajor;
};

Nvfp4LinearAddPlan select_nvfp4_add_n5120_k6144(LinearPolicy policy, std::int32_t tokens);
Nvfp4LinearAddPlan select_nvfp4_add_n5120_k17408(LinearPolicy policy, std::int32_t tokens);
void launch_nvfp4_add_n5120_k6144_tma(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace,
                                      cudaStream_t, const RankDelta*);
void launch_nvfp4_add_n5120_k17408_tma(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace,
                                       cudaStream_t, const RankDelta*);
std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t rows, std::int32_t k,
                                                      LinearPolicy policy, std::int32_t first,
                                                      std::int32_t last);
void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, MemoryArena&,
                               cudaStream_t, const RankDelta*);
} // namespace ninfer::ops::detail
