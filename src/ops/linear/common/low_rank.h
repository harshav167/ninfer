#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/low_rank.h"

#include <array>
#include <span>

namespace ninfer::ops::detail {

// One output slice's epilogue operands: B[rank][rows] (rank-major planes,
// coalesced across rows) and the shared rank intermediates z[tokens][rank]
// for this slice's A matrix. The base kernel's epilogue evaluates
// base + scale * B[row] . z[token] per output element, before rounding.
struct RankDelta {
    const float* b = nullptr;
    const float* z = nullptr;
    int rows       = 0;
    float scale    = 0;
};

// Computes z for every distinct A among the factors in one grouped launch
// and returns the per-slice epilogue operands. Slices sharing an A pointer
// share its z. Workspace is low_rank_workspace_bytes(tokens, distinct_a).
using RankDeltas = std::array<RankDelta, 4>;
[[nodiscard]] RankDeltas prepare_low_rank(const Tensor& x, std::span<const LowRankFactors> factors,
                                          std::span<const int> rows, float scale, MemoryArena&,
                                          CudaExecutionView);

} // namespace ninfer::ops::detail
