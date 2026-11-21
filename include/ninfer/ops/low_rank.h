#pragma once

#include "core/tensor.h"

#include <cstddef>

namespace ninfer::ops {

inline constexpr int kLowRank = 32;

struct LowRankFactors {
    Tensor a; // FP32 [K, rank]
    Tensor b; // FP32 [N, rank], contiguous output rows per rank coordinate
};

// Workspace holds only the FP32 rank intermediates z = A*x, laid out
// [distinct_a][tokens][kLowRank]. The expansion B*z runs inside the base
// projection's epilogue; no output-width delta is ever materialized.
// distinct_a excludes slices that repeat an A matrix (they share z).
[[nodiscard]] constexpr std::size_t low_rank_workspace_bytes(int tokens, int distinct_a) {
    return sizeof(float) * std::size_t(tokens) * kLowRank * std::size_t(distinct_a);
}

} // namespace ninfer::ops
