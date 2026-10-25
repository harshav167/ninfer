#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::execution {

// Immutable post-layer residual intervention. directions is FP32 [H,2,K] (column 0 refusal,
// column 1 disclaimer), references is FP32 [2,K], and slot k applies to layers[k], which is
// strictly ascending. The rank owns the storage; it outlives every traversal that borrows it.
struct ResidualIntervention {
    std::span<const std::uint32_t> layers;
    Tensor directions;
    Tensor references;
};

} // namespace ninfer::models::qwen3_5::execution
