#pragma once

#include "runtime/engine/context_capacity.h"

namespace ninfer::models {

enum class WeightPlacement { Resident, HostExperts };

// Main text width and depth reported to the Engine description.
struct TextGeometry {
    std::uint32_t hidden_size       = 0;
    std::uint32_t num_hidden_layers = 0;
};

struct WeightCandidate {
    WeightPlacement placement;
    std::size_t device_bytes;
    std::size_t host_bytes;
};

template <class SequencePlan, class Preparation>
struct ExecutionPlan {
    SequencePlan sequence;
    runtime::ContextCapacityResolution capacity;
    WeightPlacement placement;
    std::uint32_t chunk;
    Preparation preparation;
};

} // namespace ninfer::models
