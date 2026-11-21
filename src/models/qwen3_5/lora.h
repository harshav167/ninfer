#pragma once

#include "artifact/reader.h"
#include "models/qwen3_5/definition.h"
#include "ninfer/ops/low_rank.h"

namespace ninfer::models::qwen3_5 {

using LoraProjection = ops::LowRankFactors;

struct LoraAttention {
    LoraProjection query, key, gate, value, output;
};

struct LoraBlock {
    std::optional<LoraAttention> attention;
    LoraProjection gate, up, down;
};

struct LoraWeights {
    artifact::MaterializedArtifact backing;
    std::vector<LoraBlock> layers;
    float scale;
};

class LoraLoad {
public:
    LoraLoad(const std::filesystem::path&, const ModelDefinition&, Device);
    LoraLoad(const LoraLoad&)            = delete;
    LoraLoad& operator=(const LoraLoad&) = delete;

    [[nodiscard]] std::size_t device_bytes() const { return placement_.device_capacity_bytes; }

    [[nodiscard]] LoraWeights materialize(DeviceContext&, const StartupObserver&);

private:
    struct Pair {
        artifact::ObjectHandle a, b;
        std::int32_t k, n;
    };

    struct Block {
        std::optional<std::array<Pair, 5>> attention;
        std::array<Pair, 3> mlp;
    };

    artifact::Reader reader_;
    artifact::MaterializationPlan placement_;
    std::vector<Block> layers_;
    float scale_;
};

} // namespace ninfer::models::qwen3_5
