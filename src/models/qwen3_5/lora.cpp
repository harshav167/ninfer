#include "models/qwen3_5/lora.h"

#include <cmath>
#include <set>
#include <utility>

namespace ninfer::models::qwen3_5 {

LoraLoad::LoraLoad(const std::filesystem::path& path, const ModelDefinition& target, Device device)
    : reader_(path) {
    const auto& directory = reader_.directory();
    if (directory.components.size() != 1 || !directory.components.contains("lora") ||
        !directory.uses.empty())
        throw artifact::ArtifactError("expected an adapter-only LoRA companion");
    const auto& config = directory.component("lora").config;
    artifact::require_members(config, {"rank", "scale", "base_artifact_id"}, {}, "LoRA config");
    const auto id = target.artifact_id();
    std::string hex;
    hex.reserve(2 * id.size());
    for (auto byte : id) {
        constexpr char digits[] = "0123456789abcdef";
        const auto value        = std::to_integer<unsigned>(byte);
        hex.push_back(digits[value >> 4]);
        hex.push_back(digits[value & 15]);
    }
    if (config.at("base_artifact_id") != hex || config.at("rank") != ops::kLowRank ||
        !config.at("scale").is_number())
        throw artifact::ArtifactError("LoRA target artifact, rank or scale does not match");
    scale_ = config.at("scale").get<float>();
    if (!std::isfinite(scale_) || scale_ <= 0)
        throw artifact::ArtifactError("LoRA scale must be finite and positive");

    std::vector<artifact::PlacementRequest> requests;
    std::set<std::size_t> used_objects;
    std::size_t bindings = 0;
    const auto tensor    = [&](const std::string& name, std::uint64_t width) {
        const auto found = directory.bindings.find(name);
        if (found == directory.bindings.end() || !found->second.whole_object)
            throw artifact::ArtifactError("missing complete LoRA tensor: " + name);
        ++bindings;
        const auto object = found->second.parts.front().object;
        const auto& t     = directory.tensor(object);
        if (t.format != "fp32" || t.layout != "contiguous_le_v1" ||
            t.shape != artifact::Shape{ops::kLowRank, width})
            throw artifact::ArtifactError("LoRA tensor representation mismatch: " + name);
        if (used_objects.insert(object.index).second) requests.push_back({object, device});
        return object;
    };
    const auto pair = [&](WeightId id) -> Pair {
        const auto& weight = target.weight_data()[id.index];
        const auto& shape  = weight.view.shape;
        return {tensor(weight.name + "/a", shape.at(1)), tensor(weight.name + "/b", shape.at(0)),
                static_cast<std::int32_t>(shape[1]), static_cast<std::int32_t>(shape[0])};
    };
    const auto& source = target.weights().text.layers;
    layers_.reserve(source.size());
    for (const auto& layer : source) {
        const auto& mlp = std::get<DenseWeights>(layer.ffn);
        Block block{std::nullopt, {pair(mlp.gate), pair(mlp.up), pair(mlp.down)}};
        if (const auto* a = std::get_if<AttentionWeights>(&layer.mixer)) {
            block.attention = {pair(a->query), pair(a->key), pair(a->gate), pair(a->value),
                               pair(a->output)};
            if ((*block.attention)[0].a != (*block.attention)[2].a)
                throw artifact::ArtifactError("LoRA query and gate must share A");
        }
        layers_.push_back(block);
    }
    if (bindings != directory.bindings.size() || used_objects.size() != directory.objects.size())
        throw artifact::ArtifactError("LoRA companion has unbound tensors");
    placement_ = artifact::plan_materialization(reader_, requests);
}

LoraWeights LoraLoad::materialize(DeviceContext& device, const StartupObserver& observer) {
    LoraWeights weights{
        artifact::materialize(reader_, std::move(placement_), device.transfer_view(), &observer),
        {},
        scale_};
    const auto tensor = [&](artifact::ObjectHandle id, std::int32_t width) {
        const auto& parent = weights.backing.device_parent(id);
        return Tensor(const_cast<std::byte*>(parent.data), parent.device, DType::FP32,
                      {width, ops::kLowRank});
    };
    const auto view = [&](const Pair& p) -> LoraProjection {
        return {tensor(p.a, p.k), tensor(p.b, p.n)};
    };
    weights.layers.reserve(layers_.size());
    for (const auto& block : layers_) {
        LoraBlock lora{std::nullopt, view(block.mlp[0]), view(block.mlp[1]), view(block.mlp[2])};
        if (block.attention) {
            const auto& a = *block.attention;
            lora.attention =
                LoraAttention{view(a[0]), view(a[1]), view(a[2]), view(a[3]), view(a[4])};
        }
        weights.layers.push_back(lora);
    }
    return weights;
}

} // namespace ninfer::models::qwen3_5
