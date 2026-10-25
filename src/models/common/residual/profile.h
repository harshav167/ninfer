#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::residual {

// One edited layer. After the layer's FFN residual add, per token column h:
//   h1 = h  - max(dot(h,  refusal)    - refusal_reference,    0) * refusal
//   h2 = h1 - max(dot(h1, disclaimer) - disclaimer_reference, 0) * disclaimer
// `refusal` has unit L2 norm (1 +/- 1e-3); `disclaimer` does too, or is exactly zero, which makes
// the second step the identity. Values are exactly the FP32 values read from the profile.
struct LayerEdit {
    std::uint32_t layer = 0;
    std::vector<float> refusal;
    float refusal_reference = 0.0F;
    std::vector<float> disclaimer;
    float disclaimer_reference = 0.0F;
};

// Validated profile JSON v1. `layers` is non-empty and strictly ascending by `layer`, every
// `layer < layer_count`, and every direction holds `hidden_size` finite FP32 values.
struct Profile {
    std::string artifact_id; // 32 lowercase hexadecimal characters
    std::uint32_t hidden_size = 0;
    std::uint32_t layer_count = 0;
    std::vector<LayerEdit> layers;
};

// Parses and validates one profile document. Unknown members (except an object `provenance`),
// duplicate JSON members, trailing text and any schema violation throw std::invalid_argument
// whose message names the offending field, e.g. `layers[3].refusal_direction[17]`.
[[nodiscard]] Profile parse_profile(std::string_view json_text);

// Reads `path` and parses it; failures throw std::invalid_argument naming the path.
[[nodiscard]] Profile read_profile(const std::filesystem::path& path);

// Requires the profile to describe exactly this artifact container and model geometry. The model
// owner supplies `artifact_id` as 32 lowercase hexadecimal characters (artifact_id_string).
// Throws std::invalid_argument with both the profile's and the model's values on any mismatch.
void require_binding(const Profile& profile, std::string_view artifact_id,
                     std::uint32_t hidden_size, std::uint32_t layer_count);

} // namespace ninfer::models::residual
