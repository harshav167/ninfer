#include "models/common/residual/profile.h"

#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace ninfer::models::residual {
namespace {

using artifact::Json;

constexpr std::string_view kLabel       = "residual profile";
constexpr std::uint64_t kSchemaVersion  = 1;
constexpr std::uint64_t kMaxHiddenSize  = 16384;
constexpr std::uint64_t kHiddenMultiple = 8;
constexpr std::size_t kArtifactIdChars  = 32;
constexpr double kNormTolerance         = 1e-3;
// Round-to-nearest-even overflows to infinity from FLT_MAX plus half an ulp upward.
constexpr double kFp32Overflow = 0x1.ffffffp127;

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

std::string field(std::string_view path) { return std::string(kLabel) + " " + std::string(path); }

std::string number_text(double value) {
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}

bool is_artifact_id(const std::string& text) {
    return text.size() == kArtifactIdChars && std::all_of(text.begin(), text.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

// JSON numbers (integer or floating token) are rounded to the nearest FP32; values that would
// round to infinity are rejected, as is anything that is not a finite number.
std::optional<float> as_fp32(const Json& value) {
    if (!value.is_number()) return std::nullopt;
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::fabs(number) >= kFp32Overflow) return std::nullopt;
    return static_cast<float>(number);
}

float require_fp32(const Json& value, const std::string& label) {
    const auto result = as_fp32(value);
    if (!result) fail(label + " must be a finite number representable as FP32");
    return *result;
}

std::vector<float> require_direction(const Json& value, std::uint32_t hidden_size,
                                     const std::string& label) {
    if (!value.is_array()) fail(label + " must be an array of numbers");
    if (value.size() != hidden_size) {
        fail(label + " has " + std::to_string(value.size()) + " elements; hidden_size is " +
             std::to_string(hidden_size));
    }
    std::vector<float> result;
    result.reserve(hidden_size);
    for (const auto& element : value) {
        const auto parsed = as_fp32(element);
        if (!parsed) {
            fail(label + "[" + std::to_string(result.size()) +
                 "] must be a finite number representable as FP32");
        }
        result.push_back(*parsed);
    }
    return result;
}

// The norm is measured on the FP32 values the runtime will use; nothing is renormalized.
void require_unit_norm(const std::vector<float>& values, const std::string& label,
                       bool allow_zero) {
    if (allow_zero &&
        std::all_of(values.begin(), values.end(), [](float x) { return x == 0.0F; })) {
        return;
    }
    double sum = 0.0;
    for (const float x : values) sum += static_cast<double>(x) * static_cast<double>(x);
    const double norm = std::sqrt(sum);
    if (std::fabs(norm - 1.0) <= kNormTolerance) return;
    fail(label + " must have L2 norm 1 +/- 1e-3" + (allow_zero ? " or be exactly zero" : "") +
         ", got " + number_text(norm));
}

LayerEdit parse_layer(const Json& item, std::size_t position, std::uint32_t hidden_size,
                      std::uint32_t layer_count, const LayerEdit* previous) {
    const std::string entry = field("layers[" + std::to_string(position) + "]");
    artifact::require_members(item,
                              {"index", "refusal_direction", "refusal_reference",
                               "disclaimer_direction", "disclaimer_reference"},
                              {}, entry);

    LayerEdit edit;
    const auto index = artifact::require_u64(item.at("index"), entry + ".index");
    if (index >= layer_count) {
        fail(entry + ".index " + std::to_string(index) + " must be below num_hidden_layers " +
             std::to_string(layer_count));
    }
    if (previous && index <= previous->layer) {
        fail(entry + ".index " + std::to_string(index) +
             " must be greater than the previous index " + std::to_string(previous->layer));
    }
    edit.layer = static_cast<std::uint32_t>(index);

    edit.refusal =
        require_direction(item.at("refusal_direction"), hidden_size, entry + ".refusal_direction");
    require_unit_norm(edit.refusal, entry + ".refusal_direction", false);
    edit.refusal_reference =
        require_fp32(item.at("refusal_reference"), entry + ".refusal_reference");

    edit.disclaimer = require_direction(item.at("disclaimer_direction"), hidden_size,
                                        entry + ".disclaimer_direction");
    require_unit_norm(edit.disclaimer, entry + ".disclaimer_direction", true);
    edit.disclaimer_reference =
        require_fp32(item.at("disclaimer_reference"), entry + ".disclaimer_reference");
    return edit;
}

Profile parse(std::string_view json_text) {
    // Rejects malformed text, trailing text and duplicate members at any depth.
    const Json root = artifact::parse_json(json_text, kLabel);

    // Checked before the member set: a later schema may rename or add members.
    if (const auto version = root.find("schema_version"); version != root.end()) {
        const auto number = artifact::require_u64(*version, field("schema_version"));
        if (number != kSchemaVersion) {
            fail(field("schema_version") + " must be " + std::to_string(kSchemaVersion) + ", got " +
                 std::to_string(number));
        }
    }
    artifact::require_members(
        root, {"schema_version", "artifact_id", "hidden_size", "num_hidden_layers", "layers"},
        {"provenance"}, kLabel);

    Profile profile;
    const Json& id = root.at("artifact_id");
    if (!id.is_string() || !is_artifact_id(id.get_ref<const std::string&>())) {
        fail(field("artifact_id") + " must be 32 lowercase hexadecimal characters");
    }
    profile.artifact_id = id.get<std::string>();

    const auto hidden_size =
        artifact::require_u64(root.at("hidden_size"), field("hidden_size"), true);
    if (hidden_size > kMaxHiddenSize || hidden_size % kHiddenMultiple != 0) {
        fail(field("hidden_size") + " must be a multiple of " + std::to_string(kHiddenMultiple) +
             " no greater than " + std::to_string(kMaxHiddenSize) + ", got " +
             std::to_string(hidden_size));
    }
    profile.hidden_size = static_cast<std::uint32_t>(hidden_size);

    const auto layer_count =
        artifact::require_u64(root.at("num_hidden_layers"), field("num_hidden_layers"), true);
    if (layer_count > std::numeric_limits<std::uint32_t>::max()) {
        fail(field("num_hidden_layers") + " is too large: " + std::to_string(layer_count));
    }
    profile.layer_count = static_cast<std::uint32_t>(layer_count);

    if (const auto provenance = root.find("provenance");
        provenance != root.end() && !provenance->is_object()) {
        fail(field("provenance") + " must be a JSON object");
    }

    const Json& layers = root.at("layers");
    if (!layers.is_array() || layers.empty()) fail(field("layers") + " must be a non-empty array");
    profile.layers.reserve(layers.size());
    for (std::size_t position = 0; position < layers.size(); ++position) {
        profile.layers.push_back(
            parse_layer(layers[position], position, profile.hidden_size, profile.layer_count,
                        profile.layers.empty() ? nullptr : &profile.layers.back()));
    }
    return profile;
}

} // namespace

Profile parse_profile(std::string_view json_text) {
    try {
        return parse(json_text);
    } catch (const artifact::ArtifactError& error) { throw std::invalid_argument(error.what()); }
}

Profile read_profile(const std::filesystem::path& path) {
    const std::string name = "residual profile '" + path.string() + "'";
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        fail(name + " is missing or not a regular file");
    }
    const auto bytes = std::filesystem::file_size(path, error);
    if (error) fail("cannot stat " + name + ": " + error.message());
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("cannot open " + name);
    std::string text(bytes, '\0');
    file.read(text.data(), static_cast<std::streamsize>(bytes));
    if (file.gcount() != static_cast<std::streamsize>(bytes)) fail("cannot read " + name);
    try {
        return parse_profile(text);
    } catch (const std::invalid_argument& failure) {
        throw std::invalid_argument(path.string() + ": " + failure.what());
    }
}

void require_binding(const Profile& profile, std::string_view artifact_id,
                     std::uint32_t hidden_size, std::uint32_t layer_count) {
    if (profile.artifact_id != artifact_id) {
        fail(field("artifact_id") + " " + profile.artifact_id +
             " does not match the model artifact " + std::string(artifact_id));
    }
    if (profile.hidden_size != hidden_size) {
        fail(field("hidden_size") + " " + std::to_string(profile.hidden_size) +
             " does not match the model hidden size " + std::to_string(hidden_size));
    }
    if (profile.layer_count != layer_count) {
        fail(field("num_hidden_layers") + " " + std::to_string(profile.layer_count) +
             " does not match the model layer count " + std::to_string(layer_count));
    }
}

} // namespace ninfer::models::residual
