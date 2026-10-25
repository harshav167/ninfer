#include "artifact/framing.h"
#include "models/common/residual/profile.h"
#include <nlohmann/json.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace residual       = ninfer::models::residual;
using Json               = nlohmann::json;
constexpr const char* id = "0123456789abcdef0123456789abcdef";

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

Json valid() {
    return {
        {"schema_version", 1},
        {"artifact_id", id},
        {"hidden_size", 8},
        {"num_hidden_layers", 8},
        {"layers", Json::array({{{"index", 3},
                                 {"refusal_direction", {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                                 {"refusal_reference", -0.25},
                                 {"disclaimer_direction", {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                                 {"disclaimer_reference", 0.5}}})}};
}

void rejected(const std::string& text) {
    try {
        (void)residual::parse_profile(text);
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid profile was accepted");
}

template <class Edit>
void rejects(Edit edit) {
    auto value = valid();
    edit(value);
    rejected(value.dump());
}

template <class Action>
void binding_rejects(Action action) {
    try {
        action();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("mismatched profile binding was accepted");
}

void run() {
    const auto profile = residual::parse_profile(valid().dump());
    require(profile.artifact_id == id && profile.hidden_size == 8 && profile.layer_count == 8,
            "profile geometry changed");
    require(profile.layers.size() == 1 && profile.layers[0].layer == 3 &&
                profile.layers[0].refusal[0] == 1.0F &&
                profile.layers[0].refusal_reference == -0.25F &&
                profile.layers[0].disclaimer[1] == 1.0F &&
                profile.layers[0].disclaimer_reference == 0.5F,
            "profile numeric values changed");
    residual::require_binding(profile, id, 8, 8);
    binding_rejects([&] { residual::require_binding(profile, std::string(32, 'f'), 8, 8); });
    binding_rejects([&] { residual::require_binding(profile, id, 16, 8); });
    binding_rejects([&] { residual::require_binding(profile, id, 8, 9); });

    auto zero                                 = valid();
    zero["layers"][0]["disclaimer_direction"] = std::array<float, 8>{};
    zero["provenance"]                        = {{"method", "independent fixture"}};
    const auto parsed_zero                    = residual::parse_profile(zero.dump());
    for (float x : parsed_zero.layers[0].disclaimer)
        require(x == 0, "zero disclaimer direction changed");

    rejects([](Json& x) { x["schema_version"] = 2; });
    rejects([](Json& x) { x.erase("schema_version"); });
    rejects([](Json& x) { x["unknown"] = true; });
    rejects([](Json& x) { x["artifact_id"] = "not-an-artifact-id"; });
    rejects([](Json& x) { x["artifact_id"] = std::string(32, 'G'); });
    rejects([](Json& x) { x["hidden_size"] = 7; });
    rejects([](Json& x) { x["hidden_size"] = 16392; });
    rejects([](Json& x) { x["num_hidden_layers"] = 0; });
    rejects([](Json& x) { x["provenance"] = "not an object"; });
    rejects([](Json& x) { x["layers"] = Json::array(); });
    rejects([](Json& x) { x["layers"][0]["index"] = 8; });
    rejects([](Json& x) { x["layers"][0]["index"] = -1; });
    rejects([](Json& x) { x["layers"][0]["index"] = 1.5; });
    rejects([](Json& x) { x["layers"].push_back(x["layers"][0]); });
    rejects([](Json& x) {
        x["layers"].push_back(x["layers"][0]);
        x["layers"][1]["index"] = 2;
    });
    rejects([](Json& x) { x["layers"][0]["refusal_direction"] = std::array<float, 8>{}; });
    rejects([](Json& x) { x["layers"][0]["refusal_direction"][0] = 2; });
    rejects([](Json& x) { x["layers"][0]["refusal_direction"].erase(7); });
    rejects([](Json& x) { x["layers"][0]["disclaimer_direction"][0] = 1; });
    rejects([](Json& x) { x["layers"][0]["refusal_reference"] = nullptr; });
    rejects([](Json& x) { x["layers"][0]["refusal_reference"] = true; });
    rejects([](Json& x) { x["layers"][0]["refusal_reference"] = 1e39; });
    rejects([](Json& x) { x["layers"][0]["refusal_reference"] = -1e39; });
    auto overflow       = valid().dump();
    const auto position = overflow.find("-0.25");
    require(position != std::string::npos, "overflow fixture is malformed");
    overflow.replace(position, 5, "1e999");
    rejected(overflow);
    auto duplicate = valid().dump();
    duplicate.insert(1, "\"schema_version\":1,");
    rejected(duplicate);
    rejected(valid().dump() + " trailing");

    const auto directory =
        std::filesystem::path(NINFER_SOURCE_DIR) / ".agent-tmp/residual-profile-parser-test";
    std::filesystem::create_directories(directory);
    const auto file = directory / "profile.json";
    {
        std::ofstream output(file);
        output << valid().dump();
    }
    const auto from_file = residual::read_profile(file);
    require(from_file.layers[0].refusal_reference == -0.25F, "file profile changed");
    std::filesystem::remove(file);
    binding_rejects([&] { (void)residual::read_profile(file); });
    std::filesystem::remove(directory);
}
} // namespace

int main() {
    try {
        run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
