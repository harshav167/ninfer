#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

struct TempDirectory {
    std::filesystem::path path;

    TempDirectory()
        : path(std::filesystem::temp_directory_path() /
               ("ninfer_residual_profile_real_" + std::to_string(::getpid()))) {
        std::filesystem::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

ninfer::EngineOptions scoring_options(const char* artifact,
                                      const std::filesystem::path& profile = {}) {
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.purpose               = ninfer::EnginePurpose::CausalScoring;
    options.max_context           = 1024;
    options.kv_cache              = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.residual_profile_path = profile;
    return options;
}

struct Geometry {
    std::string artifact_id;
    std::uint32_t hidden = 0;
    std::uint32_t layers = 0;
};

// One edited layer; the refusal direction is the first channel and the disclaimer is absent.
Json profile_json(const Geometry& geometry, std::uint32_t layer, double refusal_reference) {
    std::vector<float> refusal(geometry.hidden, 0.0F), disclaimer(geometry.hidden, 0.0F);
    refusal[0] = 1.0F;
    return {{"schema_version", 1},
            {"artifact_id", geometry.artifact_id},
            {"hidden_size", geometry.hidden},
            {"num_hidden_layers", geometry.layers},
            {"layers", Json::array({{{"index", layer},
                                     {"refusal_direction", refusal},
                                     {"refusal_reference", refusal_reference},
                                     {"disclaimer_direction", disclaimer},
                                     {"disclaimer_reference", 0.0}}})}};
}

std::filesystem::path write_profile(const TempDirectory& directory, const char* name,
                                    const Json& body) {
    const auto path = directory.path / name;
    std::ofstream(path) << body.dump();
    return path;
}

bool rejected(const char* artifact, const std::filesystem::path& profile, const char* what) {
    try {
        ninfer::Engine engine(scoring_options(artifact, profile));
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).empty()) {
            std::cerr << "FAIL: " << what << " was rejected without a message\n";
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << what << " threw a different exception type: " << error.what()
                  << '\n';
        return false;
    }
    std::cerr << "FAIL: " << what << " was accepted\n";
    return false;
}

std::vector<float> score(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& tokens) {
    return engine.score_tokens(tokens, 1);
}

double max_difference(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(double(a[i]) - double(b[i])));
    return worst;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    TempDirectory directory;

    Geometry geometry;
    std::vector<ninfer::TokenId> tokens;
    std::vector<float> baseline, repeated;
    {
        ninfer::Engine engine(scoring_options(artifact));
        const auto& description = engine.description();
        geometry                = {description.artifact_id, description.hidden_size,
                                   description.num_hidden_layers};
        if (!check(geometry.artifact_id.size() == 32 && geometry.hidden > 0 && geometry.layers > 1,
                   "Engine description lacks artifact identity or geometry") ||
            !check(!description.residual_intervention.enabled,
                   "an Engine without a profile reports an intervention"))
            return 1;
        std::string text;
        const std::string paragraph =
            "NInfer applies a startup profile to every layer output of one resident model. "
            "The profile is bound to its artifact and cannot change while the Engine runs.\n";
        while (tokens.size() < 257) {
            text += paragraph;
            tokens = engine.tokenize_text(text);
        }
        tokens.resize(257);
        baseline = score(engine, tokens);
        repeated = score(engine, tokens);
    }
    if (!check(baseline.size() == 256 && repeated.size() == 256, "unexpected scoring shape"))
        return 1;
    const double noise = max_difference(baseline, repeated);
    const auto layer   = geometry.layers / 2;

    // Startup validation fails before any weight is read.
    {
        auto wrong_artifact               = profile_json(geometry, layer, 0.0);
        auto& id                          = wrong_artifact["artifact_id"].get_ref<std::string&>();
        id[0]                             = id[0] == '0' ? '1' : '0';
        auto wrong_hidden                 = profile_json(geometry, layer, 0.0);
        wrong_hidden["hidden_size"]       = geometry.hidden + 8;
        auto wrong_layers                 = profile_json(geometry, layer, 0.0);
        wrong_layers["num_hidden_layers"] = geometry.layers + 1;
        auto out_of_range                 = profile_json(geometry, geometry.layers, 0.0);
        auto not_unit                     = profile_json(geometry, layer, 0.0);
        not_unit["layers"][0]["refusal_direction"][0] = 2.0;
        auto unknown_key                              = profile_json(geometry, layer, 0.0);
        unknown_key["extra"]                          = 1;
        auto bad_schema                               = profile_json(geometry, layer, 0.0);
        bad_schema["schema_version"]                  = 2;
        if (!rejected(artifact, directory.path / "missing.json", "a missing profile") ||
            !rejected(artifact, write_profile(directory, "artifact.json", wrong_artifact),
                      "a profile for another artifact") ||
            !rejected(artifact, write_profile(directory, "hidden.json", wrong_hidden),
                      "a profile with another hidden width") ||
            !rejected(artifact, write_profile(directory, "layers.json", wrong_layers),
                      "a profile with another layer count") ||
            !rejected(artifact, write_profile(directory, "range.json", out_of_range),
                      "a layer outside the model") ||
            !rejected(artifact, write_profile(directory, "unit.json", not_unit),
                      "a non-normalized direction") ||
            !rejected(artifact, write_profile(directory, "key.json", unknown_key),
                      "an unknown key") ||
            !rejected(artifact, write_profile(directory, "schema.json", bad_schema),
                      "an unsupported schema version"))
            return 1;
    }

    // A clip that can never activate (reference above every representable residual) must leave
    // the model unchanged: the Op rewrites each BF16 word exactly.
    {
        const auto noop_reference = std::numeric_limits<float>::max();
        const auto path =
            write_profile(directory, "noop.json", profile_json(geometry, layer, noop_reference));
        ninfer::Engine engine(scoring_options(artifact, path));
        const auto& intervention = engine.description().residual_intervention;
        if (!check(intervention.enabled && intervention.profile_path == path &&
                       intervention.layers == std::vector<std::uint32_t>{layer},
                   "the Engine description does not report the selected profile"))
            return 1;
        const auto scores = score(engine, tokens);
        if (!check(max_difference(scores, baseline) <= noise + 1.0e-6,
                   "an inactive clip changed the scores"))
            return 1;
        // Calibration observes the unmodified model only.
        auto prompt = engine.prepare_tokens(tokens, false);
        const std::vector<std::uint32_t> layers{layer};
        bool refused = false;
        try {
            (void)engine.capture_residuals(std::move(prompt), {}, layers);
        } catch (const std::logic_error&) { refused = true; }
        if (!check(refused, "capture_residuals accepted an Engine with a profile")) return 1;
    }

    // Clamping the first channel far below its natural range must change the distribution.
    {
        const auto path =
            write_profile(directory, "active.json", profile_json(geometry, layer, -1.0e4));
        ninfer::Engine engine(scoring_options(artifact, path));
        const auto scores = score(engine, tokens);
        for (const float value : scores)
            if (!check(std::isfinite(value) && value <= 0.0F,
                       "an active edit produced an invalid log probability"))
                return 1;
        if (!check(max_difference(scores, baseline) > 0.05 + noise,
                   "an active edit did not change the scores"))
            return 1;
    }
    std::cout << "OK residual_profile_real\n";
    return 0;
}
