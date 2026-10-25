#pragma once

// Host-only logic of ninfer-residual-capture: command line, calibration dataset, layer
// selection and the on-disk capture format. Nothing here touches the Engine, so it is
// testable without a model.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::residual_capture {

inline constexpr std::uint32_t kDefaultMaxContext   = 8192;
inline constexpr std::uint32_t kDefaultPrefillChunk = 1024;

struct Options {
    bool help_requested = false;
    std::filesystem::path model;
    std::filesystem::path dataset;
    std::filesystem::path output;
    // "all" or an ascending list such as "21-42,50".
    std::string layers = "all";
    // Empty selects the artifact template.
    std::filesystem::path chat_template;
    int device                  = 0;
    std::uint32_t max_context   = kDefaultMaxContext;
    std::uint32_t prefill_chunk = kDefaultPrefillChunk;
    // "fp8" (production launcher) or "bf16"; selects the Main KV cache storage.
    std::string kv_dtype = "fp8";
};

[[nodiscard]] std::string usage_text();

// Throws std::invalid_argument for unknown flags, repeated flags, missing values or missing
// required options. `--help` returns immediately with help_requested set.
[[nodiscard]] Options parse_options(std::span<const char* const> arguments);

struct LayerRange {
    std::uint32_t first = 0;
    std::uint32_t last  = 0;
};

// "all", or comma-separated indices and inclusive ranges that are strictly ascending overall.
struct LayerSelection {
    bool all = false;
    std::vector<LayerRange> ranges;
};

[[nodiscard]] LayerSelection parse_layer_selection(std::string_view text);

// Expands the selection for a model with layer_count layers; every index must be below it.
[[nodiscard]] std::vector<std::uint32_t> resolve_layers(const LayerSelection& selection,
                                                        std::uint32_t layer_count);

struct Message {
    std::string role;
    std::string content;
};

struct Sample {
    std::string id;
    std::string label;
    std::vector<Message> messages;
    // Tokenized independently of the chat template and appended after the generation prefix.
    std::string assistant_opening;
};

struct Dataset {
    std::vector<Sample> samples;
    // SHA-256 of the raw input bytes, lowercase hexadecimal.
    std::string sha256;
};

// One JSON object per line; blank lines are ignored. Throws std::invalid_argument naming the
// offending line.
[[nodiscard]] Dataset parse_dataset(std::string_view bytes);
[[nodiscard]] Dataset read_dataset(const std::filesystem::path& path);

// SHA-256 of the little-endian int32 encoding of the ids, lowercase hexadecimal.
[[nodiscard]] std::string token_ids_sha256(std::span<const std::int32_t> ids);

struct SampleRecord {
    std::string id;
    std::string label;
    // Whole sequence, including opening tokens.
    std::uint32_t prompt_tokens  = 0;
    std::uint32_t opening_tokens = 0;
    std::string prompt_token_ids_sha256;
};

struct Manifest {
    std::string artifact_id;
    std::string model_name;
    std::uint32_t hidden_size       = 0;
    std::uint32_t num_hidden_layers = 0;
    std::vector<std::uint32_t> layers;
    // "artifact" or the template path as given.
    std::string chat_template;
    std::uint32_t prefill_chunk = 0;
    // "fp8" or "bf16".
    std::string kv_dtype;
    std::string dataset_sha256;
    std::vector<SampleRecord> samples;
};

// Serializes the manifest as indented JSON (schema_version 1).
[[nodiscard]] std::string manifest_json(const Manifest& manifest);

// Writes OUTDIR/residuals.f32 and OUTDIR/manifest.json. The directory must not exist or must be
// empty. Both files are written under temporary names; residuals.f32 is published first and
// manifest.json last, so a manifest only ever describes a complete residual file. The writer
// owns only the files it created. A writer destroyed before finish(), and a finish() whose
// publication fails, remove those files (temporaries and a published residuals.f32) and leave
// every other entry of the directory alone; only a completed publication keeps them.
class CaptureWriter {
public:
    CaptureWriter(const std::filesystem::path& directory, std::uint32_t values_per_sample);
    ~CaptureWriter();
    CaptureWriter(const CaptureWriter&)            = delete;
    CaptureWriter& operator=(const CaptureWriter&) = delete;

    // Rejects an existing non-empty directory without creating anything.
    static void require_available(const std::filesystem::path& directory);

    // Appends one sample row ([layer slot][hidden]) as little-endian float32.
    void append(std::span<const float> values);

    [[nodiscard]] std::uint64_t samples() const noexcept { return samples_; }

    // Publishes residuals.f32, then manifest.json. Throws std::invalid_argument, leaving the
    // writer usable, if the manifest does not describe the rows written. A publication failure
    // removes what this writer created and ends the writer; later calls throw std::logic_error.
    void finish(const Manifest& manifest);

private:
    enum class State { Open, Published, Abandoned };

    // Closes the stream and removes only outputs whose ownership has not been handed over.
    void discard() noexcept;

    std::filesystem::path residuals_temporary_;
    std::filesystem::path residuals_final_;
    std::filesystem::path manifest_temporary_;
    std::filesystem::path manifest_final_;
    std::uint32_t values_per_sample_;
    std::ofstream residuals_;
    std::uint64_t samples_    = 0;
    bool owns_residuals_      = false;
    bool residuals_published_ = false;
    bool owns_manifest_       = false;
    State state_              = State::Open;
};

} // namespace ninfer::residual_capture
