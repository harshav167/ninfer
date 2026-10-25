#include "capture.h"

#include "ninfer/engine.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace capture = ninfer::residual_capture;
using Clock       = std::chrono::steady_clock;

static_assert(std::is_same_v<ninfer::TokenId, std::int32_t>,
              "the token digest hashes little-endian int32 ids");

ninfer::ChatRole chat_role(const std::string& role) {
    if (role == "system") { return ninfer::ChatRole::System; }
    if (role == "developer") { return ninfer::ChatRole::Developer; }
    if (role == "user") { return ninfer::ChatRole::User; }
    if (role == "assistant") { return ninfer::ChatRole::Assistant; }
    if (role == "tool") { return ninfer::ChatRole::Tool; }
    throw std::invalid_argument("unsupported role '" + role + "'");
}

// Chat template with thinking enabled and a new assistant turn: the rendered sequence ends with
// the assistant generation prefix.
ninfer::PromptInput prompt_input(const capture::Sample& sample) {
    ninfer::PromptInput input;
    input.options.continuation    = ninfer::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking = true;
    for (const capture::Message& message : sample.messages) {
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Text;
        part.text = message.content;
        ninfer::ChatMessage chat;
        chat.role = chat_role(message.role);
        chat.parts.push_back(std::move(part));
        input.messages.push_back(std::move(chat));
    }
    return input;
}

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

int run(const capture::Options& options) {
    const Clock::time_point started = Clock::now();
    // Fail on everything checkable without the model before it is loaded.
    capture::CaptureWriter::require_available(options.output);
    const capture::LayerSelection selection = capture::parse_layer_selection(options.layers);
    const capture::Dataset dataset          = capture::read_dataset(options.dataset);
    std::cerr << "dataset: " << dataset.samples.size() << " samples, sha256 " << dataset.sha256
              << '\n';

    ninfer::EngineOptions engine_options;
    engine_options.artifact_path      = options.model;
    engine_options.chat_template_path = options.chat_template;
    engine_options.purpose            = ninfer::EnginePurpose::CausalScoring;
    engine_options.device             = options.device;
    engine_options.max_context        = options.max_context;
    engine_options.prefill_chunk      = options.prefill_chunk;
    engine_options.use_cuda_graph     = false;
    engine_options.kv_cache = options.kv_dtype == "bf16" ? ninfer::KvCacheStorage::BFloat16
                                                         : ninfer::KvCacheStorage::Fp8E4M3Row256;
    std::cerr << "loading " << options.model.string() << '\n';
    ninfer::Engine engine(std::move(engine_options));
    const ninfer::EngineDescription& description = engine.description();
    if (description.artifact_id.empty() || description.hidden_size == 0 ||
        description.num_hidden_layers == 0) {
        throw std::runtime_error("the Engine did not report the artifact identity and geometry");
    }
    const std::vector<std::uint32_t> layers =
        capture::resolve_layers(selection, description.num_hidden_layers);
    std::cerr << "model " << description.model_name << " | artifact " << description.artifact_id
              << " | " << description.num_hidden_layers << " layers, hidden "
              << description.hidden_size << " | capturing " << layers.size() << " layers\n";

    const std::uint64_t row_values = std::uint64_t(layers.size()) * description.hidden_size;
    if (row_values > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("one capture row exceeds the supported size");
    }
    capture::CaptureWriter writer(options.output, static_cast<std::uint32_t>(row_values));

    capture::Manifest manifest;
    manifest.artifact_id       = description.artifact_id;
    manifest.model_name        = description.model_name;
    manifest.hidden_size       = description.hidden_size;
    manifest.num_hidden_layers = description.num_hidden_layers;
    manifest.layers            = layers;
    manifest.chat_template =
        options.chat_template.empty() ? std::string("artifact") : options.chat_template.string();
    manifest.prefill_chunk  = description.prefill_chunk;
    manifest.kv_dtype       = options.kv_dtype;
    manifest.dataset_sha256 = dataset.sha256;

    const std::size_t total = dataset.samples.size();
    for (std::size_t index = 0; index < total; ++index) {
        const capture::Sample& sample          = dataset.samples[index];
        const Clock::time_point sample_started = Clock::now();
        try {
            ninfer::PreparedPrompt prompt       = engine.prepare(prompt_input(sample));
            const ninfer::PromptSummary summary = prompt.summary();
            if (summary.has_media) { throw std::runtime_error("the rendered prompt has media"); }
            if (!summary.starts_in_reasoning) {
                throw std::runtime_error(
                    "the rendered prompt does not end inside an open reasoning block; check "
                    "--chat-template");
            }
            std::vector<ninfer::TokenId> opening;
            if (!sample.assistant_opening.empty()) {
                opening = engine.tokenize_text(sample.assistant_opening);
                if (opening.empty()) {
                    throw std::runtime_error("assistant_opening tokenized to no tokens");
                }
            }
            const std::size_t expected = std::size_t(summary.prompt_tokens) + opening.size();
            const ninfer::ResidualCapture result =
                engine.capture_residuals(std::move(prompt), opening, layers);
            if (result.layers != layers || result.hidden_size != description.hidden_size ||
                result.values.size() != row_values || result.prompt_token_ids.size() != expected) {
                throw std::runtime_error("the Engine returned a capture with the wrong shape");
            }
            writer.append(result.values);

            capture::SampleRecord record;
            record.id                      = sample.id;
            record.label                   = sample.label;
            record.prompt_tokens           = static_cast<std::uint32_t>(expected);
            record.opening_tokens          = static_cast<std::uint32_t>(opening.size());
            record.prompt_token_ids_sha256 = capture::token_ids_sha256(result.prompt_token_ids);
            manifest.samples.push_back(std::move(record));
            std::cerr << '[' << index + 1 << '/' << total << "] " << sample.id << " | " << expected
                      << " tokens (" << opening.size() << " opening) | "
                      << seconds_since(sample_started) << " s\n";
        } catch (const std::exception& error) {
            throw std::runtime_error("sample '" + sample.id + "' failed: " + error.what());
        }
    }
    writer.finish(manifest);
    std::cerr << "captured " << total << " samples x " << layers.size() << " layers x "
              << description.hidden_size << " in " << seconds_since(started) << " s -> "
              << options.output.string() << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    capture::Options options;
    try {
        options = capture::parse_options(
            std::span<const char* const>(argv + 1, static_cast<std::size_t>(argc - 1)));
    } catch (const std::exception& error) {
        std::cerr << "ninfer-residual-capture: " << error.what() << '\n' << capture::usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << capture::usage_text();
        return 0;
    }
    try {
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-residual-capture: " << error.what() << '\n';
        return 1;
    }
}
