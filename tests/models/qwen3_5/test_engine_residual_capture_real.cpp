// Native residual capture through a CausalScoring Engine: shape, determinism, layer selection,
// forced suffixes, multiple prefill windows, argument rejection, and coexistence with scoring.

#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace nf = ninfer;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

nf::EngineOptions scoring_options(const char* artifact, std::uint32_t prefill_chunk) {
    nf::EngineOptions options;
    options.artifact_path  = artifact;
    options.purpose        = nf::EnginePurpose::CausalScoring;
    options.max_context    = 1024;
    options.prefill_chunk  = prefill_chunk;
    options.kv_cache       = nf::KvCacheStorage::Fp8E4M3Row256;
    options.use_cuda_graph = false;
    return options;
}

std::vector<nf::TokenId> text_tokens(const nf::Engine& engine, std::size_t count) {
    const std::string paragraph =
        "Residual capture replays one prompt through the resident model and reads the last "
        "token's state after every layer. Nothing is sampled and nothing is modified.\n";
    std::string text;
    std::vector<nf::TokenId> tokens;
    while (tokens.size() < count) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(count);
    return tokens;
}

std::vector<std::uint32_t> iota_layers(std::uint32_t count) {
    std::vector<std::uint32_t> layers(count);
    for (std::uint32_t layer = 0; layer < count; ++layer) { layers[layer] = layer; }
    return layers;
}

nf::ResidualCapture capture(nf::Engine& engine, const std::vector<nf::TokenId>& tokens,
                            std::span<const nf::TokenId> suffix,
                            std::span<const std::uint32_t> layers) {
    return engine.capture_residuals(engine.prepare_tokens(tokens, false), suffix, layers);
}

bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// The residual is BF16: the widened FP32 word has an all-zero low half.
bool is_bf16_widening(const std::vector<float>& values) {
    for (const float value : values) {
        std::uint32_t word = 0;
        std::memcpy(&word, &value, sizeof(word));
        if ((word & 0xffffU) != 0 || !std::isfinite(value)) { return false; }
    }
    return true;
}

std::span<const float> row(const nf::ResidualCapture& result, std::size_t slot) {
    return {result.values.data() + slot * result.hidden_size, result.hidden_size};
}

template <class Function>
void require_invalid_argument(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        require(std::string(error.what()).size() > 0, what + " was rejected without a message");
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(what + " threw another exception type: " + error.what());
    }
    throw std::runtime_error(what + " was accepted");
}

struct Statistics {
    std::uint64_t completed = 0, failed = 0, active = 0, score_tokens = 0;
};

Statistics statistics(const nf::Engine& engine) {
    const nf::EngineSnapshot snapshot = engine.snapshot();
    return {snapshot.scoring.completed_jobs, snapshot.scoring.failed_jobs,
            snapshot.scoring.active_jobs,
            snapshot.executions[static_cast<std::size_t>(nf::ExecutionKind::Score)].input_tokens};
}

constexpr std::size_t kShortTokens = 48;
constexpr std::size_t kLongTokens  = 333; // 128 + 128 + 77 at chunk 128; one window at 1024.

int run(const char* artifact) {
    nf::ResidualCapture long_reference;
    std::vector<std::uint32_t> all_layers;
    std::vector<nf::TokenId> long_tokens;

    {
        nf::Engine engine(scoring_options(artifact, 1024));
        const nf::EngineDescription& description = engine.description();
        require(description.hidden_size > 0 && description.num_hidden_layers > 1,
                "the Engine description lacks model geometry");
        require(!description.residual_intervention.enabled,
                "a capture Engine must observe the unmodified model");
        const std::uint32_t layer_count             = description.num_hidden_layers;
        all_layers                                  = iota_layers(layer_count);
        const std::vector<nf::TokenId> short_tokens = text_tokens(engine, kShortTokens);
        std::uint64_t successes                     = 0;
        std::uint64_t session_rejections            = 0;

        // Scoring baseline on the same Engine, before any capture.
        const std::vector<float> score_a = engine.score_tokens(short_tokens, 1);
        const std::vector<float> score_b = engine.score_tokens(short_tokens, 1);
        successes += 2;
        double score_noise = 0.0;
        require(score_a.size() == kShortTokens - 1 && score_b.size() == score_a.size(),
                "unexpected scoring shape");
        for (std::size_t i = 0; i < score_a.size(); ++i) {
            score_noise = std::max(score_noise, std::abs(double(score_a[i]) - double(score_b[i])));
        }

        // Shape, finiteness, BF16 widening and a live probe for every layer.
        const Statistics before        = statistics(engine);
        const nf::ResidualCapture full = capture(engine, short_tokens, {}, all_layers);
        ++successes;
        const Statistics after = statistics(engine);
        require(full.prompt_token_ids == short_tokens, "capture must report the observed sequence");
        require(full.layers == all_layers, "capture must echo the requested layers");
        require(full.hidden_size == description.hidden_size, "capture hidden size");
        require(full.values.size() == std::size_t(layer_count) * description.hidden_size,
                "capture must hold [layers][hidden] values");
        require(is_bf16_widening(full.values), "capture values must be finite BF16 widenings");
        for (std::size_t slot = 0; slot < layer_count; ++slot) {
            double norm = 0.0;
            for (const float value : row(full, slot)) { norm += double(value) * double(value); }
            require(norm > 0.0, "layer " + std::to_string(slot) + " was never written");
        }
        require(std::memcmp(row(full, 0).data(), row(full, 1).data(),
                            std::size_t(description.hidden_size) * sizeof(float)) != 0,
                "adjacent layers must observe different residual states");
        // All N tokens are fed through the model; scoring feeds N-1.
        require(after.completed == before.completed + 1 && after.active == 0 &&
                    after.score_tokens == before.score_tokens + kShortTokens,
                "a capture must process every prompt token as one completed job");

        // Determinism.
        const nf::ResidualCapture again = capture(engine, short_tokens, {}, all_layers);
        ++successes;
        require(bit_identical(full.values, again.values), "identical captures differ");

        // A layer subset equals the matching slice of the all-layers capture.
        const std::vector<std::uint32_t> subset{0, layer_count / 3, layer_count - 1};
        const nf::ResidualCapture partial = capture(engine, short_tokens, {}, subset);
        ++successes;
        require(partial.layers == subset &&
                    partial.values.size() == subset.size() * description.hidden_size,
                "subset capture shape");
        for (std::size_t slot = 0; slot < subset.size(); ++slot) {
            const auto expected = row(full, subset[slot]);
            const auto actual   = row(partial, slot);
            require(
                std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0,
                "subset layer " + std::to_string(subset[slot]) + " differs from the full capture");
        }
        const std::vector<std::uint32_t> single{layer_count - 1};
        const nf::ResidualCapture last = capture(engine, short_tokens, {}, single);
        ++successes;
        require(std::memcmp(last.values.data(), row(full, layer_count - 1).data(),
                            last.values.size() * sizeof(float)) == 0,
                "the last layer alone differs from the full capture");

        // A forced suffix equals the same tokens supplied by the prompt.
        const std::vector<nf::TokenId> head(short_tokens.begin(), short_tokens.end() - 5);
        const std::vector<nf::TokenId> tail(short_tokens.end() - 5, short_tokens.end());
        const nf::ResidualCapture forced = capture(engine, head, tail, all_layers);
        ++successes;
        require(forced.prompt_token_ids == short_tokens,
                "the observed sequence must be prompt + forced suffix");
        require(bit_identical(forced.values, full.values),
                "a forced suffix differs from the same tokens in the prompt");
        const std::vector<nf::TokenId> one{short_tokens.back()};
        const std::vector<nf::TokenId> body(short_tokens.begin(), short_tokens.end() - 1);
        const nf::ResidualCapture forced_one = capture(engine, body, one, all_layers);
        ++successes;
        require(bit_identical(forced_one.values, full.values),
                "a one-token forced suffix differs from the same token in the prompt");
        // Observing one token earlier is a different position.
        const nf::ResidualCapture shorter = capture(engine, body, {}, all_layers);
        ++successes;
        require(!bit_identical(shorter.values, full.values),
                "the observed position must be the last token of the sequence");

        // The production path: chat template with thinking, then a tokenized opening.
        {
            nf::PromptInput input;
            input.options.enable_thinking = true;
            nf::ChatMessage message;
            message.role = nf::ChatRole::User;
            nf::MessagePart part;
            part.text = "Say hello.";
            message.parts.push_back(std::move(part));
            input.messages.push_back(std::move(message));
            nf::PreparedPrompt prompt              = engine.prepare(std::move(input));
            const std::uint32_t prompt_tokens      = prompt.summary().prompt_tokens;
            const std::vector<nf::TokenId> opening = engine.tokenize_text("Sure");
            require(!opening.empty(), "the opening tokenized to nothing");
            const nf::ResidualCapture chat =
                engine.capture_residuals(std::move(prompt), opening, subset);
            ++successes;
            require(chat.prompt_token_ids.size() == prompt_tokens + opening.size() &&
                        std::equal(opening.begin(), opening.end(),
                                   chat.prompt_token_ids.end() - opening.size()),
                    "the chat capture must end with the forced opening");
            require(is_bf16_widening(chat.values), "chat capture values");
        }

        // Rejections leave the Engine usable.
        {
            const auto reject = [&](std::span<const nf::TokenId> suffix,
                                    std::span<const std::uint32_t> layers,
                                    const std::string& what) {
                require_invalid_argument(
                    [&] {
                        (void)engine.capture_residuals(engine.prepare_tokens(short_tokens, false),
                                                       suffix, layers);
                    },
                    what);
            };
            const std::vector<std::uint32_t> empty_layers;
            const std::vector<std::uint32_t> unsorted{5, 3};
            const std::vector<std::uint32_t> repeated{3, 3};
            const std::vector<std::uint32_t> beyond{layer_count};
            const std::vector<std::uint32_t> enormous{std::numeric_limits<std::uint32_t>::max()};
            const std::vector<std::uint32_t> valid{1};
            reject({}, empty_layers, "empty layers");
            reject({}, unsorted, "unsorted layers");
            reject({}, repeated, "repeated layers");
            reject({}, beyond, "a layer equal to the layer count");
            ++session_rejections;
            reject({}, enormous, "a huge layer index");
            ++session_rejections;
            const std::vector<nf::TokenId> negative{-1};
            const std::vector<nf::TokenId> outside{std::numeric_limits<nf::TokenId>::max()};
            reject(negative, valid, "a negative forced token");
            ++session_rejections;
            reject(outside, valid, "a forced token outside the vocabulary");
            ++session_rejections;

            require_invalid_argument(
                [&] { (void)engine.capture_residuals(nf::PreparedPrompt{}, {}, valid); },
                "an empty prepared prompt");
            const std::vector<nf::TokenId> full_window = text_tokens(engine, 1024);
            const std::vector<nf::TokenId> extra{full_window.front()};
            require_invalid_argument(
                [&] {
                    (void)engine.capture_residuals(engine.prepare_tokens(full_window, false), extra,
                                                   valid);
                },
                "a sequence longer than max_context");

            // A text-only Engine cannot even prepare media, so no media prompt reaches capture.
            nf::PromptInput media;
            nf::ChatMessage message;
            nf::MessagePart text;
            text.text = "describe";
            nf::MessagePart image;
            image.kind        = nf::MessagePartKind::Media;
            image.media.kind  = nf::MediaKind::Image;
            image.media.bytes = {0x89, 0x50, 0x4e, 0x47};
            message.parts.push_back(std::move(text));
            message.parts.push_back(std::move(image));
            media.messages.push_back(std::move(message));
            require_invalid_argument([&] { (void)engine.prepare(std::move(media)); },
                                     "a media prompt on a text-only Engine");
        }
        require(engine.is_available(), "a rejected capture must not invalidate the Engine");

        // Scoring still works on the same Engine and is unchanged by the captures.
        const Statistics scored_before   = statistics(engine);
        const std::vector<float> score_c = engine.score_tokens(short_tokens, 1);
        ++successes;
        const Statistics scored_after = statistics(engine);
        require(score_c.size() == score_a.size(), "scoring shape after capture");
        double drift = 0.0;
        for (std::size_t i = 0; i < score_c.size(); ++i) {
            require(std::isfinite(score_c[i]) && score_c[i] <= 0.0F,
                    "scoring returned an invalid log probability after capture");
            drift = std::max(drift, std::abs(double(score_c[i]) - double(score_a[i])));
        }
        require(drift <= score_noise + 1.0e-6, "capture changed later scoring results");
        require(scored_after.score_tokens == scored_before.score_tokens + kShortTokens - 1,
                "scoring still feeds all but the last token");

        const nf::EngineSnapshot snapshot = engine.snapshot();
        require(snapshot.scoring.completed_jobs == successes &&
                    snapshot.scoring.failed_jobs == session_rejections &&
                    snapshot.scoring.active_jobs == 0 && snapshot.runtime.generated_tokens == 0,
                "scoring statistics must count capture jobs as scoring jobs only");

        // Capture a longer prompt and retain its exact tokens for the multi-window case.
        long_tokens    = text_tokens(engine, kLongTokens);
        long_reference = capture(engine, long_tokens, {}, all_layers);
        require(long_reference.prompt_token_ids == long_tokens &&
                    is_bf16_widening(long_reference.values),
                "long capture shape");

        // A moved-from Engine reports its state instead of crashing.
        nf::Engine moved(std::move(engine));
        bool moved_from = false;
        try {
            (void)engine.capture_residuals(nf::PreparedPrompt{}, {}, all_layers);
        } catch (const std::logic_error&) { moved_from = true; }
        require(moved_from, "capture on a moved-from Engine must throw std::logic_error");
        // The moved-to Engine still captures.
        const nf::ResidualCapture after_move = capture(moved, short_tokens, {}, all_layers);
        require(bit_identical(after_move.values, full.values),
                "the moved-to Engine changed the captured values");
    }

    // Capture across several prefill windows with the same exact input sequence.
    {
        nf::Engine engine(scoring_options(artifact, 128));
        require(engine.description().prefill_chunk == 128, "prefill chunk was not applied");
        const std::vector<nf::TokenId> tokens = text_tokens(engine, kLongTokens);
        require(tokens == long_tokens, "tokenization differs between Engines");
        const nf::ResidualCapture chunked = capture(engine, tokens, {}, all_layers);
        require(chunked.prompt_token_ids == long_tokens && is_bf16_widening(chunked.values),
                "chunked capture shape");
        const nf::ResidualCapture repeated = capture(engine, tokens, {}, all_layers);
        require(bit_identical(chunked.values, repeated.values),
                "chunked capture is not deterministic");
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        if (run(artifact) != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK residual_capture_real\n";
    return 0;
}
