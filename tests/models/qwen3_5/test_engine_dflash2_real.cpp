#include "ninfer/engine.h"
#include "speculative_page_boundary.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(std::uint32_t outputs, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

void valid(const ninfer::GenerationResult& result, std::size_t outputs) {
    require(result.generated_token_ids.size() == outputs &&
                result.finish_reason == ninfer::FinishReason::OutputLimit,
            "DFlash2 did not honor the requested output budget");
    require(
        result.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
            (outputs == 1 || result.speculative.rounds + result.speculative.fallback_steps != 0),
        "generation bypassed DFlash2");
}

ninfer::PromptInput media_prompt(ninfer::MediaKind kind) {
    const std::string header = "P6\n64 64\n255\n";
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = kind;
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "pattern.ppm";
    media.media.bytes.assign(header.begin(), header.end());
    for (int i = 0; i < 64 * 64; ++i) {
        media.media.bytes.push_back(i & 255);
        media.media.bytes.push_back((i * 3) & 255);
        media.media.bytes.push_back((i * 7) & 255);
    }
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(std::move(media));
    user.parts.push_back({.kind  = ninfer::MessagePartKind::Text,
                          .text  = "Describe the pattern briefly.",
                          .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

bool secondary_cuda_device_available() {
    int device_count  = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) { return false; }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") +
                                 cudaGetErrorString(status));
    }
    return device_count >= 2;
}

void remote_page_frontier_continuation(ninfer::Engine& engine,
                                       const std::vector<ninfer::TokenId>& prompt) {
    auto page_prompt = prompt;
    require(page_prompt.size() <= 63, "page-frontier prompt exceeds its fixed prefix");
    page_prompt.insert(page_prompt.begin(), 63 - page_prompt.size(), 198);
    const auto reference = engine.generate(engine.prepare_tokens(page_prompt), request(16));
    valid(reference, 16);
    require(reference.generated_token_ids[0] != reference.generated_token_ids[1],
            "page-frontier fixture must expose a distinct second output token");

    auto stop_options = request(16, true);
    stop_options.stop.token_ids.push_back(reference.generated_token_ids[1]);
    const auto stopped = engine.generate(engine.prepare_tokens(page_prompt), stop_options);
    require(stopped.finish_reason == ninfer::FinishReason::StopToken &&
                stopped.speculative.rounds == 1 && stopped.speculative.drafted_tokens != 0 &&
                stopped.generated_token_ids.size() == 2 &&
                std::equal(stopped.generated_token_ids.begin(), stopped.generated_token_ids.end(),
                           reference.generated_token_ids.begin()),
            "page-frontier stop did not commit only the licensed prefix");

    auto continuation = page_prompt;
    continuation.insert(continuation.end(), stopped.generated_token_ids.begin(),
                        stopped.generated_token_ids.end());
    continuation.push_back(198);
    const auto reused = engine.generate(engine.prepare_tokens(continuation), request(16, true));
    const auto fresh  = engine.generate(engine.prepare_tokens(continuation), request(16));
    valid(reused, 16);
    valid(fresh, 16);
    require(reused.reused_prompt_tokens == 64 && fresh.reused_prompt_tokens == 0 &&
                reused.generated_token_ids == fresh.generated_token_ids,
            "reused page frontier leaked speculative output or diverged from fresh prefill");
}
} // namespace

// Optional K, Graph, optimized-head, B and KV codec arguments select representative integration
// routes without multiplying test binaries. The artifact supplies the actual weight
// representations.
int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        // NINFER_TEST_REMOTE_ATTENTION=1 opts into the real two-device Engine regression.
        const char* remote_flag     = std::getenv("NINFER_TEST_REMOTE_ATTENTION");
        const bool remote_attention = remote_flag != nullptr && std::string(remote_flag) == "1";
        if (remote_attention && !secondary_cuda_device_available()) {
            std::cout << "skip: a second CUDA device is unavailable\n";
            return 77;
        }
        const auto k =
            remote_attention ? 7U : (argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 15U);
        const bool graph = remote_attention ? false : (argc > 2 ? std::stoi(argv[2]) != 0 : true);
        const bool optimized =
            remote_attention ? true : (argc > 3 ? std::stoi(argv[3]) != 0 : true);
        const auto batch =
            remote_attention ? 2U : (argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 8U);
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 2304;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304 * batch);
        options.prefill_chunk   = 2304;
        options.max_concurrency = batch;
        options.context_cache.device_state_slots = argc > 7 ? std::stoul(argv[7]) : 3U;
        options.use_cuda_graph                   = graph;
        options.enable_vision = !remote_attention && argc > 6 && std::stoi(argv[6]) != 0;
        options.kv_cache      = remote_attention ? ninfer::KvCacheStorage::Fp8E4M3Row256
                                                 : (argc > 5 && std::string(argv[5]) == "int8"
                                                        ? ninfer::KvCacheStorage::Int8Group64
                                                        : ninfer::KvCacheStorage::BFloat16);
        auto penalty          = request(24);
        penalty.execution.sampling.presence_penalty  = 0.5F;
        penalty.execution.sampling.frequency_penalty = 0.25F;
        options.speculative.backend                  = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens             = k;
        options.speculative.proposal_head =
            optimized ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
        if (remote_attention) {
            options.remote_attention.secondary_device      = 1;
            options.remote_attention.full_attention_layers = 2;
        }
        ninfer::Engine engine(options);
        if (remote_attention) {
            const auto& configured = engine.options();
            require(configured.max_concurrency == 2 && !configured.use_cuda_graph &&
                        configured.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256 &&
                        configured.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                        configured.speculative.draft_tokens == 7 &&
                        configured.remote_attention.secondary_device == 1 &&
                        configured.remote_attention.full_attention_layers == 2 &&
                        !configured.active_kv.secondary_device.has_value(),
                    "remote-attention Engine options did not retain the requested configuration");
            const auto memory = engine.memory_summary();
            require(memory.remote_attention_primary_bytes != 0 &&
                        memory.remote_attention_secondary_bytes != 0 &&
                        memory.active_kv_secondary_device == -1 &&
                        memory.active_kv_secondary_bytes == 0,
                    "remote attention did not reserve K/V on both GPUs");
        }
        const auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");
        ninfer::test::speculative_page_boundary(engine);
        if (remote_attention) {
            remote_page_frontier_continuation(engine, prompt);

            const auto before_c2 = engine.runtime_stats();
            std::vector<ninfer::GenerationHandle> c2_handles;
            auto first_c2_prompt =
                engine.prepare_tokens(engine.tokenize_text("Continue counting upward from four:"));
            auto second_c2_prompt = engine.prepare_tokens(
                engine.tokenize_text("Continue counting downward from four:"));
            c2_handles.push_back(engine.submit(std::move(first_c2_prompt), request(32)));
            c2_handles.push_back(engine.submit(std::move(second_c2_prompt), request(32)));
            for (auto& handle : c2_handles) {
                const auto result = handle.wait();
                valid(result, 32);
                require(result.speculative.accepted_tokens != 0,
                        "concurrent DFlash2 stream accepted no draft proposal");
            }
            const auto after_c2      = engine.runtime_stats();
            const auto decode_rounds = after_c2.decode_rounds - before_c2.decode_rounds;
            const auto decode_rows   = after_c2.decode_row_rounds - before_c2.decode_row_rounds;
            require(decode_rounds != 0 && decode_rows > decode_rounds,
                    "two submitted requests never shared a batched decode round");
        }
        const auto first = engine.generate(engine.prepare_tokens(prompt), request(24));
        valid(first, 24);
        const auto& reference = first.generated_token_ids;
        require(first.speculative.accepted_tokens != 0, "real draft fixture accepted no proposal");
        const auto penalized = engine.generate(engine.prepare_tokens(prompt), penalty);
        valid(penalized, 24);


        // All rows share a known target prefix, while their budgets force P=0, partial and full W.
        std::vector<ninfer::GenerationHandle> handles;
        for (unsigned row = 0; row < batch; ++row) {
            handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(2 + row * 3)));
        }
        for (unsigned row = 0; row < batch; ++row) {
            const auto result = handles[row].wait();
            valid(result, 2 + row * 3);
        }
        // Rejection RNG uses the same logical positions and seed for repeat requests.
        const auto sample_outputs                    = remote_attention ? 32U : 16U;
        auto sampled                                 = request(sample_outputs);
        sampled.execution.sampling.temperature       = 0.8F;
        sampled.execution.sampling.top_p             = 0.9F;
        sampled.execution.sampling.top_k             = 20;
        sampled.execution.sampling.presence_penalty  = 0.3F;
        sampled.execution.sampling.frequency_penalty = 0.2F;
        sampled.execution.sampling.seed              = 42;
        const auto sample1 = engine.generate(engine.prepare_tokens(prompt), sampled);
        const auto sample2 = engine.generate(engine.prepare_tokens(prompt), sampled);
        valid(sample1, sample_outputs);
        require(sample1.generated_token_ids == sample2.generated_token_ids,
                "same DFlash2 seed and inputs did not reproduce the conditional path");
        if (remote_attention) {
            bool rejected =
                sample1.speculative.drafted_tokens > sample1.speculative.accepted_tokens;
            for (std::uint32_t seed = 43; seed < 47 && !rejected; ++seed) {
                sampled.execution.sampling.seed = seed;
                const auto candidate = engine.generate(engine.prepare_tokens(prompt), sampled);
                valid(candidate, sample_outputs);
                rejected =
                    candidate.speculative.drafted_tokens > candidate.speculative.accepted_tokens;
            }
            require(rejected, "deterministic sampling fixture rejected no draft proposal");
        }

        // Terminal flush and fork must retain the consumed frontier for a later request.
        const auto retained = engine.generate(engine.prepare_tokens(prompt), request(12, true));
        auto continuation   = prompt;
        continuation.insert(continuation.end(), retained.generated_token_ids.begin(),
                            retained.generated_token_ids.end());
        continuation.push_back(198);
        const auto reused = engine.generate(engine.prepare_tokens(continuation), request(8, true));
        const auto fresh  = engine.generate(engine.prepare_tokens(continuation), request(8, false));
        valid(reused, 8);
        valid(fresh, 8);
        require(reused.reused_prompt_tokens != 0 && fresh.reused_prompt_tokens == 0,
                "DFlash2 prefix restore did not reuse its retained frontier");
        if (remote_attention) {
            require(reused.generated_token_ids == fresh.generated_token_ids,
                    "reused continuation diverged from fresh committed output");
        }

        if (k >= 7) {
            bool checked_partial = false;
            for (std::size_t i = 1; i < std::min<std::size_t>(k, reference.size()); ++i) {
                if (std::find(reference.begin(), reference.begin() + i, reference[i]) !=
                    reference.begin() + i) {
                    continue;
                }
                auto stopped_options = request(24, true);
                stopped_options.stop.token_ids.push_back(reference[i]);
                const auto stopped =
                    engine.generate(engine.prepare_tokens(prompt), stopped_options);
                const auto licensed = 1 + stopped.speculative.rounds +
                                      stopped.speculative.accepted_tokens +
                                      stopped.speculative.fallback_steps;
                if (stopped.generated_token_ids.size() >= licensed) { continue; }
                require(stopped.finish_reason == ninfer::FinishReason::StopToken,
                        "partial terminal did not stop at its token");
                auto follow = prompt;
                follow.insert(follow.end(), stopped.generated_token_ids.begin(),
                              stopped.generated_token_ids.end());
                follow.push_back(198);
                const auto reused_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, true));
                const auto fresh_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, false));
                valid(reused_stop, 8);
                valid(fresh_stop, 8);
                require(reused_stop.reused_prompt_tokens != 0 &&
                            reused_stop.reused_prompt_tokens <=
                                prompt.size() + stopped.generated_token_ids.size() &&
                            fresh_stop.reused_prompt_tokens == 0,
                        "partial terminal exposed an uncommitted prefix");
                if (remote_attention) {
                    require(reused_stop.generated_token_ids == fresh_stop.generated_token_ids,
                            "partial-stop reuse exposed speculative output");
                }
                checked_partial = true;
                break;
            }
            require(checked_partial, "fixture did not exercise a stop within a licensed block");
        }
        if (options.enable_vision) {
            for (const auto kind : {ninfer::MediaKind::Image, ninfer::MediaKind::Video}) {
                const auto image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                const auto reused_image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                valid(image, 8);
                valid(reused_image, 8);
                require(image.prompt.has_media && image.timings.vision_seconds > 0 &&
                            reused_image.reused_prompt_tokens != 0 &&
                            reused_image.timings.vision_seconds == 0.0,
                        "Vision DFlash2 capture/restore recomputed retained media");
            }
        }
        const auto stats = engine.runtime_stats();
        require(stats.device_backend_kv_occupied_pages == 0 && stats.backend_kv_d2h_bytes == 0 &&
                    stats.backend_kv_h2d_bytes == 0,
                "DFlash2 allocated or transferred a full backend KV pool");
        if (k == 15) {
            // One oversized prefill replaces the ring, then decode appends across its wrap point.
            auto long_prompt = std::vector<ninfer::TokenId>(2100, 198);
            long_prompt.insert(long_prompt.end(), prompt.begin(), prompt.end());
            const auto long_run =
                engine.generate(engine.prepare_tokens(long_prompt), request(12, true));
            valid(long_run, 12);
            long_prompt.insert(long_prompt.end(), long_run.generated_token_ids.begin(),
                               long_run.generated_token_ids.end());
            const auto long_reuse =
                engine.generate(engine.prepare_tokens(long_prompt), request(6, true));
            const auto long_fresh =
                engine.generate(engine.prepare_tokens(long_prompt), request(6, false));
            valid(long_reuse, 6);
            valid(long_fresh, 6);
            require(long_reuse.reused_prompt_tokens > 2048 && long_fresh.reused_prompt_tokens == 0,
                    "DFlash2 ring wrap lost its retained frontier");
        }
        if (k == 15) {
            auto tail_prompt   = std::vector<ninfer::TokenId>(options.max_context - 4, 198);
            tail_prompt.back() = prompt.back();
            const auto tail    = engine.generate(engine.prepare_tokens(tail_prompt), request(9));
            require(tail.finish_reason == ninfer::FinishReason::ContextCapacity &&
                        tail.generated_token_ids.size() == 5,
                    "full proposal window escaped the target context capacity tail");
        }
        if (remote_attention) {
            auto tail_prompt   = std::vector<ninfer::TokenId>(options.max_context - 4, 198);
            tail_prompt.back() = prompt.back();
            const auto tail    = engine.generate(engine.prepare_tokens(tail_prompt), request(9));
            require(tail.finish_reason == ninfer::FinishReason::ContextCapacity &&
                        tail.generated_token_ids.size() == 5,
                    "remote DFlash2 escaped the target context-capacity tail");
        }
        std::cout << "ok K=" << k << " B=" << batch << " graph=" << graph
                  << " optimized=" << optimized << " accepted=" << first.speculative.accepted_tokens
                  << "/" << first.speculative.drafted_tokens
                  << " state_d2h=" << stats.state_d2h_count
                  << " state_h2d=" << stats.state_h2d_count << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
