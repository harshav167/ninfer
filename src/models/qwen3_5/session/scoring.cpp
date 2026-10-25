#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

// Appends text tokens to a text-only prompt. Appended positions follow the committed-token
// rule used everywhere else: the sequence index plus the prompt's rope delta on every axis.
void append_text_tokens(qwen::PreparedPromptData& prompt, std::span<const TokenId> tokens) {
    const std::size_t old_count = prompt.token_ids.size();
    if (prompt.token_types.size() != old_count || prompt.positions.size() != 3 * old_count)
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    const std::size_t count = old_count + tokens.size();
    std::vector<std::int32_t> positions(3 * count);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto source =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_count);
        const auto target = positions.begin() + static_cast<std::ptrdiff_t>(axis * count);
        std::copy(source, source + static_cast<std::ptrdiff_t>(old_count), target);
        for (std::size_t index = old_count; index < count; ++index)
            positions[axis * count + index] = static_cast<std::int32_t>(index) + prompt.rope_delta;
    }
    prompt.token_ids.insert(prompt.token_ids.end(), tokens.begin(), tokens.end());
    prompt.token_types.resize(count, 0);
    prompt.positions = std::move(positions);
}

} // namespace

std::optional<runtime::ExecutionTicket> SessionImpl::start_score(qwen::PreparedPromptData&& prompt,
                                                                 std::uint32_t first_target) {
    if (!causal_scoring) throw std::logic_error("Session is not configured for causal scoring");
    if (scoring || execution || has_context_operation()) return {};
    if (prompt.token_ids.size() < 2 || prompt.token_ids.size() > capacity || !first_target ||
        first_target >= prompt.token_ids.size())
        throw std::invalid_argument("causal scoring window or target is invalid");
    if (prompt.has_media()) throw std::invalid_argument("causal scoring accepts text tokens only");
    return begin_scoring(std::move(prompt), first_target, std::nullopt);
}

std::optional<runtime::ExecutionTicket>
SessionImpl::start_residual_capture(qwen::PreparedPromptData&& prompt,
                                    std::span<const TokenId> forced_suffix,
                                    std::span<const std::uint32_t> layers) {
    if (!causal_scoring) throw std::logic_error("Session is not configured for causal scoring");
    if (scoring || execution || has_context_operation()) return {};
    if (layers.empty() ||
        std::adjacent_find(layers.begin(), layers.end(), std::greater_equal<>{}) != layers.end() ||
        layers.back() >= plan.profile->config.text.num_hidden_layers)
        throw std::invalid_argument(
            "residual capture layers must be strictly ascending Main layer indices");
    if (prompt.has_media())
        throw std::invalid_argument("residual capture accepts text tokens only");
    const std::size_t tokens = prompt.token_ids.size() + forced_suffix.size();
    if (tokens == 0 || tokens > capacity)
        throw std::invalid_argument("residual capture window exceeds the context capacity");
    for (const TokenId id : forced_suffix)
        if (id < 0 || static_cast<std::uint32_t>(id) >= plan.profile->public_tokens)
            throw std::invalid_argument("forced suffix token is outside the public token domain");
    append_text_tokens(prompt, forced_suffix);
    return begin_scoring(std::move(prompt), 0,
                         std::vector<std::uint32_t>(layers.begin(), layers.end()));
}

std::optional<runtime::ExecutionTicket>
SessionImpl::begin_scoring(qwen::PreparedPromptData&& prompt, std::uint32_t first_target,
                           std::optional<std::vector<std::uint32_t>> capture_layers) {
    runtime::ResolvedExecutionOptions options;
    options.allow_prefix_reuse = false;
    auto base                  = plan_request(std::move(prompt), options);
    const auto source          = inspect_source(base, {});
    if (!source) throw std::logic_error("scoring has no root binding");
    const std::uint32_t tokens = base.summary().prompt_tokens;
    const bool capture         = capture_layers.has_value();
    scoring.emplace(Scoring{.base           = std::move(base),
                            .first_target   = first_target,
                            .limit          = capture ? tokens : tokens - 1,
                            .capture_layers = std::move(capture_layers)});
    try {
        scoring->logprobs.reserve(capture ? 0 : tokens - first_target);
        const auto binding = start_binding(scoring->base, {0}, *source);
        if (!binding) {
            scoring.reset();
            return {};
        }
        if (!context_operation_ || !context_operation_->ticket)
            throw std::logic_error("scoring context was not submitted");
        scoring->public_ticket = *context_operation_->ticket;
        return scoring->public_ticket;
    } catch (...) {
        if (context_operation_ && context_operation_->ticket)
            executor().fail("scoring setup failed after context submission");
        else {
            abort_context();
            scoring.reset();
        }
        throw;
    }
}

void SessionImpl::advance_score() {
    if (!scoring || execution) throw std::logic_error("scoring continuation has no free frame");
    const auto handle  = scoring->sequence;
    auto& state        = sequences[lane(handle)];
    auto& request      = requests[state.lane];
    const auto& prompt = *scoring->base.impl_->prompt;
    const auto limit   = scoring->limit;
    if (scoring->cursor == limit) {
        auto done = std::move(*scoring);
        scoring.reset();
        std::optional<ResidualObservation> observation;
        if (done.capture_layers)
            observation.emplace(ResidualObservation{done.base.impl_->prompt->token_ids,
                                                    plan.profile->config.text.hidden_size,
                                                    std::move(done.residuals)});
        clear_lane(state, request);
        after_releases.emplace(
            SessionEvent{done.public_ticket,
                         {},
                         ScoreComplete{std::move(done.logprobs), std::move(observation)}});
        submit_releases();
        return;
    }
    request.prefill->cursor = scoring->cursor;
    const std::array unit{ExecutionUnit{handle, ExecutionUnitKind::Prefill, 0}};
    if (!reserve_units(unit)) throw std::logic_error("scoring lost its resident context capacity");
    const auto end   = std::min(limit, scoring->cursor + prefill_chunk);
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) throw std::logic_error("scoring frame did not retire");
    try {
        Execution pending{.kind = OperationKind::Score};
        pending.members     = {handle};
        pending.bases       = {state.continuation};
        pending.columns     = {end - scoring->cursor};
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        ensure_sequence_kv_mapped(state, end, 0, *pending.preparation);
        ScoreWork score;
        score.row   = execution_row(state);
        score.input = input_window(state, scoring->cursor, end);
        if (scoring->capture_layers) {
            // Only the window that ends the sequence observes its last column.
            if (end == limit)
                score.capture =
                    ResidualCaptureWork{*scoring->capture_layers, end - scoring->cursor - 1};
        } else {
            const auto first = std::max(scoring->first_target, scoring->cursor + 1);
            if (first <= end) {
                score.first_column = first - scoring->cursor - 1;
                score.targets.assign(prompt.token_ids.begin() + first,
                                     prompt.token_ids.begin() + end + 1);
            }
        }
        (void)submit_execution(std::move(pending), std::move(*reservation),
                               Work{.operation = std::move(score)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
