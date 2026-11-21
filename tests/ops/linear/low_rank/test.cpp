#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/layout.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/support.h"
#include "ops/quantized_weight.h"

#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
using ops::LinearPolicy;
namespace qw = quantized_weight;

enum class Op { SwiGlu, Split, Add };
constexpr float kScale = 2.0f;
constexpr std::array kSegments{6144, 1024, 6144, 1024};

// Sparse BF16 X makes a complete FP64 reference cheap even at real prefill
// extents. Each token selects a different input coordinate and sign. The A/B
// patterns distinguish row slices, rank coordinates, and gate/up branches.
int input_column(int token, int k) { return (17 + 257 * token) % k; }

float input_value(int token) { return token % 2 == 0 ? 1.0f : -0.5f; }

float a_value(int group, int rank, int column) {
    return (float((group * 7 + rank * 11 + column * 5) % 31) - 15.0f) / 16.0f;
}

float b_value(int slice, int rank, int row) {
    return (float((slice * 5 + rank * 13 + row * 3) % 23) - 11.0f) / 32.0f;
}

float residual_value(int token, int row) { return (float((token + 3 * row) % 27) - 13.0f) / 16.0f; }

int check(Op op, QType format, int k, int n, int tokens, LinearPolicy policy, bool replay) {
    qw::PatternedWeightOptions options;
    if (format == QType::NVFP4) {
        options.weight_scale_divisor = 0.125f;
        options.input_scale_divisor  = 3.5f;
    }
    const auto packed = qw::make_patterned_weight(format, n, k, 33581U, options);
    GuardedDeviceBuffer device_weight(packed.payload.size());
    device_weight.copy_from_host(packed.payload.data(), packed.payload.size());
    const auto weight                       = packed.device_weight(device_weight.span());
    const ops::ProjectionWeights projection = ops::SingleProjectionWeight{weight, policy};

    std::vector<float> x_values(std::size_t(k) * tokens);
    for (int t = 0; t < tokens; ++t)
        x_values[std::size_t(t) * k + input_column(t, k)] = input_value(t);
    auto input = to_device_bf16(x_values);
    const Tensor x(input.span(), DType::BF16, {k, tokens});
    const int half = n / 2;
    const std::array nk{n};
    const std::array pair{half, half};
    const auto rows = op == Op::Split    ? std::span<const int>(kSegments)
                      : op == Op::SwiGlu ? std::span<const int>(pair)
                                         : std::span<const int>(nk);
    // Q and gate share A, but have different B, just like the deployed adapter.
    constexpr std::array group{0, 1, 0, 2};
    const auto source  = [&](int slice) { return op == Op::Split ? group[slice] : slice; };
    const int distinct = op == Op::Split ? 3 : static_cast<int>(rows.size());
    std::vector<DeviceBuffer> a, b, output;
    std::array<ops::LowRankFactors, 4> updates{};
    std::array<Tensor, 4> destinations{};
    std::vector<float> data(std::size_t(k) * ops::kLowRank);
    for (int g = 0; g < distinct; ++g) {
        for (int r = 0; r < ops::kLowRank; ++r)
            for (int i = 0; i < k; ++i) data[std::size_t(r) * k + i] = a_value(g, r, i);
        a.push_back(to_device(data));
    }
    for (int s = 0; s < static_cast<int>(rows.size()); ++s) {
        const int width = rows[s];
        data.resize(std::size_t(width) * ops::kLowRank);
        for (int r = 0; r < ops::kLowRank; ++r)
            for (int i = 0; i < width; ++i) data[std::size_t(r) * width + i] = b_value(s, r, i);
        b.push_back(to_device(data));
        updates[s] = {Tensor(a[source(s)].span(), DType::FP32, {k, ops::kLowRank}),
                      Tensor(b.back().span(), DType::FP32, {width, ops::kLowRank})};
        if (op == Op::SwiGlu && s == 1) continue;
        const auto count = std::size_t(width) * tokens;
        if (op == Op::Add) {
            data.resize(count);
            for (int t = 0; t < tokens; ++t)
                for (int i = 0; i < width; ++i)
                    data[std::size_t(t) * width + i] = residual_value(t, i);
            output.push_back(to_device_bf16(data));
        } else {
            output.emplace_back(cuda_environment().context.device(), count * sizeof(std::uint16_t));
        }
        destinations[s] = Tensor(output.back().span(), DType::BF16, {width, tokens});
    }

    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ops::low_rank_workspace_bytes(tokens, distinct));
    const auto base_bytes =
        op == Op::SwiGlu
            ? ops::linear_swiglu_workspace_capacity_bytes(format, n, k, policy, tokens, tokens)
        : op == Op::Split
            ? ops::linear_split_workspace_capacity_bytes(projection, rows, tokens, tokens)
            : ops::linear_add_workspace_capacity_bytes(format, n, k, policy, tokens, tokens);
    (void)layout.alloc_bytes(base_bytes);
    GuardedDeviceBuffer scratch(layout.peak_bytes(1));
    MemoryArena workspace(scratch.span());
    // On the capture stream, run the graph twice with different input values
    // in the same backing. This catches stale rank intermediates (not just
    // whether the capture can be instantiated).
    std::optional<DeviceContext> graph_context;
    if (replay) graph_context.emplace();
    const auto cuda =
        replay ? graph_context->execution_view() : cuda_environment().default_execution;
    const auto launch = [&] {
        if (op == Op::SwiGlu)
            ops::linear_swiglu(x, weight, destinations[0], policy, {updates[0], updates[1]}, kScale,
                               workspace, cuda);
        else if (op == Op::Split)
            ops::linear_split(x, projection, destinations, updates, kScale, workspace, cuda);
        else
            ops::linear_add(x, weight, destinations[0], policy, updates[0], kScale, workspace,
                            cuda);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (replay) {
        definition.capture(cuda.stream(), launch);
        graph.instantiate(definition);
        graph.launch(cuda.stream());
        cuda_synchronize(cuda.stream());
        for (auto& value : x_values) value = -value;
        std::vector<std::uint16_t> negative(x_values.size());
        for (std::size_t i = 0; i < negative.size(); ++i) negative[i] = f32_to_bf16(x_values[i]);
        input.copy_from_host(negative.data(), negative.size() * sizeof(negative[0]));
        graph.launch(cuda.stream());
    } else {
        launch();
    }
    cuda_synchronize(cuda.stream());

    // With no base-Op scratch, peak use is exactly the rank intermediates:
    // distinct * tokens * rank floats at every token count. An output-width
    // delta would show up here as a term in n.
    if (workspace.used() != 0 || workspace.peak_used() > layout.peak_bytes(1) ||
        (base_bytes == 0 &&
         workspace.peak_used() != sizeof(float) * std::size_t(distinct) * tokens * ops::kLowRank)) {
        std::cerr << "fused low-rank projection exceeded or failed to release its workspace\n";
        return 1;
    }

    const auto projected = [&](int slice, int parent_row, int row, int token) {
        const auto column = input_column(token, k);
        double delta      = 0;
        for (int r = 0; r < ops::kLowRank; ++r)
            delta += double(b_value(slice, r, row)) * a_value(source(slice), r, column);
        return x_values[std::size_t(token) * k + column] *
               (qw::logical_weight_fp64(packed, parent_row, column) + kScale * delta);
    };
    std::vector<double> actual, expected;
    int offset = 0;
    for (int s = 0; s < static_cast<int>(output.size()); ++s) {
        const int width = rows[s];
        const auto got  = from_device_bf16(output[s], std::size_t(width) * tokens);
        // Include both sides of packed slice boundaries and the last partial tile.
        for (const int t : {0, tokens / 2, tokens - 1}) {
            for (const int i : {0, 1, 7, 8, 31, 32, width / 2, width - 2, width - 1}) {
                double y = projected(s, offset + i, i, t);
                if (op == Op::SwiGlu) y = y / (1 + std::exp(-y)) * projected(1, half + i, i, t);
                if (op == Op::Add) y += residual_value(t, i);
                actual.push_back(got[std::size_t(t) * width + i]);
                expected.push_back(y);
            }
        }
        offset += width;
    }
    // Allow the existing production activation precision; compare against
    // exact represented weights and BF16 input values, not against another Op.
    const bool nonlinear = op == Op::SwiGlu;
    const auto criterion = [&]() -> ReductionCriterion {
        switch (policy) {
        case LinearPolicy::AllowA4:
            return {0.16, nonlinear ? 0.01 : 1.0 / 256, 0.16};
        case LinearPolicy::AllowA8:
            return nonlinear ? ReductionCriterion{0.08, 0.01, 0.12}
                             : ReductionCriterion{0.04, 1.0 / 256, 0.06};
        case LinearPolicy::A16Only:
            return nonlinear ? ReductionCriterion{0.0033, 0.005, 0.0063}
                             : ReductionCriterion{1.0 / 256, 1.0 / 256, 2.0 / 256};
        }
        throw std::logic_error("invalid fixture policy");
    }();
    return verify_reduction("fused low-rank projection", actual, expected, criterion) +
           scratch.verify_guards("fused low-rank workspace");
}
} // namespace

int main(int argc, char** argv) {
    using linear_suite::Execution;
    linear_suite::Suite suite;
    const auto add = [&](Op op, QType format, int k, int n, int t, LinearPolicy policy,
                         std::string_view label, bool replay = false) {
        suite.add(linear_suite::make_case(linear_suite::format_name(format), n, k, t, policy,
                                          replay ? Execution::Graph : Execution::Eager, label),
                  [=] { return check(op, format, k, n, t, policy, replay); });
    };
    constexpr auto fp8 = QType::FP8_E4M3FN_ROW_BF16;
    constexpr auto fp4 = QType::NVFP4;
    add(Op::SwiGlu, fp4, 5120, 34816, 1, LinearPolicy::A16Only, "lora-swiglu-decode", true);
    add(Op::SwiGlu, fp4, 5120, 34816, 8, LinearPolicy::AllowA4, "lora-swiglu-verify");
    add(Op::SwiGlu, fp4, 5120, 34816, 17, LinearPolicy::AllowA4, "lora-swiglu-wide-tail");
    add(Op::SwiGlu, fp4, 5120, 34816, 257, LinearPolicy::AllowA4, "lora-swiglu-tma-tail");
    add(Op::SwiGlu, fp4, 5120, 34816, 1024, LinearPolicy::AllowA4, "lora-swiglu-prefill");
    add(Op::SwiGlu, fp8, 5120, 34816, 257, LinearPolicy::AllowA8, "lora-swiglu-fp8-tail");
    add(Op::Split, fp8, 5120, 14336, 1, LinearPolicy::A16Only, "lora-split-decode");
    add(Op::Split, fp8, 5120, 14336, 8, LinearPolicy::A16Only, "lora-split-verify", true);
    add(Op::Split, fp8, 5120, 14336, 385, LinearPolicy::AllowA8, "lora-split-prefill");
    add(Op::Split, fp4, 5120, 14336, 513, LinearPolicy::AllowA4, "lora-split-tma-tail");
    add(Op::Add, fp4, 17408, 5120, 1, LinearPolicy::A16Only, "lora-add-decode");
    add(Op::Add, fp4, 17408, 5120, 8, LinearPolicy::AllowA4, "lora-add-down");
    add(Op::Add, fp8, 6144, 5120, 193, LinearPolicy::AllowA8, "lora-add-attention");
    return suite.main(argc, argv);
}
