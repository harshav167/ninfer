#include "ops/suite.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "cuda_fixture.h"
#include "ninfer/ops/clipped_direction_removal.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The Op sees slot kSlot of per-layer tables [H,2,kSlots] and [2,kSlots], like the model does.
// The neighbouring slots hold other data and must stay untouched; slot 1 of the references
// table is 8-byte but not 16-byte aligned.
constexpr int kSlots = 3;
constexpr int kSlot  = 1;

// Distance, in units of the clipped coefficient, by which a recipe makes a step clearly active
// or clearly inactive, so that FP32 evaluation cannot change which side of the clip a column is on.
constexpr double kMargin = 4.0;

// A column whose two steps are both inactive by at least this much is unchanged, bit for bit.
constexpr double kUnchangedMargin = 1.0;

// Domain of the numerical criterion below, checked on every evaluated problem.
constexpr double kMaxErrorScale = 500.0;

// Result = BF16_RN(y~), y~ the FP32 pipeline value; y is the exact (FP64) value. Round-to-nearest
// BF16 moves a value by at most 2^-8 of its magnitude, so
//   |BF16(y~) - y| <= 2^-8 |y| + (1 + 2^-8) |y~ - y|.
// Every dot product is a chain of at most 19 FP32 roundings (8 FMAs per thread, 3 folds, 5 warp and
// 3 cross-warp additions), so its error is at most gamma_19 sum_j |h_j u_j| with gamma_19 = 1.2e-6,
// and it reaches an element scaled by |u_i|. The recipes keep max|u| * sum|h u| <= kMaxErrorScale
// (checked against the oracle), so one step contributes at most 6e-4; two steps plus the FP32
// rounding of the elements stay below the 2e-3 absolute term. Inputs are finite by contract, so a
// non-finite output or oracle value is a failure.
constexpr PointwiseCriterion clipped_direction_removal_criterion() {
    return {/*absolute*/ 2.0e-3, /*relative*/ 1.0 / 256.0};
}

std::uint64_t stable_seed(std::string_view text) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const char c : text) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ---- input recipes ----------------------------------------------------------------------------

enum class Coupling {
    Independent, // u1 is an independent random direction
    Coupled,     // unit(u1) has inner product 0.6 with unit(u0)
    ZeroSecond,  // u1 is exactly zero
};

struct Reference {
    enum class Kind { Fixed, Active, Inactive };
    Kind kind;
    double value;
};

constexpr Reference fixed_reference(double value) { return {Reference::Kind::Fixed, value}; }

constexpr Reference kActive{Reference::Kind::Active, 0.0};
constexpr Reference kInactive{Reference::Kind::Inactive, 0.0};

// A column is h = noise + a u0hat + b u1hat (+ massive channels), so a and b set which side of
// each clip it lands on. Active/Inactive references are derived from the oracle's dot products so
// that every column is clearly active (or clearly inactive) at that step.
struct Recipe {
    const char* name;
    Coupling coupling = Coupling::Independent;
    double norm0      = 1.0; // direction lengths
    double norm1      = 1.0;
    double alpha      = 60.0; // typical a and b
    double beta       = 60.0;
    bool mixed_signs  = true;  // pseudo-random sign of a and b per column; column 0 positive
    bool massive      = false; // channels of magnitude 1e3..1e4, two cancelling in dot(h,u0)
    bool specials     = false; // -0, +0 and subnormal/normal-edge BF16 words in every column
    Reference first   = fixed_reference(0.0);
    Reference second  = fixed_reference(0.0);
};

enum class Regime {
    Mixed,
    Both,
    FirstOnly,
    SecondOnly,
    Neither,
    ZeroNegative,
    ZeroPositive,
    NegativeReferences,
    LargeReferences,
    Massive,
    NonUnit,
    Sequential,
    RoundTrap,
};

Recipe regime_recipe(Regime regime) {
    switch (regime) {
    case Regime::Mixed:
        return {.name = "mixed"};
    case Regime::Both:
        return {.name = "both", .first = kActive, .second = kActive};
    case Regime::FirstOnly:
        return {.name = "first-only", .first = kActive, .second = kInactive};
    case Regime::SecondOnly:
        return {.name = "second-only", .first = kInactive, .second = kActive};
    case Regime::Neither:
        return {.name = "neither", .specials = true, .first = kInactive, .second = kInactive};
    case Regime::ZeroNegative:
        return {.name     = "zero-disclaimer-negative",
                .coupling = Coupling::ZeroSecond,
                .beta     = 0.0,
                .second   = fixed_reference(-3.5)};
    case Regime::ZeroPositive:
        return {.name     = "zero-disclaimer-positive",
                .coupling = Coupling::ZeroSecond,
                .beta     = 0.0,
                .second   = fixed_reference(2.5)};
    case Regime::NegativeReferences:
        return {.name   = "negative-references",
                .first  = fixed_reference(-200.0),
                .second = fixed_reference(-50.0)};
    case Regime::LargeReferences:
        return {.name   = "large-references",
                .first  = fixed_reference(1.0e6),
                .second = fixed_reference(-1000.0)};
    case Regime::Massive:
        return {.name = "massive", .alpha = 30.0, .beta = 30.0, .massive = true};
    case Regime::NonUnit:
        return {.name = "non-unit", .norm0 = 0.5, .norm1 = 2.0};
    case Regime::Sequential:
        return {.name        = "sequential",
                .coupling    = Coupling::Coupled,
                .alpha       = 200.0,
                .beta        = 0.0,
                .mixed_signs = false,
                .first       = kActive,
                .second      = kActive};
    case Regime::RoundTrap:
        // h1 is large (about 300 u1hat) and the second step cancels it, so rounding h1 to BF16
        // between the steps leaves an error far above the criterion in the final elements.
        return {.name        = "round-trap",
                .alpha       = 20.0,
                .beta        = 300.0,
                .mixed_signs = false,
                .first       = kActive,
                .second      = fixed_reference(0.0)};
    }
    throw std::logic_error("unknown regime");
}

struct Problem {
    int h = 0;
    int t = 0;
    std::vector<std::uint16_t> residual; // BF16 words [h,t]
    std::vector<float> directions;       // FP32 [h,2]
    std::array<float, 2> references{};
};

// ---- oracle -----------------------------------------------------------------------------------

struct Evaluation {
    std::vector<double> values;     // exact h2 of every element, [h,t]
    std::vector<double> dot0, dot1; // dot(h,u0) and dot(h1,u1) per column, before the references
    std::vector<char> unchanged;    // both steps inactive by at least kUnchangedMargin
    double error_scale = 0.0;       // max over columns and steps of max|u| * sum_j |h_j u_j|
};

// Plain scalar FP64 evaluation of the complete formula over the represented inputs: BF16 words
// widened, FP32 directions and references as doubles; no intermediate rounding of any kind.
Evaluation evaluate(const Problem& p) {
    const int h     = p.h;
    const int t     = p.t;
    const float* u0 = p.directions.data();
    const float* u1 = u0 + h;
    const double c0 = p.references[0];
    const double c1 = p.references[1];
    double max0     = 0.0;
    double max1     = 0.0;
    for (int i = 0; i < h; ++i) {
        max0 = std::max(max0, std::abs(static_cast<double>(u0[i])));
        max1 = std::max(max1, std::abs(static_cast<double>(u1[i])));
    }

    Evaluation e;
    e.values.resize(static_cast<std::size_t>(h) * t);
    e.dot0.resize(t);
    e.dot1.resize(t);
    e.unchanged.resize(t);
    std::vector<double> scale(t);
    parallel_rows(t, static_cast<std::uint64_t>(h) * t, [&](int begin, int end) {
        std::vector<double> x(h);
        for (int m = begin; m < end; ++m) {
            const std::uint16_t* words = p.residual.data() + static_cast<std::size_t>(m) * h;
            double dot0                = 0.0;
            double sum0                = 0.0;
            for (int i = 0; i < h; ++i) {
                x[i]              = bf16_to_f32(words[i]);
                const double term = x[i] * u0[i];
                dot0 += term;
                sum0 += std::abs(term);
            }
            const double a0 = std::max(dot0 - c0, 0.0);
            double dot1     = 0.0;
            double sum1     = 0.0;
            for (int i = 0; i < h; ++i) {
                x[i] -= a0 * u0[i];
                const double term = x[i] * u1[i];
                dot1 += term;
                sum1 += std::abs(term);
            }
            const double a1 = std::max(dot1 - c1, 0.0);
            double* out     = e.values.data() + static_cast<std::size_t>(m) * h;
            for (int i = 0; i < h; ++i) { out[i] = x[i] - a1 * u1[i]; }
            e.dot0[m]      = dot0;
            e.dot1[m]      = dot1;
            e.unchanged[m] = dot0 - c0 <= -kUnchangedMargin && dot1 - c1 <= -kUnchangedMargin;
            scale[m]       = std::max(max0 * sum0, max1 * sum1);
        }
    });
    e.error_scale = *std::max_element(scale.begin(), scale.end());
    return e;
}

// ---- problem construction ---------------------------------------------------------------------

std::vector<double> gaussian(std::mt19937_64& rng, int count) {
    std::normal_distribution<double> normal;
    std::vector<double> values(count);
    for (auto& value : values) { value = normal(rng); }
    return values;
}

void normalize_in_place(std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) { sum += value * value; }
    const double length = std::sqrt(sum);
    for (auto& value : values) { value /= length; }
}

float reference_for(const Reference& reference, const Problem& p, bool second) {
    if (reference.kind == Reference::Kind::Fixed) { return static_cast<float>(reference.value); }
    const Evaluation e = evaluate(p);
    const auto& dots   = second ? e.dot1 : e.dot0;
    return static_cast<float>(reference.kind == Reference::Kind::Active
                                  ? *std::min_element(dots.begin(), dots.end()) - kMargin
                                  : *std::max_element(dots.begin(), dots.end()) + kMargin);
}

Problem make_problem(const Recipe& spec, int h, int t, std::uint64_t direction_seed,
                     std::uint64_t column_seed) {
    Problem p;
    p.h = h;
    p.t = t;

    std::mt19937_64 direction_rng(direction_seed);
    std::vector<double> unit0 = gaussian(direction_rng, h);
    std::vector<double> unit1(h, 0.0);
    normalize_in_place(unit0);
    if (spec.coupling == Coupling::Independent) {
        unit1 = gaussian(direction_rng, h);
        normalize_in_place(unit1);
    } else if (spec.coupling == Coupling::Coupled) {
        std::vector<double> orthogonal = gaussian(direction_rng, h);
        double along                   = 0.0;
        for (int i = 0; i < h; ++i) { along += orthogonal[i] * unit0[i]; }
        for (int i = 0; i < h; ++i) { orthogonal[i] -= along * unit0[i]; }
        normalize_in_place(orthogonal);
        for (int i = 0; i < h; ++i) { unit1[i] = 0.6 * unit0[i] + 0.8 * orthogonal[i]; }
    }
    p.directions.resize(static_cast<std::size_t>(h) * 2);
    for (int i = 0; i < h; ++i) {
        p.directions[i]     = static_cast<float>(spec.norm0 * unit0[i]);
        p.directions[h + i] = static_cast<float>(spec.norm1 * unit1[i]);
    }

    // Massive channels: the two extremes of u0 carry the same-sign activation with cancelling
    // contributions to dot(h,u0); a third channel does not cancel.
    const float* u0  = p.directions.data();
    const int top    = static_cast<int>(std::max_element(u0, u0 + h) - u0);
    const int bottom = static_cast<int>(std::min_element(u0, u0 + h) - u0);
    int third        = (top + h / 2) % h;
    while (third == top || third == bottom) { third = (third + 1) % h; }

    std::mt19937_64 rng(column_seed);
    std::normal_distribution<double> noise;
    std::uniform_real_distribution<double> amplitude(0.5, 1.5);
    p.residual.resize(static_cast<std::size_t>(h) * t);
    std::vector<double> x(h);
    for (int m = 0; m < t; ++m) {
        const auto sign = [&] {
            return (spec.mixed_signs && m > 0 && (rng() & 1) != 0) ? -1.0 : 1.0;
        };
        const double a = sign() * spec.alpha * amplitude(rng);
        const double b = sign() * spec.beta * amplitude(rng);
        for (int i = 0; i < h; ++i) { x[i] = noise(rng) + a * unit0[i] + b * unit1[i]; }
        if (spec.massive) {
            x[top] += sign() * 1.0e4 * amplitude(rng);
            x[bottom] = -x[top] * u0[top] / u0[bottom];
            x[third] += sign() * 3.0e3 * amplitude(rng);
        }
        std::uint16_t* words = p.residual.data() + static_cast<std::size_t>(m) * h;
        for (int i = 0; i < h; ++i) { words[i] = f32_to_bf16(static_cast<float>(x[i])); }
        if (spec.specials) {
            const std::array<std::uint16_t, 8> special{0x8000,
                                                       0x0000,
                                                       0x0001,
                                                       0x8001,
                                                       0x0080,
                                                       0x8080,
                                                       f32_to_bf16(8.0f),
                                                       f32_to_bf16(-8.0f)};
            std::copy(special.begin(), special.end(), words);
        }
    }

    p.references[0] = reference_for(spec.first, p, false);
    p.references[1] = reference_for(spec.second, p, true);
    return p;
}

// ---- device operands --------------------------------------------------------------------------

template <class T>
std::vector<std::uint8_t> bytes_of(const std::vector<T>& values) {
    const auto* first = reinterpret_cast<const std::uint8_t*>(values.data());
    return std::vector<std::uint8_t>(first, first + values.size() * sizeof(T));
}

class DeviceProblem {
public:
    DeviceProblem(int h, int t)
        : h_(h), t_(t),
          residual_(static_cast<std::size_t>(h) * std::max(t, 1) * sizeof(std::uint16_t)),
          directions_(static_cast<std::size_t>(h) * 2 * kSlots * sizeof(float)),
          references_(static_cast<std::size_t>(2) * kSlots * sizeof(float)) {}

    void upload(const Problem& p) {
        directions_table_.assign(static_cast<std::size_t>(h_) * 2 * kSlots, 0.0f);
        references_table_.assign(static_cast<std::size_t>(2) * kSlots, 0.0f);
        for (std::size_t i = 0; i < directions_table_.size(); ++i) {
            directions_table_[i] = 1.0f + 0.001f * static_cast<float>(i % 97);
        }
        for (std::size_t i = 0; i < references_table_.size(); ++i) {
            references_table_[i] = 3.0f + static_cast<float>(i);
        }
        std::copy(p.directions.begin(), p.directions.end(),
                  directions_table_.begin() + static_cast<std::size_t>(kSlot) * h_ * 2);
        std::copy(p.references.begin(), p.references.end(),
                  references_table_.begin() + static_cast<std::size_t>(kSlot) * 2);
        directions_.copy_from_host(directions_table_.data(),
                                   directions_table_.size() * sizeof(float));
        references_.copy_from_host(references_table_.data(),
                                   references_table_.size() * sizeof(float));
        if (!p.residual.empty()) {
            residual_.copy_from_host(p.residual.data(), p.residual.size() * sizeof(std::uint16_t));
        }
    }

    Tensor directions() {
        const std::size_t bytes = static_cast<std::size_t>(h_) * 2 * sizeof(float);
        return Tensor(directions_.span().subspan(kSlot * bytes, bytes), DType::FP32, {h_, 2});
    }

    Tensor references() {
        const std::size_t bytes = 2 * sizeof(float);
        return Tensor(references_.span().subspan(kSlot * bytes, bytes), DType::FP32, {2});
    }

    Tensor residual() { return Tensor(residual_.span(), DType::BF16, {h_, t_}); }

    const void* residual_data() const { return residual_.data(); }

    // Guards of every buffer and every byte of both tables, including the other slots.
    int verify_operands(const std::string& label) const {
        int failures = residual_.verify_guards(label + " residual");
        failures += directions_.verify_guards(label + " directions");
        failures += references_.verify_guards(label + " references");
        failures += verify_exact(
            (label + " directions preserved").c_str(),
            from_device<std::uint8_t>(directions_.data(), directions_table_.size() * sizeof(float)),
            bytes_of(directions_table_));
        failures += verify_exact(
            (label + " references preserved").c_str(),
            from_device<std::uint8_t>(references_.data(), references_table_.size() * sizeof(float)),
            bytes_of(references_table_));
        return failures;
    }

private:
    int h_;
    int t_;
    GuardedDeviceBuffer residual_;
    GuardedDeviceBuffer directions_;
    GuardedDeviceBuffer references_;
    std::vector<float> directions_table_;
    std::vector<float> references_table_;
};

// Compares the device residual with the oracle and checks every other observable effect.
int check(const std::string& label, const Problem& p, const DeviceProblem& device,
          const Evaluation& e) {
    int failures            = 0;
    const std::size_t count = static_cast<std::size_t>(p.h) * p.t;
    if (e.error_scale > kMaxErrorScale) {
        std::cerr << label << ": recipe leaves the criterion's error domain (" << e.error_scale
                  << " > " << kMaxErrorScale << ")\n";
        ++failures;
    }

    const auto words = from_device<std::uint16_t>(device.residual_data(), count);
    std::vector<double> got(count);
    std::transform(words.begin(), words.end(), got.begin(),
                   [](std::uint16_t word) { return static_cast<double>(bf16_to_f32(word)); });
    failures += verify_pointwise(label, got, e.values, clipped_direction_removal_criterion());

    std::vector<std::uint16_t> actual_unchanged, expected_unchanged;
    for (int m = 0; m < p.t; ++m) {
        if (!e.unchanged[m]) { continue; }
        const auto offset = static_cast<std::ptrdiff_t>(m) * p.h;
        actual_unchanged.insert(actual_unchanged.end(), words.begin() + offset,
                                words.begin() + offset + p.h);
        expected_unchanged.insert(expected_unchanged.end(), p.residual.begin() + offset,
                                  p.residual.begin() + offset + p.h);
    }
    failures +=
        verify_exact((label + " unchanged columns").c_str(), actual_unchanged, expected_unchanged);
    failures += device.verify_operands(label);
    return failures;
}

// ---- cases ------------------------------------------------------------------------------------

std::string case_id(std::string_view group, int h, int t) {
    return "clipped_direction_removal/" + std::string(group) + "/h" + std::to_string(h) + "/t" +
           std::to_string(t);
}

int run_case(const std::string& id, const Recipe& spec, int h, int t) {
    const Problem p =
        make_problem(spec, h, t, stable_seed(id + "/directions"), stable_seed(id + "/columns"));
    const Evaluation e = evaluate(p);
    if (spec.first.kind == Reference::Kind::Inactive &&
        spec.second.kind == Reference::Kind::Inactive &&
        std::count(e.unchanged.begin(), e.unchanged.end(), 1) != t) {
        std::cerr << id << ": recipe did not leave every column unchanged\n";
        return 1;
    }

    DeviceProblem device(h, t);
    device.upload(p);
    Tensor directions = device.directions();
    Tensor references = device.references();
    Tensor residual   = device.residual();
    ops::clipped_direction_removal(directions, references, residual,
                                   cuda_environment().default_execution);
    cuda_synchronize();
    return check(id, p, device, e);
}

// T=0: no device work. The one-column payload keeps the input, which a processed column 0 would
// change (the first column of the mixed recipe is active at the first step).
int run_empty(const std::string& id, int h) {
    const Problem p = make_problem(regime_recipe(Regime::Mixed), h, 1,
                                   stable_seed(id + "/directions"), stable_seed(id + "/columns"));
    DeviceProblem device(h, 1);
    device.upload(p);
    Tensor directions = device.directions();
    Tensor references = device.references();
    Tensor residual   = device.residual();
    residual.ne[1]    = 0;
    ops::clipped_direction_removal(directions, references, residual,
                                   cuda_environment().default_execution);
    cuda_synchronize();
    int failures = verify_exact(
        (id + " residual").c_str(),
        from_device<std::uint16_t>(device.residual_data(), p.residual.size()), p.residual);
    failures += device.verify_operands(id);
    return failures;
}

// Capture once; every replay reads the current contents at the captured addresses. Phases change
// the residual alone, then residual + directions + references with active clips, then
// everything with both clips inactive (bit-identical output).
int run_graph(const std::string& id, int h, int t) {
    struct Phase {
        Regime regime;
        int directions;
        int columns;
    };

    const std::array<Phase, 4> phases{{{Regime::Mixed, 0, 0},
                                       {Regime::Mixed, 0, 1},
                                       {Regime::Both, 1, 2},
                                       {Regime::Neither, 2, 3}}};
    const auto problem = [&](const Phase& phase) {
        return make_problem(regime_recipe(phase.regime), h, t,
                            stable_seed(id + "/directions" + std::to_string(phase.directions)),
                            stable_seed(id + "/columns" + std::to_string(phase.columns)));
    };

    const CudaExecutionView execution = cuda_environment().context.execution_view();
    DeviceProblem device(h, t);
    Problem current = problem(phases[0]);
    device.upload(current);
    Tensor directions = device.directions();
    Tensor references = device.references();
    Tensor residual   = device.residual();

    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    definition.capture(execution.stream(), [&] {
        ops::clipped_direction_removal(directions, references, residual, execution);
    });
    graph.instantiate(definition);

    int failures = 0;
    for (std::size_t phase = 0; phase < phases.size(); ++phase) {
        if (phase != 0) {
            current = problem(phases[phase]);
            device.upload(current);
        }
        graph.launch(execution.stream());
        cuda_synchronize(execution.stream());
        failures +=
            check(id + " phase " + std::to_string(phase), current, device, evaluate(current));
    }
    return failures;
}

// ---- operand contract -------------------------------------------------------------------------

struct ContractFixture {
    static constexpr int kH = 64;
    static constexpr int kT = 4;

    CudaExecutionView execution = cuda_environment().default_execution;
    Problem problem = make_problem(regime_recipe(Regime::Mixed), kH, kT,
                                   stable_seed("clipped_direction_removal/contract/directions"),
                                   stable_seed("clipped_direction_removal/contract/columns"));
    DeviceProblem device{kH, kT};

    ContractFixture() { device.upload(problem); }

    // Every rejection is std::invalid_argument naming the Op; nothing may reach the device.
    int reject(const char* what, const Tensor& directions, const Tensor& references,
               Tensor residual) const {
        try {
            ops::clipped_direction_removal(directions, references, residual, execution);
        } catch (const std::invalid_argument& error) {
            if (std::string_view(error.what()).starts_with("clipped_direction_removal:")) {
                return 0;
            }
            std::cerr << what << ": message lacks the Op prefix: " << error.what() << '\n';
            return 1;
        } catch (const std::exception& error) {
            std::cerr << what << ": unexpected exception: " << error.what() << '\n';
            return 1;
        }
        std::cerr << what << ": operands were accepted\n";
        return 1;
    }
};

template <class Change>
Tensor modified(Tensor tensor, Change&& change) {
    change(tensor);
    return tensor;
}

Tensor rebound(const Tensor& source, std::initializer_list<std::int32_t> shape) {
    Tensor tensor = Tensor::metadata(source.dtype, shape);
    tensor.data   = source.data;
    tensor.device = source.device;
    return tensor;
}

void shift_base(Tensor& tensor, std::ptrdiff_t bytes) {
    tensor.data = static_cast<std::byte*>(tensor.data) + bytes;
}

int validate_rejections() {
    ContractFixture f;
    const std::size_t count = static_cast<std::size_t>(f.problem.h) * f.problem.t;
    const Tensor directions = f.device.directions();
    const Tensor references = f.device.references();
    const Tensor residual   = f.device.residual();
    const auto reject = [&](const char* what, const Tensor& d, const Tensor& r, const Tensor& x) {
        return f.reject(what, d, r, x);
    };

    // The fixture itself is a valid call; every rejection below changes exactly one property.
    int failures = 0;
    {
        Tensor valid = residual;
        ops::clipped_direction_removal(directions, references, valid, f.execution);
        cuda_synchronize();
        failures += check("contract valid call", f.problem, f.device, evaluate(f.problem));
    }
    const auto before = from_device<std::uint16_t>(f.device.residual_data(), count);

    failures += reject("residual dtype", directions, references,
                       modified(residual, [](Tensor& t) { t.dtype = DType::FP32; }));
    failures +=
        reject("directions dtype", modified(directions, [](Tensor& t) { t.dtype = DType::BF16; }),
               references, residual);
    failures += reject("references dtype", directions,
                       modified(references, [](Tensor& t) { t.dtype = DType::BF16; }), residual);

    failures += reject("residual rank", directions, references,
                       modified(residual, [](Tensor& t) { t.ne[2] = 2; }));
    failures += reject("residual negative columns", directions, references,
                       modified(residual, [](Tensor& t) { t.ne[1] = -1; }));
    failures += reject("directions columns", modified(directions, [](Tensor& t) { t.ne[1] = 1; }),
                       references, residual);
    failures += reject("directions channels", modified(directions, [](Tensor& t) { t.ne[0] /= 2; }),
                       references, residual);
    failures += reject("references extent", directions,
                       modified(references, [](Tensor& t) { t.ne[0] = 3; }), residual);
    failures += reject("references scalar", directions,
                       modified(references, [](Tensor& t) { t.ne[0] = 1; }), residual);

    failures += reject("residual strided", directions, references,
                       modified(residual, [](Tensor& t) { t.nb[1] += 16; }));
    failures += reject("directions strided", modified(directions, [](Tensor& t) { t.nb[1] += 16; }),
                       references, residual);
    failures += reject("references strided", directions,
                       modified(references, [](Tensor& t) { t.nb[0] = 8; }), residual);

    failures += reject("H not a multiple of 8", rebound(directions, {12, 2}), references,
                       rebound(residual, {12, 4}));
    failures +=
        reject("H below 8", rebound(directions, {4, 2}), references, rebound(residual, {4, 4}));
    {
        GuardedDeviceBuffer wide_directions(std::size_t{16392} * 2 * sizeof(float));
        GuardedDeviceBuffer wide_residual(std::size_t{16392} * sizeof(std::uint16_t));
        failures += reject("H above 16384", Tensor(wide_directions.span(), DType::FP32, {16392, 2}),
                           references, Tensor(wide_residual.span(), DType::BF16, {16392, 1}));
    }

    failures += reject("directions overlap residual start",
                       modified(directions, [&](Tensor& t) { t.data = residual.data; }), references,
                       residual);
    failures += reject("directions overlap residual interior",
                       modified(directions,
                                [&](Tensor& t) {
                                    t.data = residual.data;
                                    shift_base(t, 256);
                                }),
                       references, residual);
    failures += reject("residual overlaps directions", directions, references,
                       modified(residual, [&](Tensor& t) { t.data = directions.data; }));
    failures += reject("references overlap residual start", directions,
                       modified(references, [&](Tensor& t) { t.data = residual.data; }), residual);
    failures += reject("references overlap residual tail", directions,
                       modified(references,
                                [&](Tensor& t) {
                                    t.data = residual.data;
                                    shift_base(t, 504);
                                }),
                       residual);

    failures += reject("residual misaligned by 2", directions, references,
                       modified(residual, [](Tensor& t) { shift_base(t, 2); }));
    failures += reject("residual misaligned by 8", directions, references,
                       modified(residual, [](Tensor& t) { shift_base(t, 8); }));
    failures +=
        reject("directions misaligned by 4",
               modified(directions, [](Tensor& t) { shift_base(t, 4); }), references, residual);
    failures +=
        reject("directions misaligned by 8",
               modified(directions, [](Tensor& t) { shift_base(t, 8); }), references, residual);
    failures += reject("references misaligned by 4", directions,
                       modified(references, [](Tensor& t) { shift_base(t, 4); }), residual);

    failures += reject("residual on another device", directions, references,
                       modified(residual, [](Tensor& t) { t.device = Device::cpu(); }));
    failures += reject("directions on another device",
                       modified(directions, [](Tensor& t) { t.device = Device::cpu(); }),
                       references, residual);
    failures += reject("references without a device", directions,
                       modified(references, [](Tensor& t) { t.device = Device(); }), residual);

    failures += reject("residual null", directions, references,
                       modified(residual, [](Tensor& t) { t.data = nullptr; }));
    failures += reject("directions null", modified(directions, [](Tensor& t) { t.data = nullptr; }),
                       references, residual);
    failures += reject("references null", directions,
                       modified(references, [](Tensor& t) { t.data = nullptr; }), residual);

    failures += verify_exact("contract residual untouched by rejections",
                             from_device<std::uint16_t>(f.device.residual_data(), count), before);
    failures += f.device.verify_operands("contract");
    return failures;
}

// T=0 is a no-op that still validates the remaining operands and the residual's metadata.
int validate_empty() {
    ContractFixture f;
    const Tensor directions = f.device.directions();
    const Tensor references = f.device.references();
    Tensor empty            = Tensor::metadata(DType::BF16, {ContractFixture::kH, 1});
    empty.ne[1]             = 0;
    const auto reject = [&](const char* what, const Tensor& d, const Tensor& r, const Tensor& x) {
        return f.reject(what, d, r, x);
    };

    int failures = 0;
    try {
        Tensor accepted = empty;
        ops::clipped_direction_removal(directions, references, accepted, f.execution);
    } catch (const std::exception& error) {
        std::cerr << "empty residual rejected: " << error.what() << '\n';
        ++failures;
    }
    failures += reject("empty with H not a multiple of 8", rebound(directions, {12, 2}), references,
                       modified(empty, [](Tensor& t) { t.ne[0] = 12; }));
    failures += reject("empty with residual dtype", directions, references,
                       modified(empty, [](Tensor& t) { t.dtype = DType::FP32; }));
    failures += reject("empty with residual rank", directions, references,
                       modified(empty, [](Tensor& t) { t.ne[3] = 2; }));
    failures +=
        reject("empty with directions dtype",
               modified(directions, [](Tensor& t) { t.dtype = DType::BF16; }), references, empty);
    failures +=
        reject("empty with misaligned directions",
               modified(directions, [](Tensor& t) { shift_base(t, 4); }), references, empty);
    failures += reject("empty with references extent", directions,
                       modified(references, [](Tensor& t) { t.ne[0] = 3; }), empty);
    failures += reject("empty with directions on another device",
                       modified(directions, [](Tensor& t) { t.device = Device::cpu(); }),
                       references, empty);
    failures += f.device.verify_operands("contract empty");
    return failures;
}

} // namespace

void add_clipped_direction_removal_cases(OpSuite& suite) {
    const auto add = [&](Regime regime, int h, int t) {
        const std::string id = case_id(regime_recipe(regime).name, h, t);
        suite.add(id, [=] { return run_case(id, regime_recipe(regime), h, t); },
                  {{"execution", "eager"}});
    };
    const auto add_graph = [&](int h, int t) {
        const std::string id = case_id("graph", h, t);
        suite.add(id, [=] { return run_graph(id, h, t); }, {{"execution", "graph"}});
    };

    // Real Qwen geometry across the token extents.
    for (const int t : {1, 2, 7, 16, 33, 64, 1023, 1024}) { add(Regime::Mixed, 5120, t); }

    // Hidden sizes across the supported range, with token extents that leave tails.
    add(Regime::Both, 8, 1);
    add(Regime::Both, 8, 33);
    add(Regime::Both, 136, 7);
    add(Regime::Both, 136, 64);
    add(Regime::Both, 2048, 16);
    add(Regime::Both, 2048, 1023);
    add(Regime::Both, 4096, 2);
    add(Regime::Both, 4096, 1024);
    add(Regime::Both, 16384, 1);
    add(Regime::Both, 16384, 33);

    // Clipping regimes, directions and references at the real hidden size.
    for (const Regime regime :
         {Regime::Both, Regime::FirstOnly, Regime::SecondOnly, Regime::Neither,
          Regime::ZeroNegative, Regime::ZeroPositive, Regime::NegativeReferences,
          Regime::LargeReferences, Regime::Massive, Regime::NonUnit, Regime::Sequential,
          Regime::RoundTrap}) {
        add(regime, 5120, 33);
    }

    // The same properties at the smallest, a tail-only and the largest hidden size.
    add(Regime::Neither, 8, 16);
    add(Regime::Neither, 16384, 2);
    add(Regime::Sequential, 136, 7);
    add(Regime::RoundTrap, 136, 7);
    add(Regime::RoundTrap, 16384, 3);
    add(Regime::Massive, 16384, 3);

    for (const int h : {8, 5120}) {
        const std::string id = case_id("empty", h, 0);
        suite.add(id, [=] { return run_empty(id, h); }, {{"execution", "eager"}});
    }

    add_graph(5120, 1);
    add_graph(5120, 64);
    add_graph(136, 7);

    suite.add("clipped_direction_removal/contract/rejections", validate_rejections,
              {{"execution", "eager"}});
    suite.add("clipped_direction_removal/contract/empty", validate_empty, {{"execution", "eager"}});
}
