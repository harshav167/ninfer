#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/distributed_attention.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <memory>
#include "core/decode_graph.h"
#include "core/device.h"
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim          = 256;
constexpr std::int32_t kQuantGroup       = 64;
constexpr std::int32_t kQuantGroups      = kHeadDim / kQuantGroup;
constexpr std::int32_t kFp8QuantGroup    = kHeadDim;
constexpr std::int32_t kFp8QuantGroups   = 1;
constexpr std::int32_t kNvfp4QuantGroup  = 16;
constexpr std::int32_t kNvfp4QuantGroups = kHeadDim / kNvfp4QuantGroup;
constexpr std::int32_t kNvfp4CodeBytes   = kHeadDim / 2;
constexpr float kAttentionScale          = 0.0625f;
constexpr std::uint16_t kOutputCanary    = 0x7fc1u;

// A1 and A3 use one fixed criterion for each registered storage profile; token count, geometry,
// execution envelope, and private launch route do not select or relax it.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ 3.0e-3,
};

constexpr ReductionCriterion kAttentionFp8Criterion{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 9.0e-3,
};

constexpr ReductionCriterion kAttentionNvfp4Criterion{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-2,
};

constexpr ReductionCriterion kAttentionK8V4Criterion{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-2,
};

struct TestVectorLayout {
    DType code_dtype;
    std::int32_t code_extent;
    DType scale_dtype;
    std::int32_t scale_extent;
};

struct TestCacheLayout {
    TestVectorLayout key;
    TestVectorLayout value;
};

TestCacheLayout test_cache_layout(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {{DType::BF16, kHeadDim, DType::U8, 0}, {DType::FP16, kHeadDim, DType::U8, 0}};
    case KvCacheStorage::Int8Group64:
        return {{DType::I8, kHeadDim, DType::FP16, kQuantGroups},
                {DType::I8, kHeadDim, DType::FP16, kQuantGroups}};
    case KvCacheStorage::Fp8E4M3Row256:
        return {{DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups},
                {DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups}};
    case KvCacheStorage::Nvfp4Group16:
        return {{DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups},
                {DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups}};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {{DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups},
                {DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups}};
    }
    throw std::invalid_argument("unsupported test KV storage");
}

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

constexpr Geometry kGeometries[] = {
    {"d256-h24-kv4", 24, 4},
    {"d256-h16-kv2", 16, 2},
};

ops::AttentionHeadGeometry op_geometry(const Geometry& geometry) {
    return {kHeadDim, geometry.q_heads, geometry.kv_heads};
}

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
    bool zero_q       = false;
    bool graph_replay = false;
};

enum class MappingPattern { Identity, Offset, Fragmented, Remote };

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    case MappingPattern::Remote:
        return "remote";
    }
    return "unknown";
}

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    case MappingPattern::Remote:
        // Every logical page lands in the secondary plane, so the table must reach one page past
        // the primary plane's highest ID. The unused hole at physical ID 0 stays populated and is
        // itself evidence that unmapped physical slots are never observed.
        return logical_pages + 2;
    }
    return 0;
}

std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    switch (pattern) {
    case MappingPattern::Identity:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page; }
        break;
    case MappingPattern::Offset:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page + 1; }
        break;
    case MappingPattern::Fragmented:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = 2 * page + 1; }
        break;
    case MappingPattern::Remote:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page + 2; }
        break;
    }
    return table;
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t logical_plane_index(std::int32_t leading_extent, const Geometry& geometry,
                                std::int32_t padded_context, std::int32_t head,
                                std::int32_t position, std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t fp8_scale_index(const Geometry& geometry, std::int32_t padded_context,
                            std::int32_t head, std::int32_t position) {
    (void)geometry;
    return static_cast<std::size_t>(position) +
           static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head);
}

std::size_t fp8_scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::size_t>(physical_page)));
}

std::size_t physical_plane_elements(std::int32_t leading_extent, const Geometry& geometry,
                                    std::int32_t physical_pages) {
    return static_cast<std::size_t>(leading_extent) * kPagedKVPageSize * geometry.kv_heads *
           physical_pages;
}

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity,
                             std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages) {
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(physical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

template <typename T>
void scatter_paged_into(const std::vector<T>& logical, std::int32_t leading_extent,
                        const Geometry& geometry, std::int32_t logical_capacity,
                        std::span<const std::int32_t> block_table, std::vector<T>& physical) {
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
}

template <typename T>
std::vector<T> gather_paged(std::span<const T> physical, std::int32_t leading_extent,
                            const Geometry& geometry, std::int32_t logical_capacity,
                            std::span<const std::int32_t> block_table) {
    std::vector<T> logical(static_cast<std::size_t>(leading_extent) * logical_capacity *
                           static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t target = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                logical[target] =
                    physical[paged_index(leading_extent, geometry, page, head, position, leading)];
            }
        }
    }
    return logical;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

float round_to_f16(float value) { return f16_bits_to_f32(f32_to_f16_bits(value)); }

std::vector<std::uint16_t> to_f16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_f16_bits(values[i]); }
    return bits;
}

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

float decode_e4m3fn_positive(std::uint8_t code) {
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if (exponent == 0) return std::ldexp(static_cast<float>(mantissa), -9);
    return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

float decode_e4m3fn(std::uint8_t code) {
    const float magnitude = decode_e4m3fn_positive(code & 0x7fU);
    return (code & 0x80U) == 0 ? magnitude : -magnitude;
}

std::uint8_t encode_e4m3fn_rne_satfinite(float value) {
    static const auto codebook = [] {
        std::array<float, 127> values{};
        for (int code = 0; code < 127; ++code) values[code] = decode_e4m3fn_positive(code);
        return values;
    }();
    const bool negative   = std::signbit(value);
    const float magnitude = std::abs(value);
    int selected          = 126;
    if (magnitude < 448.0f) {
        const int upper =
            std::lower_bound(codebook.begin(), codebook.end(), magnitude) - codebook.begin();
        if (upper == 0)
            selected = 0;
        else {
            const int lower        = upper - 1;
            const float low_error  = magnitude - codebook[lower],
                        high_error = codebook[upper] - magnitude;
            selected               = low_error < high_error   ? lower
                                     : high_error < low_error ? upper
                                                              : (lower % 2 == 0 ? lower : upper);
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 128 : 0));
}

float decode_e2m1(std::uint8_t code) {
    constexpr std::array<float, 8> magnitude{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = magnitude.at(static_cast<std::size_t>(code & 0x07U));
    return (code & 0x08U) == 0 ? value : -value;
}

std::uint8_t encode_e2m1_rne_satfinite(float value) {
    const bool negative   = std::signbit(value);
    const float magnitude = std::abs(value);
    std::uint8_t selected = 7;
    if (magnitude < 6.0f) {
        for (std::uint8_t upper = 1; upper <= 7; ++upper) {
            const float upper_value = decode_e2m1(upper);
            if (upper_value < magnitude) continue;
            const std::uint8_t lower = static_cast<std::uint8_t>(upper - 1);
            const float lower_value  = decode_e2m1(lower);
            const float lower_error  = magnitude - lower_value;
            const float upper_error  = upper_value - magnitude;
            selected = lower_error < upper_error   ? lower
                       : upper_error < lower_error ? upper
                                                   : ((lower & 1U) == 0U ? lower : upper);
            break;
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 0x08U : 0U));
}

struct HostCache {
    Geometry geometry;
    KvCacheStorage storage;
    std::int32_t max_context;
    std::int32_t logical_capacity;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_fp16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint8_t> k_fp8;
    std::vector<std::uint8_t> v_fp8;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
    std::vector<std::uint8_t> k_nvfp4;
    std::vector<std::uint8_t> v_nvfp4;
    std::vector<std::uint8_t> k_nvfp4_scale;
    std::vector<std::uint8_t> v_nvfp4_scale;
};

void encode_group(std::span<const float> source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const float unrounded_scale    = absmax / 127.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -127, 127);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

void normalized_hadamard_d256(std::array<float, kHeadDim>& values) {
    for (std::int32_t stride = 1; stride < kHeadDim; stride *= 2) {
        for (std::int32_t base = 0; base < kHeadDim; base += 2 * stride) {
            for (std::int32_t offset = 0; offset < stride; ++offset) {
                const float low  = values[static_cast<std::size_t>(base + offset)];
                const float high = values[static_cast<std::size_t>(base + offset + stride)];
                values[static_cast<std::size_t>(base + offset)]          = low + high;
                values[static_cast<std::size_t>(base + offset + stride)] = low - high;
            }
        }
    }
    for (float& value : values) { value *= 0x1p-4f; }
}

void normalized_hadamard_d256(std::array<double, kHeadDim>& values) {
    static const auto matrix = [] {
        std::array<std::array<double, kHeadDim>, kHeadDim> result{};
        for (unsigned row = 0; row < kHeadDim; ++row)
            for (unsigned column = 0; column < kHeadDim; ++column)
                result[row][column] = (std::popcount(row & column) & 1) ? -0.0625 : 0.0625;
        return result;
    }();
    const auto input = values;
    for (int row = 0; row < kHeadDim; ++row) {
        double sum = 0.0;
        for (int column = 0; column < kHeadDim; ++column)
            sum += matrix[row][column] * input[column];
        values[row] = sum;
    }
}

void encode_nvfp4_rotated_row(std::span<const float> source, std::size_t source_base,
                              std::vector<std::uint8_t>& codes, std::size_t code_base,
                              std::vector<std::uint8_t>& scales, std::size_t scale_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    for (std::int32_t group = 0; group < kNvfp4QuantGroups; ++group) {
        const std::int32_t d0 = group * kNvfp4QuantGroup;
        float absmax          = 0.0f;
        for (std::int32_t i = 0; i < kNvfp4QuantGroup; ++i) {
            absmax = std::max(absmax, std::abs(rotated[static_cast<std::size_t>(d0 + i)]));
        }
        std::uint8_t scale_code = 0;
        float represented_scale = 0.0f;
        if (absmax != 0.0f) {
            const float bounded = std::clamp(absmax / 6.0f, std::ldexp(1.0f, -9), 448.0f);
            scale_code          = encode_e4m3fn_rne_satfinite(bounded);
            represented_scale   = decode_e4m3fn(scale_code);
        }
        scales[scale_base + static_cast<std::size_t>(group)] = scale_code;
        for (std::int32_t pair = 0; pair < kNvfp4QuantGroup / 2; ++pair) {
            const std::int32_t low_d  = d0 + 2 * pair;
            const std::int32_t high_d = low_d + 1;
            const std::uint8_t low =
                represented_scale == 0.0f
                    ? 0
                    : encode_e2m1_rne_satfinite(rotated[static_cast<std::size_t>(low_d)] /
                                                represented_scale);
            const std::uint8_t high =
                represented_scale == 0.0f
                    ? 0
                    : encode_e2m1_rne_satfinite(rotated[static_cast<std::size_t>(high_d)] /
                                                represented_scale);
            codes[code_base + static_cast<std::size_t>(group * 8 + pair)] =
                static_cast<std::uint8_t>(low | (high << 4));
        }
    }
}

void encode_fp8_row(const std::array<float, kHeadDim>& values, std::vector<std::uint8_t>& codes,
                    std::size_t code_base, std::vector<std::uint16_t>& scales,
                    std::size_t scale_offset) {
    float absmax = 0.0f;
    for (const float value : values) absmax = std::max(absmax, std::abs(value));
    std::uint16_t scale_bits = 0;
    float represented_scale  = 0.0f;
    if (absmax != 0.0f) {
        const float raw_scale = absmax / 448.0f;
        scale_bits        = f32_to_f16_bits(std::clamp(raw_scale, std::ldexp(1.0f, -24), 65504.0f));
        represented_scale = f16_bits_to_f32(scale_bits);
    }
    scales[scale_offset]   = scale_bits;
    const float reciprocal = represented_scale == 0.0f ? 0.0f : 1.0f / represented_scale;
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        codes[code_base + static_cast<std::size_t>(d)] =
            represented_scale == 0.0f
                ? 0
                : encode_e4m3fn_rne_satfinite(values[static_cast<std::size_t>(d)] * reciprocal);
    }
}

void encode_fp8_rotated_row(std::span<const float> source, std::size_t source_base,
                            std::vector<std::uint8_t>& codes, std::size_t code_base,
                            std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    encode_fp8_row(rotated, codes, code_base, scales, scale_offset);
}

void encode_rotated_key_row(std::span<const float> source, std::size_t source_base,
                            std::vector<std::int8_t>& codes, std::size_t code_base,
                            std::vector<std::uint16_t>& scales, std::size_t scale_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    for (std::int32_t group = 0; group < kQuantGroups; ++group) {
        const std::size_t d = static_cast<std::size_t>(group * kQuantGroup);
        encode_group(rotated, d, codes, code_base + d, scales,
                     scale_base + static_cast<std::size_t>(group));
    }
}

// Logical key magnitude per position, applied before the cache codec runs so a scaled case still
// stores the represented values read by the independent oracle. The scale-boundary discriminator
// uses one positive all-ones key vector on each side, all-ones queries, and constant zero/one value
// vectors. After the FP8 key transform each side therefore has a single score for every key; a
// large cross-shard LSE gap cannot be changed by private query rounding or a near-tied argmax.
//
// Both factors are powers of two, so they move values along the BF16 grid without changing which
// grid points are representable.
struct KeyMagnitude {
    std::int32_t boundary = 0;
    float low             = 1.0f;
    float high            = 1.0f;
    bool uniform_regions  = false;

    [[nodiscard]] float at(std::int32_t position) const { return position < boundary ? low : high; }
};

// The unscaled logical key plane used by the ordinary randomized cache fixtures.
std::vector<float> logical_key_plane(const Geometry& geometry, std::int32_t logical_capacity,
                                     std::uint32_t seed) {
    return make_bf16_values(cache_elements(geometry, logical_capacity), seed, -0.25f, 0.25f);
}

HostCache make_cache(const Geometry& geometry, KvCacheStorage storage, std::int32_t max_context,
                     std::uint32_t seed, const KeyMagnitude* magnitude = nullptr) {
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    const bool uniform_regions          = magnitude != nullptr && magnitude->uniform_regions;
    std::vector<float> logical_k        = uniform_regions
                                              ? std::vector<float>(elements, 1.0f)
                                              : logical_key_plane(geometry, logical_capacity, seed);
    std::vector<float> logical_v        = uniform_regions
                                              ? std::vector<float>(elements, 0.0f)
                                              : make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    if (magnitude != nullptr) {
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::int32_t position = 0; position < logical_capacity; ++position) {
                const float factor = magnitude->at(position);
                if (factor == 1.0f && !uniform_regions) continue;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t index =
                        cache_index(geometry, logical_capacity, head, position, d);
                    if (uniform_regions) {
                        logical_k[index] = factor;
                        logical_v[index] = position < magnitude->boundary ? 0.0f : 1.0f;
                    } else {
                        logical_k[index] *= factor;
                    }
                }
            }
        }
    }
    HostCache cache{geometry, storage, max_context, logical_capacity};
    if (storage == KvCacheStorage::BFloat16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_fp16 = to_f16_bits(logical_v);
        return cache;
    }

    if (storage == KvCacheStorage::Nvfp4Group16) {
        const std::size_t code_elements =
            static_cast<std::size_t>(kNvfp4CodeBytes) * logical_capacity * geometry.kv_heads;
        const std::size_t scale_elements =
            static_cast<std::size_t>(kNvfp4QuantGroups) * logical_capacity * geometry.kv_heads;
        cache.k_nvfp4.assign(code_elements, 0);
        cache.v_nvfp4.assign(code_elements, 0);
        cache.k_nvfp4_scale.assign(scale_elements, 0);
        cache.v_nvfp4_scale.assign(scale_elements, 0);
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::int32_t position = 0; position < logical_capacity; ++position) {
                const std::size_t source =
                    cache_index(geometry, logical_capacity, head, position, 0);
                const std::size_t code  = logical_plane_index(kNvfp4CodeBytes, geometry,
                                                              logical_capacity, head, position, 0);
                const std::size_t scale = logical_plane_index(kNvfp4QuantGroups, geometry,
                                                              logical_capacity, head, position, 0);
                encode_nvfp4_rotated_row(logical_k, source, cache.k_nvfp4, code,
                                         cache.k_nvfp4_scale, scale);
                encode_nvfp4_rotated_row(logical_v, source, cache.v_nvfp4, code,
                                         cache.v_nvfp4_scale, scale);
            }
        }
        return cache;
    }

    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        const std::size_t fp8_scales = fp8_scale_elements(geometry, logical_capacity);
        const std::size_t v_codes =
            static_cast<std::size_t>(kNvfp4CodeBytes) * logical_capacity * geometry.kv_heads;
        const std::size_t v_scales =
            static_cast<std::size_t>(kNvfp4QuantGroups) * logical_capacity * geometry.kv_heads;
        cache.k_fp8.assign(elements, 0);
        cache.k_scale.assign(fp8_scales, 0);
        cache.v_nvfp4.assign(v_codes, 0);
        cache.v_nvfp4_scale.assign(v_scales, 0);
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::int32_t position = 0; position < logical_capacity; ++position) {
                const std::size_t source =
                    cache_index(geometry, logical_capacity, head, position, 0);
                const std::size_t k_scale =
                    fp8_scale_index(geometry, logical_capacity, head, position);
                const std::size_t v_code = logical_plane_index(kNvfp4CodeBytes, geometry,
                                                               logical_capacity, head, position, 0);
                const std::size_t v_scale = logical_plane_index(
                    kNvfp4QuantGroups, geometry, logical_capacity, head, position, 0);
                encode_fp8_rotated_row(logical_k, source, cache.k_fp8, source, cache.k_scale,
                                       k_scale);
                encode_nvfp4_rotated_row(logical_v, source, cache.v_nvfp4, v_code,
                                         cache.v_nvfp4_scale, v_scale);
            }
        }
        return cache;
    }

    const std::size_t scales = storage == KvCacheStorage::Int8Group64
                                   ? scale_elements(geometry, logical_capacity)
                                   : fp8_scale_elements(geometry, logical_capacity);
    cache.k_scale.assign(scales, 0);
    cache.v_scale.assign(scales, 0);
    if (storage == KvCacheStorage::Fp8E4M3Row256) {
        cache.k_fp8.assign(elements, 0);
        cache.v_fp8.assign(elements, 0);
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::int32_t position = 0; position < logical_capacity; ++position) {
                const std::size_t code = cache_index(geometry, logical_capacity, head, position, 0);
                const std::size_t scale =
                    fp8_scale_index(geometry, logical_capacity, head, position);
                encode_fp8_rotated_row(logical_k, code, cache.k_fp8, code, cache.k_scale, scale);
                std::array<float, kHeadDim> v_row{};
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    v_row[static_cast<std::size_t>(d)] =
                        logical_v[code + static_cast<std::size_t>(d)];
                }
                encode_fp8_row(v_row, cache.v_fp8, code, cache.v_scale, scale);
            }
        }
        return cache;
    }

    cache.k_i8.assign(elements, 0);
    cache.v_i8.assign(elements, 0);
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::size_t code  = cache_index(geometry, logical_capacity, head, position, 0);
            const std::size_t scale = scale_index(geometry, logical_capacity, head, position, 0);
            encode_rotated_key_row(logical_k, code, cache.k_i8, code, cache.k_scale, scale);
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d = group * kQuantGroup;
                const std::size_t group_code =
                    cache_index(geometry, logical_capacity, head, position, d);
                const std::size_t group_scale =
                    scale_index(geometry, logical_capacity, head, position, group);
                encode_group(logical_v, group_code, cache.v_i8, group_code, cache.v_scale,
                             group_scale);
            }
        }
    }
    return cache;
}

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.storage == KvCacheStorage::BFloat16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                    cache.v_fp16[target] = f32_to_f16_bits(v[source]);
                }
                continue;
            }

            const std::size_t source = kv_input_index(geometry, head, 0, token);
            const std::size_t target =
                cache_index(geometry, cache.logical_capacity, head, position, 0);
            if (cache.storage == KvCacheStorage::Nvfp4Group16) {
                const std::size_t code = logical_plane_index(
                    kNvfp4CodeBytes, geometry, cache.logical_capacity, head, position, 0);
                const std::size_t scale = logical_plane_index(
                    kNvfp4QuantGroups, geometry, cache.logical_capacity, head, position, 0);
                encode_nvfp4_rotated_row(k, source, cache.k_nvfp4, code, cache.k_nvfp4_scale,
                                         scale);
                encode_nvfp4_rotated_row(v, source, cache.v_nvfp4, code, cache.v_nvfp4_scale,
                                         scale);
                continue;
            }
            if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
                const std::size_t k_scale =
                    fp8_scale_index(geometry, cache.logical_capacity, head, position);
                const std::size_t v_code = logical_plane_index(
                    kNvfp4CodeBytes, geometry, cache.logical_capacity, head, position, 0);
                const std::size_t v_scale = logical_plane_index(
                    kNvfp4QuantGroups, geometry, cache.logical_capacity, head, position, 0);
                encode_fp8_rotated_row(k, source, cache.k_fp8, target, cache.k_scale, k_scale);
                encode_nvfp4_rotated_row(v, source, cache.v_nvfp4, v_code, cache.v_nvfp4_scale,
                                         v_scale);
                continue;
            }
            if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
                const std::size_t scale =
                    fp8_scale_index(geometry, cache.logical_capacity, head, position);
                encode_fp8_rotated_row(k, source, cache.k_fp8, target, cache.k_scale, scale);
                std::array<float, kHeadDim> v_row{};
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    v_row[static_cast<std::size_t>(d)] =
                        v[kv_input_index(geometry, head, d, token)];
                }
                encode_fp8_row(v_row, cache.v_fp8, target, cache.v_scale, scale);
                continue;
            }
            const std::size_t scale =
                scale_index(geometry, cache.logical_capacity, head, position, 0);
            encode_rotated_key_row(k, source, cache.k_i8, target, cache.k_scale, scale);
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d           = group * kQuantGroup;
                const std::size_t group_source = kv_input_index(geometry, head, d, token);
                const std::size_t group_target =
                    cache_index(geometry, cache.logical_capacity, head, position, d);
                const std::size_t group_scale =
                    scale_index(geometry, cache.logical_capacity, head, position, group);
                encode_group(v, group_source, cache.v_i8, group_target, cache.v_scale, group_scale);
            }
        }
    }
}

double cache_value(const HostCache& cache, bool key, int head, int position, int d) {
    const std::size_t row   = std::size_t(head) * cache.logical_capacity + position;
    const std::size_t index = row * kHeadDim + d;
    if (cache.storage == KvCacheStorage::BFloat16)
        return key ? double(bf16_to_f32(cache.k_bf16[index]))
                   : double(f16_bits_to_f32(cache.v_fp16[index]));
    if (cache.storage == KvCacheStorage::Int8Group64) {
        const auto& codes  = key ? cache.k_i8 : cache.v_i8;
        const auto& scales = key ? cache.k_scale : cache.v_scale;
        return double(codes[index]) *
               double(f16_bits_to_f32(scales[row * kQuantGroups + d / kQuantGroup]));
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256 ||
        (key && cache.storage == KvCacheStorage::Fp8KeyNvfp4Value)) {
        const auto& codes  = key ? cache.k_fp8 : cache.v_fp8;
        const auto& scales = key ? cache.k_scale : cache.v_scale;
        return double(decode_e4m3fn(codes[index])) * double(f16_bits_to_f32(scales[row]));
    }
    const auto& codes  = key ? cache.k_nvfp4 : cache.v_nvfp4;
    const auto& scales = key ? cache.k_nvfp4_scale : cache.v_nvfp4_scale;
    const auto byte    = codes[row * kNvfp4CodeBytes + d / 2];
    const auto code    = (d & 1) ? byte >> 4 : byte & 15;
    return double(decode_e2m1(code)) *
           double(decode_e4m3fn(scales[row * kNvfp4QuantGroups + d / kNvfp4QuantGroup]));
}

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    const int tokens = positions.size(), visible = positions.back() + 1;
    const bool rotate_q = cache.storage != KvCacheStorage::BFloat16;
    const bool rotate_v = cache.storage == KvCacheStorage::Nvfp4Group16 ||
                          cache.storage == KvCacheStorage::Fp8KeyNvfp4Value;
    std::vector<double> query(q.begin(), q.end()), output(q.size());
    if (rotate_q)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < geometry.q_heads; ++head) {
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d) row[d] = q[q_index(geometry, head, d, token)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d)
                    query[q_index(geometry, head, d, token)] = row[d];
            }
    // Decode the persistent public representation once. No private Q quantization, staging
    // casts, partial rounding or inverse-transform materialization enters this oracle.
    const auto index = [&](int d, int head, int pos) {
        return std::size_t(d) + kHeadDim * (std::size_t(pos) + std::size_t(visible) * head);
    };
    std::vector<double> keys(std::size_t(visible) * geometry.kv_heads * kHeadDim),
        values(keys.size());
    for (int head = 0; head < geometry.kv_heads; ++head)
        for (int pos = 0; pos < visible; ++pos)
            for (int d = 0; d < kHeadDim; ++d) {
                keys[index(d, head, pos)]   = cache_value(cache, true, head, pos, d);
                values[index(d, head, pos)] = cache_value(cache, false, head, pos, d);
            }
    naive_dense_softmax_attention(
        op_geometry(geometry), tokens, visible, double(kAttentionScale),
        [&](int d, int head, int token) { return query[q_index(geometry, head, d, token)]; },
        [&](int d, int head, int pos) { return keys[index(d, head, pos)]; },
        [&](int d, int head, int pos) { return values[index(d, head, pos)]; },
        [&](int token, int pos) { return pos <= positions[token]; },
        [&](int d, int head, int token, double value) {
            output[q_index(geometry, head, d, token)] = value;
        });
    if (rotate_v)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < geometry.q_heads; ++head) {
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d)
                    row[d] = output[q_index(geometry, head, d, token)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d)
                    output[q_index(geometry, head, d, token)] = row[d];
            }
    return output;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

class DeviceCache {
public:
    DeviceCache(const HostCache& cache, MappingPattern mapping)
        : geometry_(cache.geometry), storage_(cache.storage), layout_(test_cache_layout(storage_)),
          max_context_(cache.max_context), logical_capacity_(cache.logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(physical_page_count(logical_pages_, mapping)),
          block_table_host_(make_block_table(logical_pages_, mapping)),
          k_code_elements_(
              physical_plane_elements(layout_.key.code_extent, geometry_, physical_pages_)),
          v_code_elements_(
              physical_plane_elements(layout_.value.code_extent, geometry_, physical_pages_)),
          k_scale_elements_(
              physical_plane_elements(layout_.key.scale_extent, geometry_, physical_pages_)),
          v_scale_elements_(
              physical_plane_elements(layout_.value.scale_extent, geometry_, physical_pages_)),
          k_(k_code_elements_ * dtype_size(layout_.key.code_dtype)),
          v_(v_code_elements_ * dtype_size(layout_.value.code_dtype)),
          k_scale_(layout_.key.scale_extent == 0
                       ? 1
                       : k_scale_elements_ * dtype_size(layout_.key.scale_dtype)),
          v_scale_(layout_.value.scale_extent == 0
                       ? 1
                       : v_scale_elements_ * dtype_size(layout_.value.scale_dtype)),
          block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
        block_table_.copy_from_host(block_table_host_.data(),
                                    block_table_host_.size() * sizeof(std::int32_t));
        if (storage_ == KvCacheStorage::BFloat16) {
            const auto k_physical =
                scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_fp16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
        } else if (storage_ == KvCacheStorage::Int8Group64) {
            const auto k_physical =
                scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
            const auto k_physical =
                scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_fp8, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size());
            v_.copy_from_host(v_physical.data(), v_physical.size());
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            const auto k_physical =
                scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size());
            v_.copy_from_host(v_physical.data(), v_physical.size());
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size());
        } else {
            const auto k_physical =
                scatter_paged(cache.k_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size());
            v_.copy_from_host(v_physical.data(), v_physical.size());
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size());
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size());
        }
    }

    PagedKVLayerView view() {
        PagedKVLayerView result;
        result.k_pages = Tensor(
            k_.data(), layout_.key.code_dtype,
            {layout_.key.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages = Tensor(
            v_.data(), layout_.value.code_dtype,
            {layout_.value.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_table  = Tensor(block_table_.data(), DType::I32, {logical_pages_});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.storage      = storage_;
        if (layout_.key.scale_extent != 0) {
            result.k_scale_pages = Tensor(
                k_scale_.data(), layout_.key.scale_dtype,
                {layout_.key.scale_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        }
        if (layout_.value.scale_extent != 0) {
            result.v_scale_pages = Tensor(v_scale_.data(), layout_.value.scale_dtype,
                                          {layout_.value.scale_extent, kPagedKVPageSize,
                                           geometry_.kv_heads, physical_pages_});
        }
        return result;
    }

    PagedKVBatchLayerView batch_view() {
        const PagedKVLayerView direct = view();
        return {
            .k_pages       = direct.k_pages,
            .v_pages       = direct.v_pages,
            .k_scale_pages = direct.k_scale_pages,
            .v_scale_pages = direct.v_scale_pages,
            .block_tables  = direct.block_table.view({logical_pages_, 1}),
            .head_dim      = direct.head_dim,
            .num_kv_heads  = direct.num_kv_heads,
            .storage       = direct.storage,
        };
    }

    HostCache snapshot() const {
        HostCache cache{geometry_, storage_, max_context_, logical_capacity_};
        if (storage_ == KvCacheStorage::BFloat16) {
            const auto k_physical = copy_from_guarded<std::uint16_t>(k_, k_code_elements_);
            const auto v_physical = copy_from_guarded<std::uint16_t>(v_, v_code_elements_);
            cache.k_bf16 = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.v_fp16 = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                       logical_capacity_, block_table_host_);
        } else if (storage_ == KvCacheStorage::Int8Group64) {
            const auto k_physical  = copy_from_guarded<std::int8_t>(k_, k_code_elements_);
            const auto v_physical  = copy_from_guarded<std::int8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_);
            cache.k_i8    = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_,
                                                      logical_capacity_, block_table_host_);
            cache.v_i8    = gather_paged<std::int8_t>(v_physical, kHeadDim, geometry_,
                                                      logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
        } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_);
            cache.k_fp8   = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.v_fp8   = gather_paged<std::uint8_t>(v_physical, kHeadDim, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kFp8QuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kFp8QuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
        } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_);
            cache.k_fp8   = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.v_nvfp4 = gather_paged<std::uint8_t>(v_physical, kNvfp4CodeBytes, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kFp8QuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_nvfp4_scale = gather_paged<std::uint8_t>(
                vs_physical, kNvfp4QuantGroups, geometry_, logical_capacity_, block_table_host_);
        } else {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint8_t>(k_scale_, k_scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_);
            cache.k_nvfp4       = gather_paged<std::uint8_t>(k_physical, kNvfp4CodeBytes, geometry_,
                                                             logical_capacity_, block_table_host_);
            cache.v_nvfp4       = gather_paged<std::uint8_t>(v_physical, kNvfp4CodeBytes, geometry_,
                                                             logical_capacity_, block_table_host_);
            cache.k_nvfp4_scale = gather_paged<std::uint8_t>(
                ks_physical, kNvfp4QuantGroups, geometry_, logical_capacity_, block_table_host_);
            cache.v_nvfp4_scale = gather_paged<std::uint8_t>(
                vs_physical, kNvfp4QuantGroups, geometry_, logical_capacity_, block_table_host_);
        }
        return cache;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (layout_.key.scale_extent != 0) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
        }
        if (layout_.value.scale_extent != 0) {
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        failures += block_table_.verify_guards((label + " block-table").c_str());
        failures +=
            verify_exact((label + " block-table unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_table_, block_table_host_.size()),
                         block_table_host_);
        return failures;
    }

private:
    Geometry geometry_;
    KvCacheStorage storage_;
    TestCacheLayout layout_;
    std::int32_t max_context_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::size_t k_code_elements_;
    std::size_t v_code_elements_;
    std::size_t k_scale_elements_;
    std::size_t v_scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
};

class BatchDeviceCache {
public:
    BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping)
        : geometry_(rows.front().geometry), storage_(rows.front().storage),
          layout_(test_cache_layout(storage_)), rows_(rows.size()),
          logical_capacity_(rows.front().logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(mapping == MappingPattern::Fragmented
                              ? 2 * static_cast<std::int32_t>(rows_) * logical_pages_ + 1
                              : static_cast<std::int32_t>(rows_) * logical_pages_),
          block_tables_host_(rows_ * static_cast<std::size_t>(logical_pages_)),
          k_code_elements_(
              physical_plane_elements(layout_.key.code_extent, geometry_, physical_pages_)),
          v_code_elements_(
              physical_plane_elements(layout_.value.code_extent, geometry_, physical_pages_)),
          k_scale_elements_(
              physical_plane_elements(layout_.key.scale_extent, geometry_, physical_pages_)),
          v_scale_elements_(
              physical_plane_elements(layout_.value.scale_extent, geometry_, physical_pages_)),
          k_(k_code_elements_ * dtype_size(layout_.key.code_dtype)),
          v_(v_code_elements_ * dtype_size(layout_.value.code_dtype)),
          k_scale_(layout_.key.scale_extent == 0
                       ? 1
                       : k_scale_elements_ * dtype_size(layout_.key.scale_dtype)),
          v_scale_(layout_.value.scale_extent == 0
                       ? 1
                       : v_scale_elements_ * dtype_size(layout_.value.scale_dtype)),
          block_tables_(block_tables_host_.size() * sizeof(std::int32_t)) {
        for (std::size_t row = 0; row < rows_; ++row) {
            const HostCache& cache = rows[row];
            if (cache.geometry.q_heads != geometry_.q_heads ||
                cache.geometry.kv_heads != geometry_.kv_heads || cache.storage != storage_ ||
                cache.logical_capacity != logical_capacity_) {
                throw std::invalid_argument("batch cache rows must share one physical geometry");
            }
            for (std::int32_t logical = 0; logical < logical_pages_; ++logical) {
                const std::int32_t linear =
                    static_cast<std::int32_t>(row) * logical_pages_ + logical;
                block_tables_host_[row * static_cast<std::size_t>(logical_pages_) + logical] =
                    mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
            }
        }
        block_tables_.copy_from_host(block_tables_host_.data(),
                                     block_tables_host_.size() * sizeof(std::int32_t));
        upload_rows(rows);
    }

    PagedKVBatchLayerView view() {
        PagedKVBatchLayerView result;
        result.k_pages = Tensor(
            k_.data(), layout_.key.code_dtype,
            {layout_.key.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages = Tensor(
            v_.data(), layout_.value.code_dtype,
            {layout_.value.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_tables = Tensor(block_tables_.data(), DType::I32,
                                     {logical_pages_, static_cast<std::int32_t>(rows_)});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.storage      = storage_;
        if (layout_.key.scale_extent != 0) {
            result.k_scale_pages = Tensor(
                k_scale_.data(), layout_.key.scale_dtype,
                {layout_.key.scale_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        }
        if (layout_.value.scale_extent != 0) {
            result.v_scale_pages = Tensor(v_scale_.data(), layout_.value.scale_dtype,
                                          {layout_.value.scale_extent, kPagedKVPageSize,
                                           geometry_.kv_heads, physical_pages_});
        }
        return result;
    }

    using Bytes = std::array<std::vector<std::uint8_t>, 4>;

    Bytes snapshot_bytes() const {
        return {copy_from_guarded<std::uint8_t>(k_, k_.bytes()),
                copy_from_guarded<std::uint8_t>(v_, v_.bytes()),
                copy_from_guarded<std::uint8_t>(k_scale_, k_scale_.bytes()),
                copy_from_guarded<std::uint8_t>(v_scale_, v_scale_.bytes())};
    }

    void copy_from(const BatchDeviceCache& source) {
        const std::array<const GuardedDeviceBuffer*, 4> from{&source.k_, &source.v_,
                                                             &source.k_scale_, &source.v_scale_};
        const std::array<GuardedDeviceBuffer*, 4> to{&k_, &v_, &k_scale_, &v_scale_};
        for (int i = 0; i < 4; ++i)
            CUDA_CHECK(cudaMemcpy(to[i]->data(), from[i]->data(), to[i]->bytes(),
                                  cudaMemcpyDeviceToDevice));
    }

    PagedKVLayerView single_view(int row) {
        auto batch = view();
        return {.k_pages       = batch.k_pages,
                .v_pages       = batch.v_pages,
                .k_scale_pages = batch.k_scale_pages,
                .v_scale_pages = batch.v_scale_pages,
                .block_table =
                    Tensor(static_cast<std::int32_t*>(block_tables_.data()) + row * logical_pages_,
                           DType::I32, {logical_pages_}),
                .head_dim     = kHeadDim,
                .num_kv_heads = geometry_.kv_heads,
                .storage      = storage_};
    }

    int verify_untouched(const std::string& label, const Bytes& before,
                         std::span<const int> positions, std::span<const int> lanes,
                         std::span<const int> valid, int width) const {
        const auto after        = snapshot_bytes();
        const int physical_rows = physical_pages_ * geometry_.kv_heads * kPagedKVPageSize;
        std::vector<bool> writable(physical_rows, false);
        for (std::size_t b = 0; b < valid.size(); ++b)
            for (int j = 0; j < valid[b]; ++j) {
                const int pos = positions[b * width + j];
                const int page =
                    block_tables_host_[lanes[b] * logical_pages_ + pos / kPagedKVPageSize];
                for (int h = 0; h < geometry_.kv_heads; ++h)
                    writable[paged_index(1, geometry_, page, h, pos, 0)] = true;
            }
        const std::array<std::size_t, 4> row_bytes{
            layout_.key.code_extent * dtype_size(layout_.key.code_dtype),
            layout_.value.code_extent * dtype_size(layout_.value.code_dtype),
            layout_.key.scale_extent * dtype_size(layout_.key.scale_dtype),
            layout_.value.scale_extent * dtype_size(layout_.value.scale_dtype)};
        int failures = 0;
        for (int plane = 0; plane < 4; ++plane) {
            if (row_bytes[plane] == 0) {
                failures +=
                    verify_exact((label + " unused scale").c_str(), after[plane], before[plane]);
                continue;
            }
            for (int row = 0; row < physical_rows; ++row)
                if (!writable[row]) {
                    const auto first = std::size_t(row) * row_bytes[plane];
                    if (!std::equal(before[plane].begin() + first,
                                    before[plane].begin() + first + row_bytes[plane],
                                    after[plane].begin() + first)) {
                        std::cerr << label << ": unrelated cache bytes changed in plane " << plane
                                  << " row " << row << '\n';
                        ++failures;
                        break;
                    }
                }
        }
        return failures;
    }

    int verify(const std::string& label, std::span<const HostCache> expected) const {
        if (expected.size() != rows_) {
            std::cerr << label << ": expected cache row count mismatch\n";
            return 1;
        }
        int failures = 0;
        if (storage_ == KvCacheStorage::BFloat16) {
            std::vector<std::uint16_t> expected_k(k_code_elements_, 0);
            std::vector<std::uint16_t> expected_v(v_code_elements_, 0);
            scatter_bf16_rows(expected, expected_k, expected_v);
            failures +=
                verify_exact((label + " cache-k").c_str(),
                             copy_from_guarded<std::uint16_t>(k_, k_code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v").c_str(),
                             copy_from_guarded<std::uint16_t>(v_, v_code_elements_), expected_v);
        } else if (storage_ == KvCacheStorage::Int8Group64) {
            std::vector<std::int8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint16_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].v_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::int8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
            std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
            std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(k_scale_elements_, 0);
            std::vector<std::uint16_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].k_scale, kFp8QuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_scale, kFp8QuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
            std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(k_scale_elements_, 0);
            std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_nvfp4, kNvfp4CodeBytes, geometry_,
                                   logical_capacity_, table, expected_v);
                scatter_paged_into(expected[row].k_scale, kFp8QuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        } else {
            std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
            std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint8_t> expected_ks(k_scale_elements_, 0);
            std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_nvfp4, kNvfp4CodeBytes, geometry_,
                                   logical_capacity_, table, expected_k);
                scatter_paged_into(expected[row].v_nvfp4, kNvfp4CodeBytes, geometry_,
                                   logical_capacity_, table, expected_v);
                scatter_paged_into(expected[row].k_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint8_t>(k_scale_, k_scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        }
        failures +=
            verify_exact((label + " block tables unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_tables_, block_tables_host_.size()),
                         block_tables_host_);
        failures += k_.verify_guards((label + " cache-k guard").c_str());
        failures += v_.verify_guards((label + " cache-v guard").c_str());
        if (layout_.key.scale_extent != 0) {
            failures += k_scale_.verify_guards((label + " cache-k-scale guard").c_str());
        }
        if (layout_.value.scale_extent != 0) {
            failures += v_scale_.verify_guards((label + " cache-v-scale guard").c_str());
        }
        failures += block_tables_.verify_guards((label + " block tables guard").c_str());
        return failures;
    }

private:
    [[nodiscard]] std::span<const std::int32_t> row_table(std::size_t row) const {
        return std::span<const std::int32_t>(block_tables_host_.data() +
                                                 row * static_cast<std::size_t>(logical_pages_),
                                             static_cast<std::size_t>(logical_pages_));
    }

    void scatter_bf16_rows(std::span<const HostCache> rows, std::vector<std::uint16_t>& k,
                           std::vector<std::uint16_t>& v) const {
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_bf16, kHeadDim, geometry_, logical_capacity_, table, k);
            scatter_paged_into(rows[row].v_fp16, kHeadDim, geometry_, logical_capacity_, table, v);
        }
    }

    void upload_rows(std::span<const HostCache> rows) {
        if (storage_ == KvCacheStorage::BFloat16) {
            std::vector<std::uint16_t> physical_k(k_code_elements_, 0);
            std::vector<std::uint16_t> physical_v(v_code_elements_, 0);
            scatter_bf16_rows(rows, physical_k, physical_v);
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::Int8Group64) {
            std::vector<std::int8_t> physical_k(k_code_elements_, 0);
            std::vector<std::int8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
            std::vector<std::uint16_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_i8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_v);
                scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_,
                                   table, physical_ks);
                scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_,
                                   table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
            std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
            std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
            std::vector<std::uint16_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_v);
                scatter_paged_into(rows[row].k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                                   table, physical_ks);
                scatter_paged_into(rows[row].v_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                                   table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size());
            v_.copy_from_host(physical_v.data(), physical_v.size());
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
            std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
            std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                                   table, physical_v);
                scatter_paged_into(rows[row].k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                                   table, physical_ks);
                scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                                   logical_capacity_, table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size());
            v_.copy_from_host(physical_v.data(), physical_v.size());
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size());
            return;
        }
        std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
        std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
        std::vector<std::uint8_t> physical_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, physical_k);
            scatter_paged_into(rows[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, physical_v);
            scatter_paged_into(rows[row].k_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, physical_ks);
            scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size());
        v_.copy_from_host(physical_v.data(), physical_v.size());
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size());
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size());
    }

    Geometry geometry_;
    KvCacheStorage storage_;
    TestCacheLayout layout_;
    std::size_t rows_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_tables_host_;
    std::size_t k_code_elements_;
    std::size_t v_code_elements_;
    std::size_t k_scale_elements_;
    std::size_t v_scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_tables_;
};

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected,
                 bool verify_private_key_representation) {
    int failures = 0;
    if (expected.storage == KvCacheStorage::BFloat16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
        failures += verify_exact((label + " cache-v").c_str(), got.v_fp16, expected.v_fp16);
    } else if (expected.storage == KvCacheStorage::Int8Group64) {
        if (verify_private_key_representation) {
            failures += verify_exact((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
            failures +=
                verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        }
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    } else if (expected.storage == KvCacheStorage::Fp8E4M3Row256) {
        if (verify_private_key_representation) {
            failures += verify_exact((label + " cache-k-code").c_str(), got.k_fp8, expected.k_fp8);
            failures +=
                verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        }
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_fp8, expected.v_fp8);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    } else if (expected.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        if (verify_private_key_representation) {
            failures += verify_exact((label + " cache-k-code").c_str(), got.k_fp8, expected.k_fp8);
            failures +=
                verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        }
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_nvfp4, expected.v_nvfp4);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_nvfp4_scale,
                                 expected.v_nvfp4_scale);
    } else {
        if (verify_private_key_representation) {
            failures +=
                verify_exact((label + " cache-k-code").c_str(), got.k_nvfp4, expected.k_nvfp4);
            failures += verify_exact((label + " cache-k-scale").c_str(), got.k_nvfp4_scale,
                                     expected.k_nvfp4_scale);
        }
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_nvfp4, expected.v_nvfp4);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_nvfp4_scale,
                                 expected.v_nvfp4_scale);
    }
    return failures;
}

int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected) {
    int failures = verify_exact(
        label.c_str(), copy_from_guarded<std::uint16_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected) {
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::int32_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

const char* cache_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-g64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3fn-row256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4-g16";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

ReductionCriterion attention_criterion(KvCacheStorage storage) {
    if (storage == KvCacheStorage::BFloat16) return kAttentionBf16Criterion;
    if (storage == KvCacheStorage::Int8Group64) return kAttentionInt8Criterion;
    if (storage == KvCacheStorage::Fp8E4M3Row256) return kAttentionFp8Criterion;
    if (storage == KvCacheStorage::Nvfp4Group16) return kAttentionNvfp4Criterion;
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) return kAttentionK8V4Criterion;
    throw std::logic_error("unregistered causal-attention test storage");
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

std::string case_label(const char* entry, const Geometry& geometry, KvCacheStorage storage,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(storage) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max) +
           (test_case.graph_replay ? " graph-replay" : "");
}

template <class Launch>
void launch_attention_case(Launch&& launch, bool graph_replay) {
    if (!graph_replay) {
        launch(nullptr);
        cuda_synchronize();
        return;
    }

    // Prime route-local CUDA function attributes before stream capture. The append route writes
    // the same represented values to the same positions, so this does not change its oracle.
    launch(nullptr);
    cuda_synchronize();

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create causal-attention graph stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin causal-attention capture");
    launch(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end causal-attention capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate causal-attention graph");
    for (int replay = 0; replay < 2; ++replay) {
        cuda_check(cudaGraphLaunch(executable, stream), "launch causal-attention graph");
    }
    cuda_check(cudaStreamSynchronize(stream), "synchronize causal-attention graph");
    cuda_check(cudaGraphExecDestroy(executable), "destroy causal-attention graph executable");
    cuda_check(cudaGraphDestroy(graph), "destroy causal-attention graph");
    cuda_check(cudaStreamDestroy(stream), "destroy causal-attention graph stream");
}

void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v) {
    if (tokens == 0) return;
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        k[kv_input_index(geometry, 0, d, 0)]               = 0.0f;
        v[kv_input_index(geometry, 0, kQuantGroup + d, 0)] = 0.0f;
    }
    k[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = -1.0f;
    v[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = 1.0f;
}

// Separate allocations make an accidentally unstaged read fail the independent
// attention oracle instead of aliasing the original contiguous backing.
class StagedCache {
public:
    explicit StagedCache(PagedKVBatchLayerView full) : original_(full), view_(full) {
        if ((full.storage != KvCacheStorage::Fp8KeyNvfp4Value &&
             full.storage != KvCacheStorage::Fp8E4M3Row256) ||
            full.k_pages.ne[3] < 2) {
            throw std::invalid_argument("staged test requires multiple FP8 or K8V4 pages");
        }
        full_            = {full.k_pages, full.v_pages, full.k_scale_pages, full.v_scale_pages};
        const auto split = full.k_pages.ne[3] / 2;
        for (std::size_t p = 0; p < 4; ++p) {
            resident_[p]       = full_[p].slice(3, 0, split);
            tail_[p]           = full_[p].slice(3, split, full.k_pages.ne[3] - split);
            local_storage_[p]  = std::make_unique<GuardedDeviceBuffer>(resident_[p].bytes());
            staged_storage_[p] = std::make_unique<GuardedDeviceBuffer>(tail_[p].bytes());
            CUDA_CHECK(cudaMemcpy(local_storage_[p]->data(), resident_[p].data,
                                  resident_[p].bytes(), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(staged_storage_[p]->data(), tail_[p].data, tail_[p].bytes(),
                                  cudaMemcpyDeviceToDevice));
            resident_[p].data = local_storage_[p]->data();
            tail_[p].data     = staged_storage_[p]->data();
        }
        view_.k_pages       = resident_[0];
        view_.v_pages       = resident_[1];
        view_.k_scale_pages = resident_[2];
        view_.v_scale_pages = resident_[3];
        view_.staged        = {tail_[0], tail_[1], tail_[2], tail_[3]};
    }

    PagedKVBatchLayerView batch_view() const { return view_; }

    PagedKVLayerView view() const {
        return {.k_pages       = view_.k_pages,
                .v_pages       = view_.v_pages,
                .k_scale_pages = view_.k_scale_pages,
                .v_scale_pages = view_.v_scale_pages,
                .block_table   = view_.block_tables.view({view_.block_tables.ne[0]}),
                .head_dim      = view_.head_dim,
                .num_kv_heads  = view_.num_kv_heads,
                .storage       = view_.storage,
                .staged        = view_.staged};
    }

    void merge() const {
        cuda_synchronize();
        for (std::size_t p = 0; p < 4; ++p) {
            CUDA_CHECK(cudaMemcpy(full_[p].data, resident_[p].data, resident_[p].bytes(),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(full_[p].data) + resident_[p].bytes(),
                                  tail_[p].data, tail_[p].bytes(), cudaMemcpyDeviceToDevice));
        }
        cuda_synchronize();
    }

    int guards(const std::string& label) const {
        int failures = 0;
        for (std::size_t p = 0; p < 4; ++p) {
            failures += local_storage_[p]->verify_guards((label + " resident").c_str());
            failures += staged_storage_[p]->verify_guards((label + " staged").c_str());
        }
        return failures;
    }
private:
    PagedKVBatchLayerView original_, view_;
    std::array<Tensor, 4> full_, resident_, tail_;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 4> local_storage_, staged_storage_;
};

// Chooses the primary/secondary device pair once per process. The secondary
// MUST be the compute-capable device the distributed route is specified for:
// its secondary kernel is compiled for SM86 and for the primary architecture,
// and a device without a matching image fails at first launch rather than here.
// The selected pair's capabilities are reported so the caller can require the
// Ampere pairing instead of silently running an unqualified one, and the
// current device is restored on every path.
struct DistributedDevices {
    int primary         = 0;
    int secondary       = 1;
    bool selectable     = false;
    int secondary_major = 0;
    int secondary_minor = 0;
    std::string reason;

    [[nodiscard]] bool secondary_is_sm86() const {
        return secondary_major == 8 && secondary_minor == 6;
    }
};

DistributedDevices select_distributed_devices() {
    DistributedDevices selection;
    // A host with no usable CUDA device has nothing to choose from. That
    // condition is reported as a skip reason by the caller, so it must be
    // established before the count query rather than aborting inside it.
    if (cuda_unavailable()) {
        selection.reason = "requires a usable CUDA device";
        return selection;
    }
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        selection.reason = "requires two CUDA devices";
        return selection;
    }
    int original = 0;
    CUDA_CHECK(cudaGetDevice(&original));
    std::vector<int> usable;
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
            cudaGetLastError(); // Clear the sticky invalid-device error and keep
                                // scanning.
            continue;
        }
        if (properties.major <= 0) continue;
        usable.push_back(device);
    }
    if (usable.size() >= 2) {
        // Prefer the highest compute capability as primary; the oldest usable
        // device becomes the secondary so the pairing is stable no matter which
        // device the caller had bound.
        std::sort(usable.begin(), usable.end(), [&](int left, int right) {
            cudaDeviceProp a{}, b{};
            CUDA_CHECK(cudaGetDeviceProperties(&a, left));
            CUDA_CHECK(cudaGetDeviceProperties(&b, right));
            if (a.major != b.major) return a.major > b.major;
            return a.minor > b.minor;
        });
        cudaDeviceProp secondary_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&secondary_properties, usable.back()));
        selection.primary         = usable.front();
        selection.secondary       = usable.back();
        selection.secondary_major = secondary_properties.major;
        selection.secondary_minor = secondary_properties.minor;
        selection.selectable      = true;
    } else {
        selection.reason = "requires two CUDA-capable devices";
    }
    CUDA_CHECK(cudaSetDevice(original));
    return selection;
}

// Selects a device for a teardown path. Teardown runs while a stack may already
// be unwinding, so it cannot abort the process or throw: a failure is reported
// in the same form the repository's other cleanup paths use and the remaining
// cleanup still runs. Buffer frees already take that stance, since
// ~DeviceBuffer releases through the logging, non-aborting free_device().
void restore_device_noexcept(int device) noexcept {
    const cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice(%d): %s: %s\n", device,
                     cudaGetErrorName(status), cudaGetErrorString(status));
    }
}

// How the physical page ID namespace is partitioned for one distributed case.
// The split is derived from the physical IDs the visible keys actually resolve
// to, so a case keeps its meaning when a mapping or fixture geometry changes,
// and a participation check then proves the resulting partition is what the
// case claims. Without that check a "remote" case could resolve every key
// through the primary plane and prove nothing about the second device, and an
// "empty contribution" case could exercise a full shard on both devices.
enum class RemoteCase { EvenIds, AllLocalPlanes, AllRemoteIds, PrimaryLowestId, ScaleBoundary };

const char* remote_case_name(RemoteCase remote_case) {
    switch (remote_case) {
    case RemoteCase::EvenIds:
        return "even-ids";
    case RemoteCase::AllLocalPlanes:
        return "all-local-planes";
    case RemoteCase::AllRemoteIds:
        return "all-remote-ids";
    case RemoteCase::PrimaryLowestId:
        return "primary-lowest-id";
    case RemoteCase::ScaleBoundary:
        return "scale-boundary";
    }
    return "unknown";
}

// The physical ID partition implied by a case. Primary IDs are [0,
// primary_pages); secondary IDs are [primary_pages, physical_pages). Both
// planes are always real allocations: an empty contribution is a plane whose
// IDs the block table never references, not a plane that is absent. That keeps
// every case meaningful without depending on an absent plane being accepted.
struct RemotePartition {
    std::int32_t primary_pages = 1;
};

// `referenced` holds every physical page ID the visible key set resolves to.
// The split always leaves at least one ID on each side, so both devices run a
// real shard; an all-local or all-remote visible set is produced by the block
// table pointing away from one of them. A key set that stays inside a single
// page cannot be split at all, which verify_remote_partition reports rather
// than silently running as an all-local case.
//
// A scale-boundary case declares its split as the physical page where its
// scaled region begins, so `boundary_page` overrides the derived split and
// keeps the partition and the declared magnitude describing the same boundary
// instead of two independently drifting ones.
RemotePartition remote_partition(std::span<const std::int32_t> referenced,
                                 std::int32_t physical_pages, RemoteCase remote_case,
                                 std::int32_t boundary_page = 0) {
    if (referenced.empty()) throw std::invalid_argument("a partition case needs visible keys");
    if (physical_pages < 2) {
        throw std::invalid_argument("a distributed case needs at least two physical pages");
    }
    std::int32_t lowest = physical_pages, highest = -1;
    for (const std::int32_t physical : referenced) {
        if (physical < 0 || physical >= physical_pages) {
            throw std::invalid_argument("referenced physical page is outside the pool");
        }
        lowest  = std::min(lowest, physical);
        highest = std::max(highest, physical);
    }
    RemotePartition partition;
    switch (remote_case) {
    case RemoteCase::EvenIds:
        // Straddle the referenced range, so both planes serve part of the visible
        // set.
        partition.primary_pages = (lowest + highest + 1) / 2;
        break;
    case RemoteCase::AllLocalPlanes:
        // Every referenced ID stays primary; the last pool ID becomes secondary and
        // is referenced by nothing, so the remote contribution is empty. The
        // mapping must therefore leave that ID unreferenced, which the
        // participation check confirms.
        partition.primary_pages = physical_pages - 1;
        break;
    case RemoteCase::AllRemoteIds:
        // The primary plane keeps only the IDs below the lowest referenced one, and
        // the block table never reaches them, so the local contribution is empty.
        partition.primary_pages = lowest;
        break;
    case RemoteCase::PrimaryLowestId:
        // The primary keeps the lowest referenced ID and the secondary owns every
        // other referenced page, so both shard ranges are non-empty but unequal:
        // one page against the rest. Catches a shard bound or merge step that
        // assumes a balanced split, or that derives a range from the whole pool
        // instead of from the partition, even while a symmetric case still passes.
        partition.primary_pages = lowest + 1;
        break;
    case RemoteCase::ScaleBoundary: {
        // The declared scaled region begins at logical page `boundary_page`, and that same page
        // must be where the device split falls, so each shard's own rows really carry the magnitude
        // the case declares. The split is one past the highest physical ID that the pages below the
        // boundary resolve to: the mapping decides which IDs those are, so a split derived from the
        // page index alone would separate two referenced IDs only for an identity mapping.
        if (boundary_page <= 0 || boundary_page >= static_cast<std::int32_t>(referenced.size())) {
            throw std::invalid_argument(
                "a scale-boundary case needs the boundary inside its visible pages");
        }
        std::int32_t below = -1;
        for (std::int32_t page = 0; page < boundary_page; ++page) {
            below = std::max(below, referenced[static_cast<std::size_t>(page)]);
        }
        partition.primary_pages = below + 1;
        break;
    }
    }
    partition.primary_pages = std::clamp(partition.primary_pages, 1, physical_pages - 1);
    return partition;
}

// Physical page IDs a single-sequence visible key set of `visible_keys` keys
// resolves to.
std::vector<std::int32_t> referenced_pages(std::span<const std::int32_t> table,
                                           std::int32_t visible_keys) {
    if (visible_keys <= 0) return {};
    const auto pages =
        static_cast<std::size_t>((visible_keys + kPagedKVPageSize - 1) / kPagedKVPageSize);
    if (pages > table.size()) {
        throw std::invalid_argument("visible keys exceed the block table");
    }
    return std::vector<std::int32_t>(table.begin(),
                                     table.begin() + static_cast<std::ptrdiff_t>(pages));
}

// Block tables exactly as each fixture builds them, so the participation check
// reads the same addressing the Op does instead of a restated assumption.
std::vector<std::int32_t> batch_block_table(std::int32_t logical_pages, std::int32_t rows,
                                            MappingPattern mapping) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages) *
                                    static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t logical = 0; logical < logical_pages; ++logical) {
            const std::int32_t linear = row * logical_pages + logical;
            table[static_cast<std::size_t>(row) * static_cast<std::size_t>(logical_pages) +
                  static_cast<std::size_t>(logical)] =
                mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
        }
    }
    return table;
}

// Physical page IDs reached by the live rows of a batch, each through its own
// table row. A page reached by several rows is listed once.
std::vector<std::int32_t> batch_referenced_pages(std::span<const std::int32_t> tables,
                                                 std::int32_t logical_pages,
                                                 std::span<const int> lanes,
                                                 std::span<const int> visible_keys) {
    std::vector<std::int32_t> referenced;
    const std::int32_t table_rows =
        logical_pages > 0 ? static_cast<std::int32_t>(tables.size()) / logical_pages : 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (lanes[row] < 0 || lanes[row] >= table_rows) continue;
        const std::size_t offset =
            static_cast<std::size_t>(lanes[row]) * static_cast<std::size_t>(logical_pages);
        const auto table = tables.subspan(offset, static_cast<std::size_t>(logical_pages));
        for (const std::int32_t physical : referenced_pages(table, visible_keys[row])) {
            if (std::find(referenced.begin(), referenced.end(), physical) == referenced.end()) {
                referenced.push_back(physical);
            }
        }
    }
    return referenced;
}

// Checks that a case's visible keys really straddle the device split (or really
// do not) by reading the same block-table entries the Op reads. Failures are
// reported, not thrown, so a mis-declared case cannot silently degrade into a
// passing all-local test.
int verify_remote_partition(const std::string& label, std::span<const std::int32_t> referenced,
                            std::int32_t primary_pages, RemoteCase remote_case) {
    bool local = false, remote = false;
    for (const std::int32_t physical : referenced) {
        (physical < primary_pages ? local : remote) = true;
    }
    const bool wants_remote = remote_case != RemoteCase::AllLocalPlanes;
    const bool wants_local  = remote_case != RemoteCase::AllRemoteIds;
    if (wants_remote == remote && wants_local == local) return 0;
    std::cerr << label << ": case " << remote_case_name(remote_case)
              << " does not exercise its declared partition (local=" << (local ? "yes" : "no")
              << " remote=" << (remote ? "yes" : "no") << ")\n";
    return 1;
}

// Checks that the scale-extreme case really stores one tiny and one huge key
// region, and that the boundary it declares coincides with the device split.
// The stored FP16 key row scales are read back from the same host cache the
// Op's state was gathered from, so a case that intended one tiny and one huge
// scale cannot silently run with two ordinary ones.
//
// The required observed ratio is derived from the magnitude the case declares,
// so changing that declaration keeps this check meaningful instead of leaving a
// second, independently drifting constant behind.
int verify_scale_extremes(const std::string& label, const HostCache& cache,
                          const Geometry& geometry, std::span<const std::int32_t> block_table,
                          const KeyMagnitude& magnitude, std::int32_t visible_keys,
                          std::int32_t primary_pages) {
    const std::int32_t boundary = magnitude.boundary;
    if (boundary <= 0 || boundary % kPagedKVPageSize != 0) {
        std::cerr << label << ": scale boundary is not a page boundary\n";
        return 1;
    }
    // Both regions use the same all-ones prototype, so their FP8 row absmaxes track the declared
    // magnitude ratio directly. Requiring one quarter of it leaves room for codec rounding while
    // rejecting an ordinary, near-balanced scale pair.
    const float expected_ratio = magnitude.high / magnitude.low;
    // Physical IDs of the visible pages, read from the same block table the Op
    // resolves through.
    const auto visible_pages =
        static_cast<std::size_t>((visible_keys + kPagedKVPageSize - 1) / kPagedKVPageSize);
    if (visible_pages > block_table.size()) {
        std::cerr << label << ": scale case visible set exceeds its block table\n";
        return 1;
    }
    const std::int32_t boundary_page = boundary / kPagedKVPageSize;
    for (std::size_t page = 0; page < visible_pages; ++page) {
        const std::int32_t physical = block_table[page];
        const bool below            = static_cast<std::int32_t>(page) < boundary_page;
        if (below == (physical < primary_pages)) continue;
        std::cerr << label << ": scale boundary does not coincide with the device split (page "
                  << page << " physical=" << physical << " primary_pages=" << primary_pages
                  << ")\n";
        return 1;
    }
    float low_scale = 0.0f, high_scale = 0.0f;
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < visible_keys; ++position) {
            const std::size_t index =
                fp8_scale_index(geometry, cache.logical_capacity, head, position);
            float& side_scale = position < boundary ? low_scale : high_scale;
            side_scale        = std::max(side_scale, f16_bits_to_f32(cache.k_scale[index]));
        }
    }
    const float observed_ratio = low_scale > 0.0f ? high_scale / low_scale : 0.0f;
    if (low_scale > 0.0f && high_scale > 0.0f && observed_ratio >= expected_ratio / 4.0f &&
        observed_ratio <= expected_ratio * 4.0f) {
        return 0;
    }
    std::cerr << label << ": declared scale extremes are not present (low=" << low_scale
              << " high=" << high_scale << ")\n";
    return 1;
}

// Verifies that the scale fixture remains discriminating for every query head and causal token.
// Each side's represented FP8 keys must have one score, and the independent FP64 LSEs must
// leave the declared dominant shard at least 100 nats ahead. The all-ones query is exactly
// E4M3-safe; row-uniform scores prevent private query rounding from changing an argmax.
// Distinct zero/one value regions make a wrong shard blend observable.
int verify_scale_discriminator(const std::string& label, const std::vector<float>& q,
                               const HostCache& cache, const Geometry& geometry,
                               const KeyMagnitude& magnitude,
                               std::span<const std::int32_t> positions) {
    if (!magnitude.uniform_regions) return 0;
    if (magnitude.low == magnitude.high || magnitude.boundary <= 0) {
        std::cerr << label << ": scale discriminator has no declared contrast\n";
        return 1;
    }

    int failures                    = 0;
    const double expected_direction = magnitude.high > magnitude.low ? 1.0 : -1.0;
    for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
        const std::int32_t kv_head = q_head / geometry.query_group();
        double low_value_sum = 0.0, high_value_sum = 0.0;
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            low_value_sum += cache_value(cache, false, kv_head, 0, d);
            high_value_sum += cache_value(cache, false, kv_head, magnitude.boundary, d);
        }
        if (std::abs(high_value_sum - low_value_sum) / static_cast<double>(kHeadDim) < 0.5) {
            std::cerr << label << ": scale regions do not produce distinct shard values\n";
            ++failures;
            break;
        }

        for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
            std::array<double, kHeadDim> query{};
            bool exact_query = true;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const float value = q[q_index(geometry, q_head, d, token)];
                exact_query &= value == 1.0f;
                query[static_cast<std::size_t>(d)] = value;
            }
            if (!exact_query) {
                std::cerr << label << ": scale query is not the exact E4M3-safe all-ones row "
                          << "(head=" << q_head << " token=" << token << ")\n";
                ++failures;
                continue;
            }
            normalized_hadamard_d256(query);

            const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
            if (magnitude.boundary >= visible) {
                std::cerr << label << ": scale boundary is outside a causal query frontier\n";
                ++failures;
                continue;
            }
            double low_min = std::numeric_limits<double>::infinity(), low_max = -low_min;
            double high_min = std::numeric_limits<double>::infinity(), high_max = -high_min;
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += query[static_cast<std::size_t>(d)] *
                           cache_value(cache, true, kv_head, position, d);
                }
                const double score = dot * static_cast<double>(kAttentionScale);
                if (position < magnitude.boundary) {
                    low_min = std::min(low_min, score);
                    low_max = std::max(low_max, score);
                } else {
                    high_min = std::min(high_min, score);
                    high_max = std::max(high_max, score);
                }
            }
            const double low_lse = low_max + std::log(static_cast<double>(magnitude.boundary));
            const double high_lse =
                high_max + std::log(static_cast<double>(visible - magnitude.boundary));
            const double contrast = expected_direction * (high_lse - low_lse);
            const auto nonuniform = [](double minimum, double maximum) {
                return maximum - minimum > 1e-8 * std::max(1.0, std::abs(maximum));
            };
            if (nonuniform(low_min, low_max) || nonuniform(high_min, high_max) ||
                contrast < 100.0) {
                std::cerr << label << ": scale discriminator is not stable for head=" << q_head
                          << " token=" << token << " (LSE contrast=" << contrast << ")\n";
                ++failures;
            }
        }
    }
    return failures;
}

// Owns the secondary-resident planes for one distributed call and mirrors them
// back into the primary cache so the same independent oracle and the same exact
// cache-append parity checks apply to split and unsplit routes alike.
//
// The primary/secondary split is a partition of global physical page IDs, not a
// split of the visible key set: this fixture assigns whole physical pages, so
// the block table decides which device serves each visible key. Every visible
// key must still be contributed exactly once.
//
// The remote planes must live on the secondary device, so the fixture copies
// them there on construction, restores the caller's device, and copies them
// back on merge(). Both planes are real allocations; which visible keys they
// serve is decided by the block table alone.
class RemoteAttentionFixture {
public:
    RemoteAttentionFixture(PagedKVBatchLayerView full, DistributedDevices devices,
                           RemotePartition partition)
        : full_(full), local_(full), devices_(devices), primary_pages_(partition.primary_pages) {
        const std::int32_t total_pages  = full.k_pages.ne[3];
        const std::int32_t remote_pages = total_pages - primary_pages_;
        if (remote_pages <= 0) {
            throw std::invalid_argument("a distributed case needs a secondary page range");
        }
        CUDA_CHECK(cudaGetDevice(&entry_device_));
        const std::array<Tensor, 4> planes{full.k_pages, full.v_pages, full.k_scale_pages,
                                           full.v_scale_pages};
        CUDA_CHECK(cudaSetDevice(devices_.secondary));
        for (int p = 0; p < 4; ++p) {
            remote_[p] = planes[p].slice(3, primary_pages_, remote_pages);
            std::vector<std::byte> bytes(remote_[p].bytes());
            CUDA_CHECK(cudaMemcpy(static_cast<void*>(bytes.data()), remote_[p].data, bytes.size(),
                                  cudaMemcpyDeviceToHost));
            buffers_[p] = std::make_unique<DeviceBuffer>(bytes.size());
            buffers_[p]->copy_from_host(bytes.data(), bytes.size());
            remote_[p].data = buffers_[p]->p;
        }
        CUDA_CHECK(cudaSetDevice(entry_device_));
        local_.k_pages       = planes[0].slice(3, 0, primary_pages_);
        local_.v_pages       = planes[1].slice(3, 0, primary_pages_);
        local_.k_scale_pages = planes[2].slice(3, 0, primary_pages_);
        local_.v_scale_pages = planes[3].slice(3, 0, primary_pages_);
    }

    // Declared noexcept so a future teardown step that could throw is a compile
    // error instead of a terminate() while a stack unwinds.
    ~RemoteAttentionFixture() noexcept {
        restore_device_noexcept(devices_.secondary);
        for (auto& buffer : buffers_) buffer.reset();
        restore_device_noexcept(entry_device_);
    }

    [[nodiscard]] std::int32_t primary_pages() const { return primary_pages_; }

    [[nodiscard]] PagedKVBatchLayerView local() const { return local_; }

    [[nodiscard]] PagedKVStagedPlanes remote() const {
        return {remote_[0], remote_[1], remote_[2], remote_[3]};
    }

    void merge() {
        cuda_synchronize();
        const std::array<Tensor, 4> planes{full_.k_pages, full_.v_pages, full_.k_scale_pages,
                                           full_.v_scale_pages};
        for (int p = 0; p < 4; ++p) {
            std::vector<std::byte> bytes(remote_[p].bytes());
            CUDA_CHECK(cudaSetDevice(devices_.secondary));
            buffers_[p]->copy_to_host(bytes.data(), bytes.size());
            CUDA_CHECK(cudaSetDevice(entry_device_));
            auto destination = planes[p].slice(3, primary_pages_, remote_[p].ne[3]);
            CUDA_CHECK(
                cudaMemcpy(destination.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        }
        cuda_synchronize();
    }

private:
    PagedKVBatchLayerView full_, local_;
    DistributedDevices devices_;
    int entry_device_           = 0;
    std::int32_t primary_pages_ = 0;
    std::array<Tensor, 4> remote_;
    std::array<std::unique_ptr<DeviceBuffer>, 4> buffers_;
};

int run_a1_case(const Geometry& geometry, KvCacheStorage storage, const AttentionCase& test_case,
                MappingPattern mapping, bool use_staging = false,
                const RemoteCase* remote_case     = nullptr,
                const DistributedDevices* devices = nullptr,
                const KeyMagnitude* magnitude     = nullptr) {
    const bool distributed         = remote_case != nullptr;
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements  = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.q_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q          = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    if (magnitude != nullptr && magnitude->uniform_regions) std::fill(q.begin(), q.end(), 1.0f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    inject_codec_edges(geometry, test_case.tokens, k, v);
    // The scale discriminator uses identical logical keys for resident and appended rows.
    if (magnitude != nullptr) {
        for (std::int32_t token = 0; token < test_case.tokens; ++token) {
            const std::int32_t position = positions[static_cast<std::size_t>(token)];
            const float factor          = magnitude->at(position);
            if (factor == 1.0f && !magnitude->uniform_regions) continue;
            for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t index = kv_input_index(geometry, head, d, token);
                    if (magnitude->uniform_regions) {
                        k[index] = factor;
                        v[index] = position < magnitude->boundary ? 0.0f : 1.0f;
                    } else {
                        k[index] *= factor;
                    }
                }
            }
        }
    }
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         test_case.envelope_max};

    const HostCache initial =
        make_cache(geometry, storage, max_context, test_case.seed + 10u, magnitude);
    HostCache expected = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);
    const int scale_fixture_failures =
        magnitude == nullptr ? 0
                             : verify_scale_discriminator("scale-boundary fixture", q, expected,
                                                          geometry, *magnitude, positions);
    DeviceCache cache(initial, mapping);
    std::optional<StagedCache> staged;
    if (use_staging) { staged.emplace(cache.batch_view()); }

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});
    std::optional<RemoteAttentionFixture> remote;
    std::optional<ops::DistributedAttention> distributed_attention;
    std::vector<std::int32_t> distributed_referenced;
    if (distributed) {
        const std::vector<std::int32_t> table =
            make_block_table(cache.batch_view().block_tables.ne[0], mapping);
        distributed_referenced = referenced_pages(table, total);
        // A scale case declares its split as the page where its scaled region
        // begins, so the partition is derived from that same boundary rather than
        // restated separately.
        const std::int32_t boundary_page =
            magnitude == nullptr ? 0 : magnitude->boundary / kPagedKVPageSize;
        remote.emplace(cache.batch_view(), *devices,
                       remote_partition(distributed_referenced, cache.batch_view().k_pages.ne[3],
                                        *remote_case, boundary_page));
        distributed_attention.emplace(devices->primary, devices->secondary, op_geometry(geometry),
                                      test_case.tokens, 1, cache.batch_view().block_tables.ne[1],
                                      cache.batch_view().block_tables.ne[0]);
    }

    launch_attention_case(
        [&](cudaStream_t stream) {
            if (distributed_attention) {
                distributed_attention->execute(tq, tk, tv, tp, Tensor{}, ttable_row,
                                               kAttentionScale, remote->local(), remote->remote(),
                                               envelope, tout, stream);
                return;
            }
            ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, ttable_row,
                                          op_geometry(geometry), kAttentionScale,
                                          staged ? staged->batch_view() : cache.batch_view(),
                                          envelope, workspace, tout, stream);
        },
        test_case.graph_replay);
    if (staged) { staged->merge(); }
    if (remote) remote->merge();

    const std::string label =
        case_label("causal_softmax_attention", geometry, storage, test_case, mapping) +
        (remote ? " remote=" + std::to_string(remote->primary_pages()) + "/" +
                      std::to_string(cache.batch_view().k_pages.ne[3])
                : std::string());
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures =
        scale_fixture_failures + verify_attention(label, bf16_bits_to_double(output_bits),
                                                  reference, attention_criterion(storage));
    if (remote_case != nullptr) {
        failures += verify_remote_partition(label, distributed_referenced, remote->primary_pages(),
                                            *remote_case);
    }
    if (magnitude != nullptr && remote_case != nullptr) {
        // The readback runs after merge(), so the primary cache holds both shards'
        // rows again and the declared scales are checked on the exact state the Op
        // read.
        const std::vector<std::int32_t> table =
            make_block_table(cache.batch_view().block_tables.ne[0], mapping);
        failures += verify_scale_extremes(label, cache.snapshot(), geometry, table, *magnitude,
                                          total, remote->primary_pages());
    }
    failures += verify_cache(label, cache.snapshot(), expected,
                             storage == KvCacheStorage::BFloat16 ||
                                 storage == KvCacheStorage::Nvfp4Group16 ||
                                 storage == KvCacheStorage::Fp8KeyNvfp4Value);
    if (storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value ||
        storage == KvCacheStorage::Fp8E4M3Row256) {
        DeviceCache standalone(initial, mapping);
        ops::kv_cache_append(tk, tv, tp, standalone.view(), nullptr);
        cuda_synchronize();
        failures += verify_cache(label + " fused/standalone byte parity", cache.snapshot(),
                                 standalone.snapshot(), true);
        failures += standalone.verify_guards(label + " standalone append");
    }
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    if (staged) { failures += staged->guards(label); }
    return failures;
}

int run_a3_case(const Geometry& geometry, KvCacheStorage storage, const AttentionCase& test_case,
                MappingPattern mapping, bool use_staging = false) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q         = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         test_case.envelope_max};

    const HostCache cache_host = make_cache(geometry, storage, max_context, test_case.seed + 10u);
    const std::vector<double> reference = ideal_attention(q, cache_host, positions);
    DeviceCache cache(cache_host, mapping);
    std::optional<StagedCache> staged;
    if (use_staging) { staged.emplace(cache.batch_view()); }

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    launch_attention_case(
        [&](cudaStream_t stream) {
            ops::causal_softmax_attention_cached(tq, tp, op_geometry(geometry), kAttentionScale,
                                                 staged ? staged->view() : cache.view(), envelope,
                                                 workspace, tout, stream);
        },
        test_case.graph_replay);
    if (staged) { staged->merge(); }

    const std::string label =
        case_label("causal_softmax_attention_cached", geometry, storage, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(storage));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host, true);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    if (staged) { failures += staged->guards(label); }
    return failures;
}

struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
    bool graph_replay = false;
};

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_batch_case(const Geometry& geometry, KvCacheStorage storage,
                   const BatchAttentionCase& test_case, bool use_staging = false,
                   const RemoteCase* remote_case     = nullptr,
                   const DistributedDevices* devices = nullptr) {
    const bool distributed = remote_case != nullptr;
    if (distributed && test_case.graph_replay) {
        throw std::invalid_argument("distributed batch cases use eager execution");
    }
    const int batch = test_case.contexts.size(), width = test_case.width;
    const int pool_rows = std::max(
        batch, *std::max_element(test_case.table_rows.begin(), test_case.table_rows.end()) + 1);
    int maximum_visible = 1;
    for (int b = 0; b < batch; ++b)
        maximum_visible = std::max(
            maximum_visible,
            test_case.contexts[b] + (test_case.graph_replay ? width : test_case.valid_columns[b]));
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<unsigned>(maximum_visible),
                                                         static_cast<unsigned>(maximum_visible)};
    const std::size_t q_column_elements  = std::size_t(kHeadDim) * geometry.q_heads,
                      kv_column_elements = std::size_t(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = std::size_t(width) * batch;
    auto q = make_bf16_values(q_column_elements * columns, test_case.seed, -.25f, .25f);
    auto k = make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -.25f, .25f);
    auto v = make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.f, 1.f);
    inject_codec_edges(geometry, columns, k, v);
    std::vector<HostCache> initial;
    for (int row = 0; row < pool_rows; ++row)
        initial.push_back(
            make_cache(geometry, storage, maximum_visible + 3, test_case.seed + 20u + 3u * row));
    auto expected = initial;
    BatchDeviceCache cache(initial, test_case.mapping);
    std::optional<StagedCache> staged;
    if (use_staging) {
        if (test_case.graph_replay) {
            throw std::invalid_argument("active staging test uses eager execution");
        }
        staged.emplace(cache.view());
    }
    std::optional<BatchDeviceCache> control;
    if (test_case.graph_replay) control.emplace(initial, test_case.mapping);
    GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2), dp(columns * 4),
        dvalid(batch * 4), dlanes(batch * 4), dout(q.size() * 2);
    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, width, batch}),
        tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, width, batch});
    Tensor tp(dp.data(), DType::I32, {width, batch}), tvalid(dvalid.data(), DType::I32, {batch}),
        tlanes(dlanes.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, width, batch});
    const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, batch, width, width);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    int current_device = 0;
    CUDA_CHECK(cudaGetDevice(&current_device));
    DeviceContext device(current_device);
    const bool masked = test_case.graph_replay ||
                        std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](int count) { return count != width; });
    std::optional<RemoteAttentionFixture> remote;
    std::optional<ops::DistributedAttention> distributed_attention;
    std::vector<std::int32_t> distributed_referenced;
    if (distributed) {
        const std::vector<std::int32_t> table = batch_block_table(
            cache.view().block_tables.ne[0], cache.view().block_tables.ne[1], test_case.mapping);
        // Only live columns address cache, so the partition is derived from each row's live
        // frontier. An empty row addresses nothing and must not widen the partition.
        std::vector<int> row_visible(static_cast<std::size_t>(batch));
        for (int b = 0; b < batch; ++b) {
            const int valid = test_case.valid_columns[static_cast<std::size_t>(b)];
            row_visible[static_cast<std::size_t>(b)] =
                valid > 0 ? test_case.contexts[static_cast<std::size_t>(b)] + valid : 0;
        }
        distributed_referenced = batch_referenced_pages(table, cache.view().block_tables.ne[0],
                                                        test_case.table_rows, row_visible);
        remote.emplace(
            cache.view(), *devices,
            remote_partition(distributed_referenced, cache.view().k_pages.ne[3], *remote_case));
        distributed_attention.emplace(devices->primary, devices->secondary, op_geometry(geometry),
                                      width, batch, cache.view().block_tables.ne[1],
                                      cache.view().block_tables.ne[0]);
    }
    const auto launch = [&] {
        if (distributed_attention) {
            distributed_attention->execute(tq, tk, tv, tp, masked ? tvalid : Tensor{}, tlanes,
                                           kAttentionScale, remote->local(), remote->remote(),
                                           envelope, tout, device.stream);
            return;
        }
        ops::causal_softmax_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, tlanes,
                                      op_geometry(geometry), kAttentionScale,
                                      staged ? staged->batch_view() : cache.view(), envelope,
                                      workspace, tout, device.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    auto valid = test_case.valid_columns, lanes = test_case.table_rows;
    std::vector<int> positions(columns);
    int failures = 0;
    for (int phase = 0; phase < (test_case.graph_replay ? 3 : 1); ++phase) {
        if (phase) {
            for (auto& x : q) x *= -0.5f;
            for (auto& x : k) x *= -0.5f;
            for (auto& x : v) x *= -0.5f;
            round_to_bf16(q);
            round_to_bf16(k);
            round_to_bf16(v);
            std::rotate(lanes.begin(), lanes.begin() + 1, lanes.end());
            for (auto& count : valid) count = (count + 1) % (width + 1);
        }
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < width; ++j)
                positions[b * width + j] =
                    valid[b] == 0 ? 0 : test_case.contexts[b] + std::min(j, valid[b] - 1);
        const auto q_bits = to_bf16_bits(q), k_bits = to_bf16_bits(k), v_bits = to_bf16_bits(v);
        dq.copy_from_host(q_bits.data(), q_bits.size() * 2);
        dk.copy_from_host(k_bits.data(), k_bits.size() * 2);
        dv.copy_from_host(v_bits.data(), v_bits.size() * 2);
        dp.copy_from_host(positions.data(), positions.size() * 4);
        dvalid.copy_from_host(valid.data(), batch * 4);
        dlanes.copy_from_host(lanes.data(), batch * 4);
        dout.fill(0xff);
        scratch.fill(phase ? 0xa5 : 0x5a);
        cuda_synchronize();
        const auto before = cache.snapshot_bytes();
        if (control) control->copy_from(cache);
        std::vector<double> reference(q.size(), 0.0);
        for (int b = 0; b < batch; ++b)
            if (valid[b]) {
                const std::vector<int> row_positions(positions.begin() + b * width,
                                                     positions.begin() + b * width + valid[b]);
                auto row_q = extract_request_columns(q, q_column_elements, width, b, valid[b]);
                auto row_k = extract_request_columns(k, kv_column_elements, width, b, valid[b]);
                auto row_v = extract_request_columns(v, kv_column_elements, width, b, valid[b]);
                append_cache(expected[lanes[b]], row_k, row_v, row_positions);
                insert_request_columns(ideal_attention(row_q, expected[lanes[b]], row_positions),
                                       q_column_elements, width, b, reference);
            }
        if (test_case.graph_replay && phase == 0) {
            launch();
            cuda_synchronize(
                device.stream); // Warm lazy CUDA attributes; repeated append is idempotent.
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
            dout.fill(0xff);
            scratch.fill(0x5a);
            cuda_synchronize();
        }
        if (test_case.graph_replay)
            graph.launch(device.stream);
        else
            launch();
        cuda_synchronize(device.stream);
        if (staged) { staged->merge(); }
        if (remote) remote->merge();
        const std::string label = std::string("causal batch ") + geometry.name + " " +
                                  cache_name(storage) + " W=" + std::to_string(width) +
                                  " B=" + std::to_string(batch) +
                                  " phase=" + std::to_string(phase) +
                                  (remote ? " remote=" + std::to_string(remote->primary_pages()) +
                                                "/" + std::to_string(cache.view().k_pages.ne[3])
                                          : std::string());
        const auto output       = copy_from_guarded<std::uint16_t>(dout, q.size());
        failures += verify_attention(label, bf16_bits_to_double(output), reference,
                                     attention_criterion(storage));
        failures += verify_invalid_columns_zero(label, output, geometry, width, valid);
        if (remote_case != nullptr) {
            failures += verify_remote_partition(label, distributed_referenced,
                                                remote->primary_pages(), *remote_case);
        }
        failures += cache.verify(label, expected);
        failures += cache.verify_untouched(label, before, positions, lanes, valid, width);
        if (control) {
            for (int b = 0; b < batch; ++b)
                if (valid[b]) {
                    Tensor ck(static_cast<std::uint16_t*>(dk.data()) +
                                  b * width * kv_column_elements,
                              DType::BF16, {kHeadDim, geometry.kv_heads, valid[b]});
                    Tensor cv(static_cast<std::uint16_t*>(dv.data()) +
                                  b * width * kv_column_elements,
                              DType::BF16, {kHeadDim, geometry.kv_heads, valid[b]});
                    Tensor cp(static_cast<int*>(dp.data()) + b * width, DType::I32, {valid[b]});
                    ops::kv_cache_append(ck, cv, cp, control->single_view(lanes[b]), device.stream);
                }
            cuda_synchronize(device.stream);
            const auto actual = cache.snapshot_bytes(), standalone = control->snapshot_bytes();
            for (int plane = 0; plane < 4; ++plane)
                failures += verify_exact(
                    (label + " standalone parity plane=" + std::to_string(plane)).c_str(),
                    actual[plane], standalone[plane]);
        }
        failures += verify_input(label + " q unchanged", dq, q_bits) +
                    verify_input(label + " k unchanged", dk, k_bits) +
                    verify_input(label + " v unchanged", dv, v_bits);
        failures += verify_positions(label + " positions unchanged", dp, positions) +
                    verify_positions(label + " counts unchanged", dvalid, valid) +
                    verify_positions(label + " lanes unchanged", dlanes, lanes);
        failures += dout.verify_guards(label) + scratch.verify_guards(label);
        if (staged) { failures += staged->guards(label); }
        if (workspace.used() != 0 || workspace.peak_used() > capacity) {
            std::cerr << label << ": workspace mismatch\n";
            ++failures;
        }
    }
    return failures;
}

int report_quantization_quality(KvCacheStorage storage, std::uint32_t seed) {
    const Geometry& geometry       = kGeometries[0];
    constexpr std::int32_t tokens  = 6;
    constexpr std::int32_t base    = 61;
    constexpr std::int32_t context = 128;
    const std::size_t q_elements   = static_cast<std::size_t>(kHeadDim) * geometry.q_heads * tokens;
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens;
    std::vector<float> q          = make_bf16_values(q_elements, seed, -0.25f, 0.25f);
    std::vector<float> k          = make_bf16_values(kv_elements, seed + 1u, -0.25f, 0.25f);
    std::vector<float> v          = make_bf16_values(kv_elements, seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    std::vector<std::int32_t> positions(tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    // Both fixtures start from the same public BF16 logical cache. The first oracle applies the
    // target codec exactly; the second evaluates the corresponding unquantized BF16/FP64
    // attention formula. This evidence is deliberately independent of production output.
    HostCache represented = make_cache(geometry, storage, context, seed + 10u);
    HostCache unquantized = make_cache(geometry, KvCacheStorage::BFloat16, context, seed + 10u);
    append_cache(represented, k, v, positions);
    append_cache(unquantized, k, v, positions);
    const std::vector<double> quantized_output = ideal_attention(q, represented, positions);
    const std::vector<double> bf16_output      = ideal_attention(q, unquantized, positions);

    double sum_abs       = 0.0;
    double sum_sq        = 0.0;
    double reference_sq  = 0.0;
    double maximum_error = 0.0;
    for (std::size_t i = 0; i < quantized_output.size(); ++i) {
        const double error = quantized_output[i] - bf16_output[i];
        sum_abs += std::abs(error);
        sum_sq += error * error;
        reference_sq += bf16_output[i] * bf16_output[i];
        maximum_error = std::max(maximum_error, std::abs(error));
    }
    const double count         = static_cast<double>(quantized_output.size());
    const double mae           = sum_abs / count;
    const double rmse          = std::sqrt(sum_sq / count);
    const double reference_rms = std::sqrt(reference_sq / count);
    const double relative_rmse = reference_rms == 0.0 ? 0.0 : rmse / reference_rms;
    if (!std::isfinite(mae) || !std::isfinite(rmse) || !std::isfinite(relative_rmse) ||
        !std::isfinite(maximum_error)) {
        std::cerr << cache_name(storage)
                  << " quantization quality produced non-finite statistics\n";
        return 1;
    }
    std::cout << cache_name(storage) << " quantized-vs-bf16 oracle quality" << " mae=" << mae
              << " rmse=" << rmse << " rel_rmse=" << relative_rmse << " max_abs=" << maximum_error
              << '\n';
    return 0;
}

int run_quantized_batch_cases(KvCacheStorage storage, std::uint32_t seed) {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], storage,
                               {1, {63}, {1}, {0}, MappingPattern::Identity, seed});
    failures +=
        run_batch_case(kGeometries[1], storage,
                       {6, {61, 127}, {6, 3}, {1, 0}, MappingPattern::Fragmented, seed + 1u});
    failures += run_batch_case(
        kGeometries[0], storage,
        {6, {0, 63, 127, 2048}, {1, 6, 0, 3}, {2, 0, 3, 1}, MappingPattern::Fragmented, seed + 2u});
    failures += run_batch_case(kGeometries[1], storage,
                               {1,
                                {0, 1, 63, 64, 127, 128, 2048, 8191},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Fragmented,
                                seed + 3u});
    if (storage == KvCacheStorage::Nvfp4Group16) {
        failures += run_batch_case(kGeometries[0], storage,
                                   {12,
                                    {0, 17, 61, 127, 0, 128, 63, 31},
                                    {12, 11, 1, 0, 7, 3, 0, 12},
                                    {7, 0, 4, 2, 6, 1, 5, 3},
                                    MappingPattern::Fragmented,
                                    seed + 4u,
                                    true});
    }
    return failures;
}

int run_dflash2_cases() {
    constexpr int order[]{7, 0, 4, 2, 6, 1, 5, 3};
    int failures = 0;
    for (auto storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        const auto run = [&](int width, int batch, int base, bool graph) {
            BatchAttentionCase c{width,
                                 {},
                                 {},
                                 {},
                                 MappingPattern::Fragmented,
                                 static_cast<unsigned>(1700 + width + 31 * batch),
                                 graph};
            for (int b = 0; b < batch; ++b) {
                c.contexts.push_back(base + (b % 3 == 0 ? 0 : b % 3 == 1 ? 17 : 61));
                c.valid_columns.push_back(b % 4 == 0   ? width
                                          : b % 4 == 1 ? width - 1
                                          : b % 4 == 2 ? 1
                                                       : 0);
                c.table_rows.push_back(order[b]);
            }
            return run_batch_case(kGeometries[0], storage, c);
        };
        for (int width = 2; width <= 16; ++width)
            for (int batch : {1, 8}) failures += run(width, batch, 0, false);
        for (int width : {2, 8, 16})
            for (int batch = 2; batch <= 7; ++batch) failures += run(width, batch, 0, false);
        for (int width : {7, 8, 9, 16})
            for (int batch : {1, 8}) failures += run(width, batch, 127, true);
        failures += run(16, 8, 2048, true);
        for (int width : {8, 9, 16}) {
            failures +=
                run_a1_case(kGeometries[0], storage,
                            {width, 2048, static_cast<unsigned>(2048 + width), 1801u, false, true},
                            MappingPattern::Fragmented);
            failures +=
                run_a3_case(kGeometries[0], storage,
                            {width, 8192, static_cast<unsigned>(8192 + width), 1802u, false, true},
                            MappingPattern::Fragmented);
        }
    }
    return failures;
}

int run_batch_cases() {
    int failures = 0;
    for (auto storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        failures += run_batch_case(kGeometries[0], storage,
                                   {16, {0}, {0}, {0}, MappingPattern::Fragmented, 1501u});
        failures += run_batch_case(kGeometries[0], storage,
                                   {16, {0}, {1}, {0}, MappingPattern::Fragmented, 1502u});
    }
    failures += run_batch_case(kGeometries[0], KvCacheStorage::Int8Group64,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
    failures += run_batch_case(kGeometries[0], KvCacheStorage::BFloat16,
                               {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
    failures += run_batch_case(kGeometries[0], KvCacheStorage::BFloat16,
                               {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
    failures += run_batch_case(kGeometries[1], KvCacheStorage::Int8Group64,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                502u});
    failures +=
        run_batch_case(kGeometries[0], KvCacheStorage::Int8Group64,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    failures += run_batch_case(kGeometries[1], KvCacheStorage::BFloat16,
                               {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    failures +=
        run_batch_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 505u});
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    for (const KvCacheStorage storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64}) {
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_a1_case(geometry, storage, {6, 61, 67, 190u}, mapping);
            failures += run_a3_case(geometry, storage, {1, 128, 129, 191u}, mapping);
        }

        const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {66, 63, 129, 205u},
        };
        for (const AttentionCase& test_case : a1_cases) {
            failures += run_a1_case(geometry, storage, test_case, MappingPattern::Identity);
        }

        const AttentionCase a3_cases[] = {
            {1, 31, 32, 301u},
            {7, 17, 512, 302u},
            {17, 31, 48, 303u},
        };
        for (const AttentionCase& test_case : a3_cases) {
            failures += run_a3_case(geometry, storage, test_case, MappingPattern::Identity);
        }

        if (geometry.q_heads == 16) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures +=
                run_a1_case(geometry, storage, {7, 17, 513, 401u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, storage, {7, 17, 513, 402u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, storage, {16, 17, 1024, 403u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, storage, {16, 17, 1025, 404u}, MappingPattern::Identity);
        }
    }
    return failures;
}

int run_fp8_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, KvCacheStorage::Fp8E4M3Row256, {65, 63, 192, 601u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8E4M3Row256, {65, 63, 192, 602u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8E4M3Row256, {1, 128, 192, 603u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8E4M3Row256, {1, 128, 192, 604u},
                                MappingPattern::Offset);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8E4M3Row256, {6, 61, 192, 605u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8E4M3Row256, {8, 61, 192, 951u},
                                MappingPattern::Fragmented, true);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8E4M3Row256, {129, 63, 256, 952u},
                                MappingPattern::Fragmented, true);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8E4M3Row256, {65, 63, 192, 953u},
                                MappingPattern::Offset, true);
    }
    failures += run_a3_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256, {3, 63, 192, 606u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256, {1, 16384, 16385, 607u},
                            MappingPattern::Fragmented);
    failures += run_a1_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256, {1024, 0, 1088, 954u},
                            MappingPattern::Fragmented, true);
    failures += run_a3_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256, {1, 65535, 65536, 955u},
                            MappingPattern::Fragmented, true);
    failures +=
        run_batch_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256,
                       {8, {61, 4091}, {7, 5}, {1, 0}, MappingPattern::Fragmented, 956u}, true);
    failures +=
        run_batch_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256,
                       {8, {63, 129}, {8, 0}, {0, 1}, MappingPattern::Fragmented, 957u}, true);
    return failures;
}

int run_nvfp4_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 0, 1, 701u},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 64, 65, 702u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 32, 33, 707u, true},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 1, 2, 708u, true},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 7, 8, 709u, true},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 15, 16, 710u, true},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 31, 32, 711u, true},
                                MappingPattern::Identity);
        failures += run_a1_case(geometry, KvCacheStorage::Nvfp4Group16, {6, 61, 192, 703u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, KvCacheStorage::Nvfp4Group16, {64, 0, 128, 704u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {65, 63, 192, 705u},
                                MappingPattern::Offset);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 2048, 2049, 706u},
                                MappingPattern::Fragmented);
    }
    failures += run_a1_case(kGeometries[0], KvCacheStorage::Nvfp4Group16,
                            {1, 64, 65, 712u, false, true}, MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[0], KvCacheStorage::Nvfp4Group16,
                            {1, 64, 65, 713u, false, true}, MappingPattern::Fragmented);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {5, 17, 22, 714u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {7, 17, 24, 715u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {7, 511, 518, 716u},
                            MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 8191, 8192, 717u},
                            MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 16383, 16384, 718u},
                            MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 65535, 65536, 719u},
                            MappingPattern::Fragmented);
    return failures;
}

int run_k8v4_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 0, 1, 801u},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 64, 65, 802u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {6, 61, 192, 803u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {64, 0, 128, 804u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {65, 63, 192, 805u},
                                MappingPattern::Offset);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 2048, 2049, 806u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {8, 61, 192, 901u},
                                MappingPattern::Fragmented, true);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {129, 63, 256, 902u},
                                MappingPattern::Fragmented, true);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {65, 63, 192, 903u},
                                MappingPattern::Offset, true);
    }
    failures += run_a1_case(kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value, {1024, 0, 1088, 904u},
                            MappingPattern::Fragmented, true);
    failures += run_a3_case(kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 65535, 65536, 905u}, MappingPattern::Fragmented, true);
    failures +=
        run_batch_case(kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                       {8, {61, 4091}, {7, 5}, {1, 0}, MappingPattern::Fragmented, 906u}, true);
    failures +=
        run_batch_case(kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                       {8, {63, 129}, {8, 0}, {0, 1}, MappingPattern::Fragmented, 907u}, true);
    failures += run_a1_case(kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 64, 65, 807u, false, true}, MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 64, 65, 808u, false, true}, MappingPattern::Fragmented);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {5, 17, 22, 809u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {7, 17, 24, 810u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {7, 511, 518, 811u},
                            MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {1, 8191, 8192, 812u},
                            MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 16383, 16384, 813u}, MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 65535, 65536, 814u}, MappingPattern::Fragmented);
    return failures;
}

int verify_workspace_capacity_contract() {
    int failures = 0;
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        constexpr ops::CausalAttentionExecutionEnvelope envelope{1, 1025};
        constexpr ops::AttentionHeadGeometry geometry{kHeadDim, 16, 2};
        const std::size_t interval = ops::causal_softmax_attention_workspace_capacity_bytes(
            geometry, storage, envelope, 1, 1, 17);
        std::size_t witness = 0;
        for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
            witness = std::max(witness, ops::causal_softmax_attention_workspace_capacity_bytes(
                                            geometry, storage, envelope, 1, tokens, tokens));
        }
        if (interval != witness) {
            std::cerr << "causal_softmax_attention interval capacity has no exact route witness\n";
            ++failures;
        }
    }
    try {
        (void)ops::causal_softmax_attention_workspace_capacity_bytes(
            {kHeadDim, 16, 2}, KvCacheStorage::BFloat16,
            {1, ops::kCausalAttentionMaximumVisibleKeys}, 1, 1, 1);
    } catch (const std::invalid_argument&) {
        std::cerr << "causal_softmax_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::causal_softmax_attention_workspace_capacity_bytes(
            {kHeadDim, 16, 2}, KvCacheStorage::BFloat16,
            {1, ops::kCausalAttentionMaximumVisibleKeys + 1}, 1, 1, 1);
        std::cerr << "causal_softmax_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}
} // namespace

int run_softmax_attention_nvfp4_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_nvfp4_cases();
    failures += run_quantized_batch_cases(KvCacheStorage::Nvfp4Group16, 720u);
    failures += report_quantization_quality(KvCacheStorage::Nvfp4Group16, 724u);
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention nvfp4 independent correctness\n";
    return failures == 0 ? 0 : 1;
}

int run_softmax_attention_k8v4_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_k8v4_cases();
    failures += run_quantized_batch_cases(KvCacheStorage::Fp8KeyNvfp4Value, 815u);
    failures += report_quantization_quality(KvCacheStorage::Fp8KeyNvfp4Value, 819u);
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention k8v4 independent correctness\n";
    return failures == 0 ? 0 : 1;
}

int run_softmax_attention_fp8_tests() {
    if (cuda_unavailable()) { return 77; }
    int failures = run_fp8_cases();
    failures += run_quantized_batch_cases(KvCacheStorage::Fp8E4M3Row256, 960u);
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention fp8 independent correctness\n";
    return failures == 0 ? 0 : 1;
}

int run_softmax_attention_fp8_ampere_tests() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    int original = 0;
    CUDA_CHECK(cudaGetDevice(&original));
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (properties.major != 8 || properties.minor != 6) continue;
        CUDA_CHECK(cudaSetDevice(device));
        int failures = 0;
        try {
            failures = run_a1_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256,
                                   {1, 128, 192, 1101u}, MappingPattern::Fragmented);
            failures +=
                run_batch_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256,
                               {8, {61, 4091}, {7, 5}, {1, 0}, MappingPattern::Fragmented, 1102u});
            failures += run_batch_case(
                kGeometries[0], KvCacheStorage::Fp8E4M3Row256,
                {8, {63, 129}, {8, 0}, {0, 1}, MappingPattern::Fragmented, 1103u, true});
            failures += run_a3_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256,
                                    {1, 65535, 65536, 1104u}, MappingPattern::Fragmented);
            failures += run_a1_case(kGeometries[0], KvCacheStorage::Fp8E4M3Row256,
                                    {129, 63, 256, 1105u}, MappingPattern::Fragmented);
        } catch (const std::exception& error) {
            std::cerr << "FP8 Ampere attention: " << error.what() << '\n';
            failures = 1;
        }
        CUDA_CHECK(cudaSetDevice(original));
        return failures == 0 ? 0 : 1;
    }
    std::cout << "SKIP: FP8 secondary attention requires an SM86 device\n";
    return 77;
}

int run_distributed_successive_cases(const DistributedDevices& devices);

// Distributed regression cases. Each case is paired with the failure it protects against; the
// set is deliberately small and boundary-focused instead of a wide parameter sweep.
int run_distributed_cases(const DistributedDevices& devices) {
    constexpr KvCacheStorage kStorage = KvCacheStorage::Fp8E4M3Row256;
    const Geometry& wide              = kGeometries[0];
    const Geometry& narrow            = kGeometries[1];
    int failures                      = 0;

    const auto single = [&](const Geometry& geometry, const AttentionCase& test_case,
                            MappingPattern mapping, RemoteCase remote_case,
                            const KeyMagnitude* magnitude = nullptr) {
        return run_a1_case(geometry, kStorage, test_case, mapping, false, &remote_case, &devices,
                           magnitude);
    };
    const auto batch = [&](const Geometry& geometry, const BatchAttentionCase& test_case,
                           RemoteCase remote_case) {
        return run_batch_case(geometry, kStorage, test_case, false, &remote_case, &devices);
    };

    // Mixed contributions: the split lands between the referenced physical IDs, so a single row
    // draws keys from both planes at different depths. Catches a physical-ID partition or merge
    // boundary error that silently drops or double-counts one plane's keys. The second case
    // repeats it through the prompt-width route with a page-crossing visible set.
    failures += single(wide, {1, 128, 192, 1201u}, MappingPattern::Fragmented, RemoteCase::EvenIds);
    failures +=
        single(wide, {129, 63, 256, 1202u}, MappingPattern::Fragmented, RemoteCase::EvenIds);

    // Local-only contribution: every visible key resolves through primary physical IDs and the
    // secondary plane stays entirely unreferenced. Catches a merge that assumes both shards are
    // non-empty, and a secondary launch that reads its unreferenced pages.
    failures +=
        single(wide, {6, 61, 192, 1203u}, MappingPattern::Fragmented, RemoteCase::AllLocalPlanes);

    // Remote-only contribution: every visible key is served from the secondary plane while the
    // primary plane holds one nominally unused ID. Catches the reverse asymmetry: a zero local
    // contribution must be numerically neutral (zero result vector, LSE -inf), not NaN or a
    // stale value, and the secondary launch must not require a nonempty local range.
    failures +=
        single(wide, {6, 61, 192, 1204u}, MappingPattern::Fragmented, RemoteCase::AllRemoteIds);

    // Repeat the empty-shard orientations on both page-crossing prompt routes. The paired
    // H24/W129 and H16/W65 cases distinguish primary compact, secondary compact, and neutral
    // empty-shard reductions beyond the short decode cases above; the mixed-split counterparts
    // are the existing cases at the start of this group.
    failures +=
        single(wide, {129, 63, 256, 1211u}, MappingPattern::Fragmented, RemoteCase::AllLocalPlanes);
    failures +=
        single(wide, {129, 63, 256, 1212u}, MappingPattern::Fragmented, RemoteCase::AllRemoteIds);
    failures += single(narrow, {65, 63, 192, 1213u}, MappingPattern::Fragmented,
                       RemoteCase::AllLocalPlanes);
    failures +=
        single(narrow, {65, 63, 192, 1214u}, MappingPattern::Fragmented, RemoteCase::AllRemoteIds);

    // Semantic boundary: an all-zero query makes every score identical, so both shards report the
    // same partial maximum and the merge has to combine equal maxima rather than assume a strict
    // ordering. A merge that tracks the maximum with a strict comparison, or that rescales only
    // when one shard's maximum is strictly larger, produces a wrong denominator here while the
    // random-query cases still pass. Zero rows are supported by the existing contract, which
    // returns exact BF16 zero for a query row with no visible key.
    failures +=
        single(wide, {6, 61, 192, 1207u, true}, MappingPattern::Fragmented, RemoteCase::EvenIds);

    // Frontier that starts on the last key of one logical page and continues into the next, on a
    // different head grouping. The visible set therefore reaches two pages on opposite sides of
    // the split, so the local shard holds the page-tail keys and the secondary shard holds the
    // following page's keys. Catches a per-page iteration that assumes a full page, an off-by-one
    // at the page boundary, and a shard range that is derived from key indices instead of physical
    // pages.
    failures +=
        single(narrow, {7, 63, 192, 1205u}, MappingPattern::Fragmented, RemoteCase::EvenIds);

    // Uneven non-empty split: the visible set spans three pages, the primary keeps only the lowest
    // referenced page and the secondary owns the other two, so on the widest row the primary shard
    // contributes 64 keys against the secondary's 128. Catches a shard bound, split statistic or
    // merge step that assumes both shards hold comparable amounts of work, and a range derived from
    // half the pool instead of from the partition. The case's declared asymmetry is verified after
    // the call, so a mapping or geometry change cannot leave it running as a balanced split.
    failures +=
        single(wide, {6, 186, 192, 1301u}, MappingPattern::Fragmented, RemoteCase::PrimaryLowestId);

    // Extreme per-shard scale mismatch, in both orientations. Each fixture uses all-ones queries
    // and an all-ones key prototype per side, with factors 1 and 4096. After the FP8 key transform,
    // the unscaled and scaled logits are about 16 and 65,536 respectively. Thus every head/token
    // has a cross-shard LSE contrast far above 100 nats, independent of private query rounding or
    // a near-tied winning key; the fixture check verifies this against the decoded FP8 values.
    // Constant zero/one value regions make an equally weighted shard blend observably wrong. A
    // direct exponentiation of either partial LSE overflows FP32. Mirroring the scale boundary
    // makes the dominant shard remote in one case and local in the other.
    const KeyMagnitude extremes{/*boundary*/ 64, /*low*/ 1.0f, /*high*/ 4096.0f,
                                /*uniform_regions*/ true};
    failures += single(wide, {2, 190, 192, 1311u}, MappingPattern::Fragmented,
                       RemoteCase::ScaleBoundary, &extremes);
    const KeyMagnitude mirrored{/*boundary*/ 128, /*low*/ 4096.0f, /*high*/ 1.0f,
                                /*uniform_regions*/ true};
    failures += single(wide, {2, 190, 192, 1312u}, MappingPattern::Fragmented,
                       RemoteCase::ScaleBoundary, &mirrored);

    // Prompt-width frontier: 65 columns ending exactly on a page boundary, so the visible set
    // spans two full pages. Catches a visible-set bound that drops the final page when the
    // frontier lands on the boundary rather than inside a page.
    failures +=
        single(narrow, {65, 63, 192, 1206u}, MappingPattern::Fragmented, RemoteCase::EvenIds);

    // Successive calls on one object with different inputs, different split points and a
    // shifted page mapping. Outputs are compared on the first call and again after the second,
    // so a stale transfer buffer or a cached remote plane would fail on the second call.
    failures += run_distributed_successive_cases(devices);

    // DFlash2-style widths on the batch route: masked rows with unequal live frontiers, a
    // zero-length row and inert tail columns. Catches per-row visible-set handling that leaks
    // between rows, a tail column that mutates cache, and a split that ignores the per-row mask.
    const int orders[] = {7, 0, 4};
    for (const int width : {2, 8, 16}) {
        BatchAttentionCase test_case{width,
                                     {0, 17, 61},
                                     {width, width - 1, 1},
                                     {orders[0], orders[1], orders[2]},
                                     MappingPattern::Fragmented,
                                     static_cast<std::uint32_t>(1230u + width)};
        failures += batch(wide, test_case, RemoteCase::EvenIds);
    }

    // Two rows with unequal frontiers across the device split: one row's live keys are entirely
    // remote and the other's are entirely local. Catches a per-row page base computed once for
    // the batch and a merge that mixes rows between the two partial-result tensors. The second
    // case repeats it with a zero-length row and three rows spread over the table.
    failures += batch(narrow, {6, {0, 127}, {1, 6}, {0, 1}, MappingPattern::Fragmented, 1250u},
                      RemoteCase::AllRemoteIds);
    failures +=
        batch(wide, {8, {0, 61, 127}, {0, 5, 8}, {2, 0, 1}, MappingPattern::Fragmented, 1240u},
              RemoteCase::EvenIds);

    return failures;
}

// Two complete calls through one DistributedAttention object with different inputs, different
// physical page mappings and different split points. Only the object is shared, so a transfer
// buffer or workspace allocation that is sized or filled once cannot satisfy the second call.
int run_distributed_successive_cases(const DistributedDevices& devices) {
    constexpr KvCacheStorage kStorage = KvCacheStorage::Fp8E4M3Row256;
    const Geometry& geometry          = kGeometries[0];
    const std::int32_t tokens         = 5;
    const std::int32_t base           = 61;
    const std::int32_t total          = base + tokens;
    const std::int32_t max_context    = total + 3;
    const std::size_t q_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads * tokens;
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens;

    std::vector<float> first_q  = make_bf16_values(q_elements, 1301u, -0.25f, 0.25f);
    std::vector<float> first_k  = make_bf16_values(kv_elements, 1302u, -0.25f, 0.25f);
    std::vector<float> first_v  = make_bf16_values(kv_elements, 1303u, -1.0f, 1.0f);
    std::vector<float> second_q = make_bf16_values(q_elements, 1311u, -0.25f, 0.25f);
    std::vector<float> second_k = make_bf16_values(kv_elements, 1312u, -0.25f, 0.25f);
    std::vector<float> second_v = make_bf16_values(kv_elements, 1313u, -1.0f, 1.0f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache first_initial  = make_cache(geometry, kStorage, max_context, 1320u);
    const HostCache second_initial = make_cache(geometry, kStorage, max_context, 1330u);
    HostCache first_expected = first_initial, second_expected = second_initial;
    append_cache(first_expected, first_k, first_v, positions);
    append_cache(second_expected, second_k, second_v, positions);
    const std::vector<double> first_reference = ideal_attention(first_q, first_expected, positions);
    const std::vector<double> second_reference =
        ideal_attention(second_q, second_expected, positions);

    DeviceCache first_cache(first_initial, MappingPattern::Fragmented);
    DeviceCache second_cache(second_initial, MappingPattern::Remote);
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         static_cast<std::uint32_t>(max_context)};
    GuardedDeviceBuffer dq(q_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(kv_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(kv_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer drows(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_elements * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    drows.copy_from_host(&table_row, sizeof(table_row));
    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dp.data(), DType::I32, {tokens});
    Tensor trows(drows.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});

    // One object serves both calls: its startup resources are fixed by the constructor and may not
    // be re-derived from the first call's inputs.
    ops::DistributedAttention attention(devices.primary, devices.secondary, op_geometry(geometry),
                                        tokens, 1, first_cache.batch_view().block_tables.ne[1],
                                        first_cache.batch_view().block_tables.ne[0]);

    int failures = 0;
    const std::array<const char*, 2> names{"first", "second"};
    // Both mappings must leave the even split inside the referenced range; an identity mapping
    // can collapse the referenced pages into adjacent IDs that no split separates.
    const std::array<MappingPattern, 2> mappings{MappingPattern::Fragmented,
                                                 MappingPattern::Remote};
    for (int call = 0; call < 2; ++call) {
        const std::vector<float>& q = call == 0 ? first_q : second_q;
        const std::vector<float>& k = call == 0 ? first_k : second_k;
        const std::vector<float>& v = call == 0 ? first_v : second_v;
        const auto q_bits = to_bf16_bits(q), k_bits = to_bf16_bits(k), v_bits = to_bf16_bits(v);
        dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
        dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
        dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
        std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
        dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

        DeviceCache& cache           = call == 0 ? first_cache : second_cache;
        HostCache& expected          = call == 0 ? first_expected : second_expected;
        const MappingPattern mapping = mappings[static_cast<std::size_t>(call)];
        const std::vector<std::int32_t> table =
            make_block_table(cache.batch_view().block_tables.ne[0], mapping);
        const std::vector<std::int32_t> referenced = referenced_pages(table, total);
        RemoteAttentionFixture remote(
            cache.batch_view(), devices,
            remote_partition(referenced, cache.batch_view().k_pages.ne[3], RemoteCase::EvenIds));
        attention.execute(tq, tk, tv, tp, Tensor{}, trows, kAttentionScale, remote.local(),
                          remote.remote(), envelope, tout, nullptr);
        cuda_synchronize();
        remote.merge();

        const std::string label = std::string("distributed successive ") + names[call] +
                                  " mapping=" + mapping_name(mapping) +
                                  " remote=" + std::to_string(remote.primary_pages()) + "/" +
                                  std::to_string(cache.batch_view().k_pages.ne[3]);
        const auto output       = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
        failures += verify_attention(label, bf16_bits_to_double(output),
                                     call == 0 ? first_reference : second_reference,
                                     attention_criterion(kStorage));
        failures +=
            verify_remote_partition(label, referenced, remote.primary_pages(), RemoteCase::EvenIds);
        failures += verify_cache(label, cache.snapshot(), expected, false);
        failures += verify_input(label + " q unchanged", dq, q_bits);
        failures += verify_input(label + " k unchanged", dk, k_bits);
        failures += verify_input(label + " v unchanged", dv, v_bits);
        failures += verify_positions(label + " positions unchanged", dp, positions);
        failures += dout.verify_guards(label);
        failures += cache.verify_guards(label);
    }
    return failures;
}

int run_softmax_attention_distributed_tests() {
    const DistributedDevices devices = select_distributed_devices();
    if (!devices.selectable) {
        std::cout << "SKIP: distributed attention " << devices.reason << '\n';
        return 77;
    }
    // The route's secondary kernel is compiled for the Ampere pairing it was qualified against, so
    // a host without an SM86 secondary has not exercised the specified case at all. Reporting SKIP
    // keeps the missing hardware visible instead of letting an unqualified pairing print PASS.
    if (!devices.secondary_is_sm86()) {
        std::cout << "SKIP: distributed attention requires an SM86 secondary device; device "
                  << devices.secondary << " is SM" << devices.secondary_major
                  << devices.secondary_minor << '\n';
        return 77;
    }
    const int original = [&] {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        return device;
    }();
    int failures = 0;
    try {
        CUDA_CHECK(cudaSetDevice(devices.primary));
        failures = run_distributed_cases(devices);
    } catch (const std::exception& error) {
        std::cerr << "distributed attention: " << error.what() << '\n';
        failures = 1;
    }
    CUDA_CHECK(cudaSetDevice(original));
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " distributed causal attention two-device independent correctness\n";
    return failures == 0 ? 0 : 1;
}

int run_softmax_attention_causal_cache_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = verify_workspace_capacity_contract();
    failures += run_nvfp4_cases();
    failures += run_quantized_batch_cases(KvCacheStorage::Nvfp4Group16, 720u);
    failures += report_quantization_quality(KvCacheStorage::Nvfp4Group16, 724u);
    failures += run_k8v4_cases();
    failures += run_quantized_batch_cases(KvCacheStorage::Fp8KeyNvfp4Value, 815u);
    failures += report_quantization_quality(KvCacheStorage::Fp8KeyNvfp4Value, 819u);
    for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
    failures += run_fp8_cases();
    failures += run_batch_cases();
    failures += run_dflash2_cases();
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}

int run_softmax_attention_dflash2_tests() {
    if (cuda_unavailable()) return 77;
    const int failures = run_dflash2_cases();
    std::cout << (failures ? "FAIL" : "PASS") << " DFlash2 causal attention\n";
    return failures ? 1 : 0;
}
