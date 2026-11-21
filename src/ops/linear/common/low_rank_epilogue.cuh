#pragma once

#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/low_rank.h"
#include "ops/common/numeric/math.cuh"

namespace ninfer::ops::detail {

// B*z for one output element (or an adjacent row pair sharing z). B is
// read through the read-only path, coalesced across the rows a warp owns
// at each rank.
// Deliberately not inlined: these run inside register-saturated MMA
// epilogues, where inlining the dot spills the accumulator tile to per-
// thread stack (multi-KB/thread at prefill, whose backing allocation can
// fail at launch). One call per output element keeps the caller's
// register allocation identical to the unadapted epilogue.
static __device__ __noinline__ float rank_dot(RankDelta d, int row, int token) {
    // One accumulator chain: this helper must stay under the register cap
    // of every kernel that calls it (small GEMV epilogues cap at 40).
    float sum     = 0.0f;
    const auto* b = d.b + row;
    const auto* z = d.z + std::int64_t(token) * kLowRank;
#pragma unroll 4
    for (int r = 0; r < kLowRank; ++r) {
        sum = fmaf(__ldg(b), z[r], sum);
        b += d.rows;
    }
    return sum;
}

static __device__ __noinline__ float2 rank_dot_pair(RankDelta d, int row, int token) {
    float2 sum{};
    const auto* b  = reinterpret_cast<const float2*>(d.b + row);
    const auto* z  = d.z + std::int64_t(token) * kLowRank;
    const int step = d.rows / 2;
#pragma unroll 4
    for (int r = 0; r < kLowRank; ++r) {
        const auto w = *b;
        sum.x        = fmaf(w.x, z[r], sum.x);
        sum.y        = fmaf(w.y, z[r], sum.y);
        b += step;
    }
    return sum;
}

__device__ __forceinline__ float rank_add(RankDelta d, int row, int token, float base) {
    return fmaf(d.scale, rank_dot(d, row, token), base);
}

__device__ __forceinline__ float2 rank_add_pair(RankDelta d, int row, int token, float2 base) {
    const auto v = rank_dot_pair(d, row, token);
    return make_float2(fmaf(d.scale, v.x, base.x), fmaf(d.scale, v.y, base.y));
}

struct RankResidualAdd {
    LinearBf16InputView residual;
    RankDelta delta;

    __device__ __forceinline__ float apply(int row, int token, float base) const {
        return rank_add(delta, row, token, base) + residual.load(row, token);
    }

    __device__ __forceinline__ float2 apply_row_pair(int row, int token, float2 base) const {
        const auto v = rank_add_pair(delta, row, token, base);
        return make_float2(v.x + residual.load(row, token), v.y + residual.load(row + 1, token));
    }
};

struct RankSwiGluProduct {
    RankDelta gate, up;

    __device__ __forceinline__ float operator()(int row, int token, float g, float u) const {
        return common::silu(rank_add(gate, row, token, g)) * rank_add(up, row, token, u);
    }

    __device__ __forceinline__ float2 apply_pair(int row, int token, float2 g, float2 u) const {
        g = rank_add_pair(gate, row, token, g);
        u = rank_add_pair(up, row, token, u);
        return make_float2(common::silu(g.x) * u.x, common::silu(g.y) * u.y);
    }
};

template <int First, int Second, int Third>
struct RankSplit {
    RankDelta slices[4];

    __device__ __forceinline__ RankDelta segment(int& row) const {
        if (row < First) return slices[0];
        row -= First;
        if (row < Second) return slices[1];
        row -= Second;
        if (row < Third) return slices[2];
        row -= Third;
        return slices[3];
    }

    __device__ __forceinline__ float apply(int row, int token, float base) const {
        const auto d = segment(row);
        return rank_add(d, row, token, base);
    }

    __device__ __forceinline__ float2 apply_row_pair(int row, int token, float2 base) const {
        const auto d = segment(row);
        return rank_add_pair(d, row, token, base);
    }
};

} // namespace ninfer::ops::detail
