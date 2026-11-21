#include "ops/linear/common/low_rank.h"

#include <cuda_bf16.h>
#include <array>

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {

struct ShrinkInputs {
    const float* a[detail::RankDeltas{}.size()]{};
};

// Small token counts (decode, speculative verification): one CTA per rank
// reduces K across 128 threads; each A element is loaded once and reused
// for every token in the CTA's tile. A is FP32 [rank][K]; x is BF16 [K,T]
// with K contiguous, so lanes reading consecutive K are coalesced.
constexpr int kShrinkThreads = 128;

template <int Tokens>
__global__ void shrink_small(const __nv_bfloat16* x, ShrinkInputs inputs, float* z, int k,
                             int tokens) {
    __shared__ float partial[Tokens][kShrinkThreads / 32];
    const int rank  = blockIdx.x;
    const int first = blockIdx.y * Tokens;
    const auto* a   = inputs.a[blockIdx.z];
    z += std::size_t(blockIdx.z) * tokens * kLowRank;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float sums[Tokens]{};
    for (int i = threadIdx.x; i < k; i += kShrinkThreads) {
        const float w = a[rank * k + i];
#pragma unroll
        for (int j = 0; j < Tokens; ++j) {
            if (first + j < tokens)
                sums[j] = fmaf(w, __bfloat162float(x[(first + j) * k + i]), sums[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < Tokens; ++j) {
        for (int d = 16; d > 0; d /= 2) sums[j] += __shfl_down_sync(0xffffffff, sums[j], d);
        if (lane == 0) partial[j][warp] = sums[j];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < Tokens; ++j) {
            if (first + j >= tokens) break;
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < kShrinkThreads / 32; ++w) sum += partial[j][w];
            z[(first + j) * kLowRank + rank] = sum;
        }
    }
}

// Wide token counts (prefill): a CTA owns 8 ranks x 32 tokens and streams K
// in 128-wide tiles staged once in shared memory. Each warp owns one rank;
// a lane loads one A element per K step and reuses it across all 32 token
// accumulators, so A traffic is one pass per 32-token tile (L2-resident
// between tiles) and x traffic is one shared staging per tile.
constexpr int kWideTokens = 32;
constexpr int kWideRanks  = 8;
constexpr int kWideK      = 128;

__global__ void shrink_wide(const __nv_bfloat16* x, ShrinkInputs inputs, float* z, int k,
                            int tokens) {
    __shared__ __nv_bfloat16 tile[kWideTokens][kWideK];
    const int lane  = threadIdx.x % 32;
    const int rank  = blockIdx.x * kWideRanks + threadIdx.x / 32;
    const int first = blockIdx.y * kWideTokens;
    const auto* a   = inputs.a[blockIdx.z] + std::size_t(rank) * k;
    float sums[kWideTokens]{};
    for (int begin = 0; begin < k; begin += kWideK) {
        for (int i = threadIdx.x; i < kWideTokens * kWideK; i += blockDim.x) {
            const int token = first + i / kWideK;
            const int col   = begin + i % kWideK;
            tile[i / kWideK][i % kWideK] =
                token < tokens && col < k ? x[std::size_t(token) * k + col] : __nv_bfloat16{};
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kWideK; i += 32) {
            const int col = begin + i + lane;
            const float w = col < k ? a[col] : 0.0f;
#pragma unroll
            for (int t = 0; t < kWideTokens; ++t)
                sums[t] = fmaf(w, __bfloat162float(tile[t][i + lane]), sums[t]);
        }
        __syncthreads();
    }
    auto* out = z + std::size_t(blockIdx.z) * tokens * kLowRank + rank;
#pragma unroll
    for (int t = 0; t < kWideTokens; ++t) {
        for (int d = 16; d > 0; d /= 2) sums[t] += __shfl_down_sync(0xffffffff, sums[t], d);
        if (lane == 0 && first + t < tokens) out[std::size_t(first + t) * kLowRank] = sums[t];
    }
}

bool matrix(const Tensor& v, DType dtype, Device device) {
    return v.data && v.dtype == dtype && v.device == device && v.ne[0] > 0 && v.ne[1] > 0 &&
           v.ne[2] == 1 && v.ne[3] == 1 && v.nb[0] == dtype_size(dtype) &&
           v.nb[1] >= v.ne[0] * v.nb[0] && v.nb[1] % v.nb[0] == 0;
}

} // namespace

detail::RankDeltas detail::prepare_low_rank(const Tensor& x,
                                            std::span<const LowRankFactors> factors,
                                            std::span<const int> rows, float scale,
                                            MemoryArena& workspace, CudaExecutionView execution) {
    RankDeltas result{};
    const auto device = x.device;
    const int tokens  = x.ne[1];
    if (device != execution.device() || workspace.device() != device ||
        device.type() != DeviceType::CUDA || !matrix(x, DType::BF16, device) ||
        !x.is_contiguous() || factors.empty() || factors.size() > result.size() ||
        rows.size() != factors.size() || !std::isfinite(scale))
        throw std::invalid_argument("low rank: invalid input, slice count or scale");
    ShrinkInputs sources{};
    std::array<int, RankDeltas{}.size()> indices{};
    int distinct = 0;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        const auto& [a, b] = factors[i];
        if (!matrix(a, DType::FP32, device) || !a.is_contiguous() || a.ne[0] != x.ne[0] ||
            a.ne[1] != kLowRank || !matrix(b, DType::FP32, device) || !b.is_contiguous() ||
            b.ne[0] != rows[i] || b.ne[1] != kLowRank || (rows[i] % 2) != 0 ||
            reinterpret_cast<std::uintptr_t>(b.data) % alignof(float2) != 0)
            throw std::invalid_argument("low rank: incompatible A/B slice");
        const auto* input = static_cast<const float*>(a.data);
        int j             = 0;
        while (j < distinct && sources.a[j] != input) ++j;
        if (j == distinct) sources.a[distinct++] = input;
        indices[i] = j;
    }
    auto* z = reinterpret_cast<float*>(
        workspace.alloc_bytes(low_rank_workspace_bytes(tokens, distinct), 256).data);
    const auto launch_small = [&]<int Tiles>() {
        const auto groups = (tokens + Tiles - 1) / Tiles;
        shrink_small<Tiles>
            <<<dim3(kLowRank, groups, distinct), kShrinkThreads, 0, execution.stream()>>>(
                static_cast<const __nv_bfloat16*>(x.data), sources, z, x.ne[0], tokens);
    };
    if (tokens == 1)
        launch_small.template operator()<1>();
    else if (tokens <= 4)
        launch_small.template operator()<4>();
    else if (tokens <= 16)
        launch_small.template operator()<8>();
    else
        shrink_wide<<<dim3(kLowRank / kWideRanks, (tokens + kWideTokens - 1) / kWideTokens,
                           distinct),
                      kWideRanks * 32, 0, execution.stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), sources, z, x.ne[0], tokens);
    CUDA_CHECK(cudaGetLastError());
    for (std::size_t i = 0; i < factors.size(); ++i)
        result[i] = {static_cast<const float*>(factors[i].b.data),
                     z + std::size_t(indices[i]) * tokens * kLowRank, rows[i], scale};
    return result;
}

} // namespace ninfer::ops
