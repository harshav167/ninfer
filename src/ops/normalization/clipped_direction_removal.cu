#include "ninfer/ops/clipped_direction_removal.h"
#include "core/device.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/convert.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

inline constexpr const char* kOp           = "clipped_direction_removal";
inline constexpr int kBlock                = 256;
inline constexpr std::int32_t kPackWidth   = 8; // BF16 values per 16-byte vector
inline constexpr std::int32_t kMaxChannels = 16384;

__device__ __forceinline__ void unpack(uint4 bits, float (&x)[kPackWidth]) {
    const float2 a = common::bf16x2_bits_to_float2(bits.x);
    const float2 b = common::bf16x2_bits_to_float2(bits.y);
    const float2 c = common::bf16x2_bits_to_float2(bits.z);
    const float2 d = common::bf16x2_bits_to_float2(bits.w);
    x[0]           = a.x;
    x[1]           = a.y;
    x[2]           = b.x;
    x[3]           = b.y;
    x[4]           = c.x;
    x[5]           = c.y;
    x[6]           = d.x;
    x[7]           = d.y;
}

__device__ __forceinline__ uint4 pack_rn(const float (&x)[kPackWidth]) {
    return make_uint4(common::pack_bf16x2(x[0], x[1]), common::pack_bf16x2(x[2], x[3]),
                      common::pack_bf16x2(x[4], x[5]), common::pack_bf16x2(x[6], x[7]));
}

__device__ __forceinline__ void load_direction(const float* direction, int pack,
                                               float (&d)[kPackWidth]) {
    const float4 lo = common::load_ldg<float4>(direction + pack * kPackWidth);
    const float4 hi = common::load_ldg<float4>(direction + pack * kPackWidth + 4);
    d[0]            = lo.x;
    d[1]            = lo.y;
    d[2]            = lo.z;
    d[3]            = lo.w;
    d[4]            = hi.x;
    d[5]            = hi.y;
    d[6]            = hi.z;
    d[7]            = hi.w;
}

__device__ __forceinline__ void accumulate(float (&sum)[kPackWidth], const float (&x)[kPackWidth],
                                           const float (&d)[kPackWidth]) {
#pragma unroll
    for (int i = 0; i < kPackWidth; ++i) { sum[i] = fmaf(x[i], d[i], sum[i]); }
}

__device__ __forceinline__ float fold(const float (&s)[kPackWidth]) {
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

__device__ __forceinline__ void remove_direction(float (&x)[kPackWidth], float coefficient,
                                                 const float (&d)[kPackWidth]) {
#pragma unroll
    for (int i = 0; i < kPackWidth; ++i) { x[i] = fmaf(-coefficient, d[i], x[i]); }
}

// One CTA owns one column. Each step's coefficient depends on the whole column, so every step is
// a pass over it; the passes re-read the column and the directions instead of staging them, which
// keeps the resources independent of H. A thread owns the packs p = tid, tid + kBlock, ... in every
// pass, so the in-place store of the last pass never races with another thread's reads. A step
// whose coefficient is 0 is skipped rather than applied, which keeps untouched columns bit-exact.
__launch_bounds__(kBlock) __global__
    void clipped_direction_removal_kernel(const float* __restrict__ directions,
                                          const float* __restrict__ references,
                                          uint4* __restrict__ residual, std::int32_t packs) {
    __shared__ float warp_sums[kBlock / common::kWarpSize];
    __shared__ float total;
    const auto reduce = [&](float value) {
        value = common::block_reduce_sum<kBlock>(value, warp_sums);
        if (threadIdx.x == 0) { total = value; }
        __syncthreads();
        return total;
    };

    const float* u0 = directions;
    const float* u1 = directions + static_cast<std::int64_t>(packs) * kPackWidth;
    uint4* column   = residual + static_cast<std::int64_t>(blockIdx.x) * packs;
    const float2 c  = common::load_ldg<float2>(references);

    float sum0[kPackWidth] = {};
    for (int p = threadIdx.x; p < packs; p += kBlock) {
        float x[kPackWidth], d[kPackWidth];
        unpack(column[p], x);
        load_direction(u0, p, d);
        accumulate(sum0, x, d);
    }
    const float a0 = fmaxf(reduce(fold(sum0)) - c.x, 0.0f);

    float sum1[kPackWidth] = {};
    for (int p = threadIdx.x; p < packs; p += kBlock) {
        float x[kPackWidth], d[kPackWidth];
        unpack(column[p], x);
        if (a0 > 0.0f) {
            load_direction(u0, p, d);
            remove_direction(x, a0, d);
        }
        load_direction(u1, p, d);
        accumulate(sum1, x, d);
    }
    const float a1 = fmaxf(reduce(fold(sum1)) - c.y, 0.0f);
    if (!(a0 > 0.0f) && !(a1 > 0.0f)) { return; }

    for (int p = threadIdx.x; p < packs; p += kBlock) {
        float x[kPackWidth], d[kPackWidth];
        unpack(column[p], x);
        if (a0 > 0.0f) {
            load_direction(u0, p, d);
            remove_direction(x, a0, d);
        }
        if (a1 > 0.0f) {
            load_direction(u1, p, d);
            remove_direction(x, a1, d);
        }
        column[p] = pack_rn(x);
    }
}

void launch_clipped_direction_removal(const Tensor& directions, const Tensor& references,
                                      Tensor& residual, cudaStream_t stream) {
    clipped_direction_removal_kernel<<<static_cast<unsigned>(residual.ne[1]), kBlock, 0, stream>>>(
        static_cast<const float*>(directions.data), static_cast<const float*>(references.data),
        static_cast<uint4*>(residual.data), residual.ne[0] / kPackWidth);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void clipped_direction_removal(const Tensor& directions, const Tensor& references, Tensor& residual,
                               CudaExecutionView execution) {
    namespace operand = common::operand;
    operand::dtype(residual, DType::BF16, kOp, "residual must be BF16");
    if (residual.ne[1] < 0 || residual.ne[2] != 1 || residual.ne[3] != 1) {
        throw std::invalid_argument(std::string(kOp) + ": residual must have shape [H,T]");
    }
    const std::int32_t channels = residual.ne[0];
    const std::int32_t columns  = residual.ne[1];
    if (channels < kPackWidth || channels > kMaxChannels || channels % kPackWidth != 0) {
        throw std::invalid_argument(std::string(kOp) + ": H must be a multiple of 8 in [8, 16384]");
    }
    const Device device = execution.device();
    operand::tensor(directions, DType::FP32, {channels, 2}, 16, kOp, "directions", device);
    operand::tensor(references, DType::FP32, {2}, 8, kOp, "references", device);
    if (columns == 0) { return; }
    operand::tensor(residual, DType::BF16, {channels, columns}, 16, kOp, "residual", device);
    const auto written = operand::range(residual, kOp);
    operand::disjoint(written, operand::range(directions, kOp), kOp);
    operand::disjoint(written, operand::range(references, kOp), kOp);

    launch_clipped_direction_removal(directions, references, residual, execution.stream());
}

} // namespace ninfer::ops
