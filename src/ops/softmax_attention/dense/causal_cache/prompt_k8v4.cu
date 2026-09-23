// ninfer::ops::detail - asymmetric FP8-K/NVFP4-V causal prompt launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_k8v4.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, bool Tiered, typename CacheView, typename Metadata>
void causal_attention_prompt_k8v4_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                       float scale, const CacheView& cache,
                                                       Metadata metadata, Tensor& out,
                                                       cudaStream_t stream) {
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_k8v4_kernel<Geometry, Metadata, Tiered>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptK8V4SmemBytes);
    CUDA_CHECK(attr);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(div_up(tokens, kCausalPromptK8V4Br)),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_k8v4_kernel<Geometry, Metadata, Tiered>
        <<<grid, kCausalPromptK8V4Threads, kCausalPromptK8V4SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            kv_plane_pointer<const std::uint8_t, Tiered>(cache.k_pages, cache.staged.k),
            kv_plane_pointer<const std::uint8_t, Tiered>(cache.v_pages, cache.staged.v),
            kv_plane_pointer<const __half, Tiered>(cache.k_scale_pages, cache.staged.k_scale),
            kv_plane_pointer<const std::uint8_t, Tiered>(cache.v_scale_pages, cache.staged.v_scale),
            metadata, static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void causal_attention_prompt_k8v4_attention_dispatch(const Tensor& q, const Tensor& positions,
                                                     float scale, const CacheView& cache,
                                                     Metadata metadata, Tensor& out,
                                                     cudaStream_t stream) {
    const auto launch = [&]<typename Geometry>() {
        if (cache.staged.enabled()) {
            causal_attention_prompt_k8v4_attention_launch_for<Geometry, true>(
                q, positions, scale, cache, metadata, out, stream);
        } else {
            causal_attention_prompt_k8v4_attention_launch_for<Geometry, false>(
                q, positions, scale, cache, metadata, out, stream);
        }
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch.template operator()<CausalD256H24Kv4>();
    } else {
        launch.template operator()<CausalD256H16Kv2>();
    }
}

} // namespace

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    causal_attention_prompt_k8v4_attention_dispatch(q, positions, scale, cache, metadata, out,
                                                    stream);
}

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache, Tensor& out,
                                         cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        causal_attention_prompt_k8v4_attention_dispatch(q, positions, scale, cache, metadata, out,
                                                        stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
