// ninfer::ops::detail - row-scaled E4M3FN-cache causal prompt launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/kv_cache/append/kernel.cuh"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh"

#include <atomic>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, bool Tiered, typename CacheView, typename Metadata>
void causal_attention_prompt_fp8_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                      float scale, const CacheView& cache,
                                                      Metadata metadata, Tensor& out,
                                                      cudaStream_t stream) {
    int device = 0, major = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    const bool ampere      = major < 9;
    const int block_rows   = ampere ? kCausalPromptFp8AmpereBr : kCausalPromptFp8Br;
    const int threads      = ampere ? kCausalPromptFp8AmpereThreads : kCausalPromptFp8Threads;
    const int shared_bytes = ampere ? kCausalPromptFp8AmpereSmemBytes : kCausalPromptFp8SmemBytes;
    auto kernel            = causal_attention_prompt_fp8_kernel<Geometry, Metadata, Tiered, false>;
    static std::atomic<unsigned> configured_devices{0};
    const bool trackable      = device >= 0 && device < 32;
    const unsigned device_bit = trackable ? 1u << device : 0u;
    if (!trackable ||
        (configured_devices.load(std::memory_order_relaxed) & device_bit) != device_bit) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        shared_bytes));
        if (trackable) { configured_devices.fetch_or(device_bit, std::memory_order_relaxed); }
    }

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(div_up(tokens, block_rows)),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    kernel<<<grid, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        kv_plane_pointer<const std::uint8_t, Tiered>(cache.k_pages, cache.staged.k),
        kv_plane_pointer<const std::uint8_t, Tiered>(cache.v_pages, cache.staged.v),
        kv_plane_pointer<const __half, Tiered>(cache.k_scale_pages, cache.staged.k_scale),
        kv_plane_pointer<const __half, Tiered>(cache.v_scale_pages, cache.staged.v_scale), metadata,
        static_cast<const std::int32_t*>(positions.data), scale,
        static_cast<__nv_bfloat16*>(out.data), nullptr, tokens, 0, std::numeric_limits<int>::max());
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void causal_attention_prompt_fp8_attention_dispatch(const Tensor& q, const Tensor& positions,
                                                    float scale, const CacheView& cache,
                                                    Metadata metadata, Tensor& out,
                                                    cudaStream_t stream) {
    const auto launch = [&]<typename Geometry>() {
        if (cache.staged.enabled()) {
            causal_attention_prompt_fp8_attention_launch_for<Geometry, true>(
                q, positions, scale, cache, metadata, out, stream);
        } else {
            causal_attention_prompt_fp8_attention_launch_for<Geometry, false>(
                q, positions, scale, cache, metadata, out, stream);
        }
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch.template operator()<CausalD256H24Kv4>();
    } else {
        launch.template operator()<CausalD256H16Kv2>();
    }
}

template <typename Geometry, bool Masked, bool Tiered>
__global__ void fp8_prompt_append_owned_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v, const std::int32_t* positions,
    const std::int32_t* valid_columns, const std::int32_t* table_rows, const std::int32_t* tables,
    std::int32_t table_stride, KVPlanePointer<std::uint8_t, Tiered> cache_k,
    KVPlanePointer<std::uint8_t, Tiered> cache_v, KVPlanePointer<__half, Tiered> cache_k_scale,
    KVPlanePointer<__half, Tiered> cache_v_scale, std::int32_t width, int physical_begin,
    int physical_end) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    int tokens                  = width;
    if constexpr (Masked) { tokens = max(0, min(valid_columns[0], width)); }
    const int unit = static_cast<int>(blockIdx.x) * Warps + warp;
    if (unit >= tokens * Geometry::KVHeads) return;

    const int kv_head  = unit % Geometry::KVHeads;
    const int token    = unit / Geometry::KVHeads;
    const int position = positions[token];
    const std::int32_t* block_table =
        tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page     = __shfl_sync(FullMask, physical_page, 0);
    if (physical_page < physical_begin || physical_page >= physical_end) return;

    ninfer::ops::kv_cache_append_full_fp8_row<Geometry, Tiered>(
        k, v, cache_k, cache_v, cache_k_scale, cache_v_scale, token, kv_head,
        physical_page - physical_begin, position & kPagedKVPageMask, lane);
}

template <typename Geometry, bool Tiered, bool Masked>
void launch_fp8_prompt_append_owned(const Tensor& k, const Tensor& v, const Tensor& positions,
                                    const Tensor& valid_columns, const Tensor& table_rows,
                                    PagedKVBatchLayerView cache, int physical_begin,
                                    int physical_end, cudaStream_t stream) {
    constexpr int Warps = 8;
    const int units     = k.ne[2] * Geometry::KVHeads;
    const dim3 grid(static_cast<unsigned>(div_up(units, Warps)));
    fp8_prompt_append_owned_kernel<Geometry, Masked, Tiered><<<grid, Warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        kv_plane_pointer<std::uint8_t, Tiered>(cache.k_pages, cache.staged.k),
        kv_plane_pointer<std::uint8_t, Tiered>(cache.v_pages, cache.staged.v),
        kv_plane_pointer<__half, Tiered>(cache.k_scale_pages, cache.staged.k_scale),
        kv_plane_pointer<__half, Tiered>(cache.v_scale_pages, cache.staged.v_scale), k.ne[2],
        physical_begin, physical_end);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, bool Tiered, bool Masked>
void launch_fp8_prompt_compact_shard(const Tensor& q, const Tensor& positions,
                                     const Tensor& valid_columns, const Tensor& table_rows,
                                     float scale, PagedKVBatchLayerView cache, int physical_begin,
                                     int physical_end, Tensor& result, cudaStream_t stream) {
    const PagedKVBatchMetadata<Masked> metadata{
        .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
        .valid_columns = Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        .table_rows    = static_cast<const std::int32_t*>(table_rows.data),
        .table_stride  = cache.block_tables.ne[0],
    };
    int device = 0, major = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    const bool ampere      = major < 9;
    const int block_rows   = ampere ? kCausalPromptFp8AmpereBr : kCausalPromptFp8Br;
    const int threads      = ampere ? kCausalPromptFp8AmpereThreads : kCausalPromptFp8Threads;
    const int shared_bytes = ampere ? kCausalPromptFp8AmpereSmemBytes : kCausalPromptFp8SmemBytes;
    auto kernel =
        causal_attention_prompt_fp8_kernel<Geometry, PagedKVBatchMetadata<Masked>, Tiered, true>;
    static std::atomic<unsigned> configured_devices{0};
    const bool trackable      = device >= 0 && device < 32;
    const unsigned device_bit = trackable ? 1u << device : 0u;
    if (!trackable ||
        (configured_devices.load(std::memory_order_relaxed) & device_bit) != device_bit) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        shared_bytes));
        if (trackable) { configured_devices.fetch_or(device_bit, std::memory_order_relaxed); }
    }
    const dim3 grid(static_cast<unsigned>(div_up(q.ne[2], block_rows)),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    kernel<<<grid, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        kv_plane_pointer<const std::uint8_t, Tiered>(cache.k_pages, cache.staged.k),
        kv_plane_pointer<const std::uint8_t, Tiered>(cache.v_pages, cache.staged.v),
        kv_plane_pointer<const __half, Tiered>(cache.k_scale_pages, cache.staged.k_scale),
        kv_plane_pointer<const __half, Tiered>(cache.v_scale_pages, cache.staged.v_scale), metadata,
        static_cast<const std::int32_t*>(positions.data), scale, nullptr,
        static_cast<float*>(result.data), q.ne[2], physical_begin, physical_end);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, bool Tiered, bool Masked>
void launch_fp8_prompt_shard_for(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 int physical_begin, int physical_end, Tensor& result,
                                 cudaStream_t stream) {
    launch_fp8_prompt_append_owned<Geometry, Tiered, Masked>(
        k, v, positions, valid_columns, table_rows, cache, physical_begin, physical_end, stream);
    launch_fp8_prompt_compact_shard<Geometry, Tiered, Masked>(
        q, positions, valid_columns, table_rows, scale, cache, physical_begin, physical_end, result,
        stream);
}

} // namespace

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    causal_attention_prompt_fp8_attention_dispatch(q, positions, scale, cache, metadata, out,
                                                   stream);
}

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
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
        causal_attention_prompt_fp8_attention_dispatch(q, positions, scale, cache, metadata, out,
                                                       stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

void causal_attention_prompt_fp8_shard_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                              const Tensor& positions, const Tensor& valid_columns,
                                              const Tensor& table_rows, float scale,
                                              PagedKVBatchLayerView cache, int physical_begin,
                                              int physical_end, Tensor& result,
                                              cudaStream_t stream) {
    const auto launch_geometry = [&]<typename Geometry>() {
        const auto launch_cache = [&]<bool Tiered>() {
            if (valid_columns.data == nullptr) {
                launch_fp8_prompt_shard_for<Geometry, Tiered, false>(
                    q, k, v, positions, valid_columns, table_rows, scale, cache, physical_begin,
                    physical_end, result, stream);
            } else {
                launch_fp8_prompt_shard_for<Geometry, Tiered, true>(
                    q, k, v, positions, valid_columns, table_rows, scale, cache, physical_begin,
                    physical_end, result, stream);
            }
        };
        if (cache.staged.enabled()) {
            launch_cache.template operator()<true>();
        } else {
            launch_cache.template operator()<false>();
        }
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_geometry.template operator()<CausalD256H24Kv4>();
    } else {
        launch_geometry.template operator()<CausalD256H16Kv2>();
    }
}

} // namespace ninfer::ops::detail
