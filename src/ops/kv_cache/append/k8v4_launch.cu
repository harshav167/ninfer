// ninfer::ops::detail - asymmetric FP8-K/NVFP4-V append launch ownership.
#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/k8v4_kernel.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <typename Geometry, bool Tiered, typename CacheView, typename Metadata>
void launch_k8v4_for(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                     Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    auto cache_k      = kv_plane_pointer<std::uint8_t, Tiered>(cache.k_pages, cache.staged.k);
    auto cache_v      = kv_plane_pointer<std::uint8_t, Tiered>(cache.v_pages, cache.staged.v);
    auto scale_k      = kv_plane_pointer<__half, Tiered>(cache.k_scale_pages, cache.staged.k_scale);
    auto scale_v =
        kv_plane_pointer<std::uint8_t, Tiered>(cache.v_scale_pages, cache.staged.v_scale);
    if (tokens >= 128 && Geometry::KVHeads == 2) {
        constexpr int TokensPerTile = 8;
        const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
        const dim3 grid(static_cast<unsigned>(max_tiles), static_cast<unsigned>(Geometry::KVHeads));
        kv_cache_append_full_k8v4_page_kernel<Geometry, Metadata, Tiered>
            <<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data),
                                          static_cast<const std::int32_t*>(positions.data),
                                          metadata, cache_k, cache_v, scale_k, scale_v, tokens);
    } else {
        constexpr int FillWarps       = kBlock / 32;
        const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
        const int grid = static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
        kv_cache_append_full_k8v4_kernel<Geometry, Metadata, Tiered><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata, cache_k, cache_v, scale_k,
            scale_v, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void dispatch_k8v4(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                   Metadata metadata, cudaStream_t stream) {
    const auto launch = [&]<typename Geometry>() {
        if (cache.staged.enabled()) {
            launch_k8v4_for<Geometry, true>(k, v, positions, cache, metadata, stream);
        } else {
            launch_k8v4_for<Geometry, false>(k, v, positions, cache, metadata, stream);
        }
    };
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch.template operator()<KVCacheAppendD256Kv4>();
    } else {
        launch.template operator()<KVCacheAppendD256Kv2>();
    }
}

} // namespace

void kv_cache_append_k8v4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                 PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch_k8v4(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_k8v4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       const Tensor& valid_columns, const Tensor& table_rows,
                                       PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        dispatch_k8v4(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
