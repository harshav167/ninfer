// ninfer::ops::detail - FP8 E4M3FN split-KV small-T launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

// Declares this file's shard-workspace capacity query, so the definition is checked against the
// interface its callers compile against.
#include "ops/softmax_attention/dense/causal_cache/distributed_launch.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_fp8.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
void causal_attention_prompt_fp8_shard_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                              const Tensor& positions, const Tensor& valid_columns,
                                              const Tensor& table_rows, float scale,
                                              PagedKVBatchLayerView cache, int physical_begin,
                                              int physical_end, Tensor& result,
                                              cudaStream_t stream);

namespace {
constexpr std::int32_t kWideShardPromptMinimumWidth = 64;

// A device's compute capability never changes during a process, and a two-device shard call makes
// this query once per chunk per device. Caching it per device index keeps the per-round path to
// one thread-local device read.
int cached_compute_major(int device) {
    constexpr int kCachedDevices = 64;
    static std::atomic<int> majors[kCachedDevices];
    if (device >= 0 && device < kCachedDevices) {
        const int cached = majors[device].load(std::memory_order_relaxed);
        if (cached > 0) { return cached; }
    }
    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (device >= 0 && device < kCachedDevices) {
        majors[device].store(major, std::memory_order_relaxed);
    }
    return major;
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          bool Tiered>
void launch_fp8_partial(const Tensor& q, CacheInput input, const Tensor& positions, float scale,
                        PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                        std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                        Tensor& partial_m, Tensor& partial_l, std::int32_t physical_begin,
                        std::int32_t physical_end, cudaStream_t stream) {
    constexpr int RowCount  = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles  = (RowCount + 15) / 16;
    constexpr int Warps     = RowTiles == 3 ? 12 : 8;
    constexpr int MinBlocks = TokenTile == 1 ? 2 : 1;
    using KernelInput       = CacheInput;
    int device              = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const int major = cached_compute_major(device);
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    const auto launch = [&]<int KeyBlock>() {
        const std::size_t DynamicBytes = (major < 9 ? 6u : 4u) * KeyBlock * kCausalHeadDim;
        auto kernel =
            causal_attention_small_t_fp8_tiled_kernel<Geometry, TokenTile, Warps, MinBlocks,
                                                      KeyBlock, true, MultiBatch, Masked,
                                                      KernelInput, Tiered>;
        // The attribute is per-device state of one kernel instantiation. A shard call launches one
        // chunk per tile width, so caching per device removes a driver round trip from every chunk
        // after the first without ever configuring a device with another device's value.
        static std::atomic<unsigned> configured_devices{0};
        const bool trackable      = device >= 0 && device < 32;
        const unsigned device_bit = trackable ? 1u << device : 0u;
        if (!trackable ||
            (configured_devices.load(std::memory_order_relaxed) & device_bit) != device_bit) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(DynamicBytes)));
            if (trackable) { configured_devices.fetch_or(device_bit, std::memory_order_relaxed); }
        }

        const auto q_ptr         = static_cast<const __nv_bfloat16*>(q.data);
        const auto positions_ptr = static_cast<const std::int32_t*>(positions.data);
        const auto cache_k_ptr =
            kv_plane_pointer<std::uint8_t, Tiered>(cache.k_pages, cache.staged.k);
        const auto cache_v_ptr =
            kv_plane_pointer<std::uint8_t, Tiered>(cache.v_pages, cache.staged.v);
        const auto k_scale_ptr =
            kv_plane_pointer<__half, Tiered>(cache.k_scale_pages, cache.staged.k_scale);
        const auto v_scale_ptr =
            kv_plane_pointer<__half, Tiered>(cache.v_scale_pages, cache.staged.v_scale);
        const auto tables_ptr = static_cast<const std::int32_t*>(cache.block_tables.data);
        const auto valid_ptr =
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data);
        const auto rows_ptr   = invocation.table_rows == nullptr
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(invocation.table_rows->data);
        auto* partial_acc_ptr = static_cast<float*>(partial_acc.data);
        auto* partial_m_ptr   = static_cast<float*>(partial_m.data);
        auto* partial_l_ptr   = static_cast<float*>(partial_l.data);
        kernel<<<grid, Warps * 32, DynamicBytes, stream>>>(
            q_ptr, input, positions_ptr, cache_k_ptr, cache_v_ptr, k_scale_ptr, v_scale_ptr,
            tables_ptr, valid_ptr, rows_ptr, cache.block_tables.ne[0], invocation.full_width,
            invocation.column_begin, logical_capacity, scale, partial_acc_ptr, partial_m_ptr,
            partial_l_ptr, physical_begin, physical_end);
        CUDA_CHECK(cudaGetLastError());
    };

    if (major < 9 || TokenTile == 1) {
        launch.template operator()<32>();
    } else {
        launch.template operator()<64>();
    }
}

template <typename Geometry, bool MultiBatch, bool Masked>
void launch_fp8_reduce(const Tensor& positions, const CausalSmallTInvocation& invocation,
                       std::int32_t splits, const Tensor& partial_acc, const Tensor& partial_m,
                       const Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    constexpr int Block = 256;

    constexpr int DChunk = Geometry::QHeads == 24 ? 256 : 64;
    const auto launch    = [&]<bool Offset>() {
        const dim3 grid(Geometry::QHeads, div_up(kCausalHeadDim, DChunk),
                        invocation.width * invocation.batch_size);
        causal_attention_small_t_fp8_reduce_output_kernel<Geometry, DChunk, MultiBatch, Masked,
                                                          Offset><<<grid, Block, 0, stream>>>(
            static_cast<const float*>(partial_acc.data), static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data),
            static_cast<const std::int32_t*>(positions.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.width, invocation.full_width, invocation.column_begin, invocation.batch_size,
            splits, static_cast<__nv_bfloat16*>(out.data));
    };
    if (invocation.column_begin == 0)
        launch.template operator()<false>();
    else
        launch.template operator()<true>();
    CUDA_CHECK(cudaGetLastError());
}

// Compact-shard reduction of one device's own page range. Its output is the FP32
// [257,Hq,W,B] tensor the cross-device merge consumes, so a two-device call can
// publish exactly this shard and never exchange unnormalized numerators.
template <typename Geometry>
void launch_fp8_compact(const Tensor& positions, const CausalSmallTInvocation& invocation,
                        std::int32_t splits, const Tensor& partial_acc, const Tensor& partial_m,
                        const Tensor& partial_l, Tensor& result, cudaStream_t stream) {
    const std::int32_t columns = invocation.width * invocation.batch_size;
    const auto blocks          = static_cast<unsigned>(columns * Geometry::QHeads);
    causal_attention_small_t_fp8_compact_kernel<Geometry>
        <<<blocks, kCausalFp8CompactBlock, 0, stream>>>(
            static_cast<const float*>(partial_acc.data), static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data),
            static_cast<const std::int32_t*>(positions.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.width, invocation.full_width, invocation.column_begin, splits,
            static_cast<float*>(result.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_fp8_launch_for(
    const Tensor& q, CacheInput input, const Tensor& positions, float scale,
    PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
    CausalAttentionExecutionEnvelope envelope, std::int32_t physical_begin,
    std::int32_t physical_end, bool compact_result, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits           = causal_attention_split_capacity(
        Geometry::QHeads, invocation.width, cache.storage, envelope, invocation.batch_size);

    const auto launch_partial = [&]<int Tokens, bool MultiBatch, bool Masked>() {
        if (cache.staged.enabled()) {
            launch_fp8_partial<Geometry, Tokens, MultiBatch, Masked, CacheInput, true>(
                q, input, positions, scale, cache, invocation, logical_capacity, splits,
                partial_acc, partial_m, partial_l, physical_begin, physical_end, stream);
        } else {
            launch_fp8_partial<Geometry, Tokens, MultiBatch, Masked, CacheInput, false>(
                q, input, positions, scale, cache, invocation, logical_capacity, splits,
                partial_acc, partial_m, partial_l, physical_begin, physical_end, stream);
        }
    };
    const auto dispatch_metadata = [&]<int Tokens>() {
        const bool masked = invocation.valid_columns != nullptr;
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_partial.template operator()<Tokens, false, true>();
            } else {
                launch_partial.template operator()<Tokens, false, false>();
            }
        } else if (masked) {
            launch_partial.template operator()<Tokens, true, true>();
        } else {
            launch_partial.template operator()<Tokens, true, false>();
        }
    };

    switch (invocation.width) {
    case 1:
        dispatch_metadata.template operator()<1>();
        break;
    case 2:
        dispatch_metadata.template operator()<2>();
        break;
    case 3:
        dispatch_metadata.template operator()<3>();
        break;
    case 4:
        dispatch_metadata.template operator()<4>();
        break;
    case 5:
        dispatch_metadata.template operator()<5>();
        break;
    case 6:
        dispatch_metadata.template operator()<6>();
        break;
    case 7:
        if constexpr (Geometry::QHeads == 24) {
            dispatch_metadata.template operator()<7>();
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    case 8:
        if constexpr (Geometry::QHeads == 24) {
            dispatch_metadata.template operator()<8>();
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    default:
        throw std::invalid_argument("causal_attention_small_t_fp8_launch: unsupported T");
    }

    if (compact_result) {
        launch_fp8_compact<Geometry>(positions, invocation, splits, partial_acc, partial_m,
                                     partial_l, out, stream);
        return;
    }
    // The ordinary route publishes the final BF16 row directly. Its physical range is the whole
    // global namespace, so the tiled kernel's page filter stays inert.
    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_fp8_reduce<Geometry, false, true>(positions, invocation, splits, partial_acc,
                                                     partial_m, partial_l, out, stream);
        } else {
            launch_fp8_reduce<Geometry, false, false>(positions, invocation, splits, partial_acc,
                                                      partial_m, partial_l, out, stream);
        }
    } else if (masked) {
        launch_fp8_reduce<Geometry, true, true>(positions, invocation, splits, partial_acc,
                                                partial_m, partial_l, out, stream);
    } else {
        launch_fp8_reduce<Geometry, true, false>(positions, invocation, splits, partial_acc,
                                                 partial_m, partial_l, out, stream);
    }
}

// Cross-device merge. Both shard tensors are FP32 [257,Hq,W,B] contiguous, so the flat column
// index W*batch + column addresses the same component plane in every tensor.
template <typename Geometry>
void launch_fp8_merge(const Tensor& first, const Tensor& second, Tensor& out, cudaStream_t stream) {
    const std::int32_t columns = first.ne[2] * first.ne[3];
    const auto blocks          = static_cast<unsigned>(columns * Geometry::QHeads);
    causal_attention_small_t_fp8_merge_kernel<Geometry>
        <<<blocks, kCausalFp8MergeBlock, 0, stream>>>(
            static_cast<const float*>(first.data), static_cast<const float*>(second.data),
            static_cast<__nv_bfloat16*>(out.data), columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_fp8_launch_for<CausalD256H24Kv4>(
            q, input, positions, scale, cache, invocation, envelope, 0, kCausalFp8AllLocalPages,
            false, partial_acc, partial_m, partial_l, out, stream);
        return;
    }
    causal_attention_small_t_fp8_launch_for<CausalD256H16Kv2>(
        q, input, positions, scale, cache, invocation, envelope, 0, kCausalFp8AllLocalPages, false,
        partial_acc, partial_m, partial_l, out, stream);
}

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream) {
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_fp8_launch_for<CausalD256H24Kv4>(
            q, input, positions, scale, batch_cache, invocation, envelope, 0,
            kCausalFp8AllLocalPages, false, partial_acc, partial_m, partial_l, out, stream);
        return;
    }
    causal_attention_small_t_fp8_launch_for<CausalD256H16Kv2>(
        q, input, positions, scale, batch_cache, invocation, envelope, 0, kCausalFp8AllLocalPages,
        false, partial_acc, partial_m, partial_l, out, stream);
}

// One device's contribution to a two-device call. The cache view resolves global physical page IDs
// through the caller's block table, and only IDs inside [physical_begin, physical_end) are read or
// appended by this device. The published result is the compact FP32 [257,Hq,W,B] tensor that
// fp8_attention_merge combines, so no unnormalized numerator or KV byte crosses the link.
//
// The query block, its new K/V rows and all metadata live on the executing device. A range with no
// visible key still runs the complete path: every split reports the neutral (m=-inf, l=0) state and
// the compact kernel publishes zero components with an LSE of -inf, which the merge drops exactly.
namespace {

// The transient allocation recipe one shard chunk consumes. The launcher and the shard's capacity
// query both call this, so an arena sized from the query is never short mid-call.
struct Fp8ShardChunkWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
Fp8ShardChunkWorkspace allocate_fp8_shard_chunk(Allocator& workspace, std::int32_t q_heads,
                                                std::int32_t tokens, std::int32_t splits,
                                                std::int32_t batch_size) {
    return {
        workspace.alloc(DType::FP32, {kCausalHeadDim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
    };
}

template <typename Geometry>
void fp8_attention_shard_for(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid, const Tensor& rows,
                             float scale, PagedKVBatchLayerView cache,
                             CausalAttentionExecutionEnvelope envelope, int physical_begin,
                             int physical_end, WorkspaceArena& workspace, Tensor& result,
                             cudaStream_t stream) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    // Distributed shards use the tiled prompt kernel for wide single-batch prefill: its SM86
    // variant uses FP16 Tensor Cores, while SM120 retains native FP8. The compact route also emits
    // the neutral result immediately when this shard owns no visible page. Public attention routes
    // are unchanged; decode and shorter shard calls keep the split-KV small-T path.
    if (q.ne[3] == 1 && q.ne[2] >= kWideShardPromptMinimumWidth &&
        cached_compute_major(device) >= 8) {
        causal_attention_prompt_fp8_shard_launch(q, k, v, positions, valid, rows, scale, cache,
                                                 physical_begin, physical_end, result, stream);
        return;
    }
    constexpr int kMaximumTile = Geometry::QHeads == 24 ? 8 : 6;
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    // The tiled kernel advances q/k/v/positions by the invocation's column_begin itself, so every
    // chunk is launched with the full tensors and the compact kernel writes that chunk's columns.
    for (std::int32_t begin = 0; begin < q.ne[2];) {
        const std::int32_t count  = std::min(kMaximumTile, q.ne[2] - begin);
        auto chunk_scope          = workspace.scope();
        const std::int32_t splits = causal_attention_split_capacity(
            Geometry::QHeads, count, cache.storage, envelope, q.ne[3]);
        Fp8ShardChunkWorkspace chunk =
            allocate_fp8_shard_chunk(workspace, Geometry::QHeads, count, splits, q.ne[3]);
        const CausalSmallTInvocation invocation{
            .valid_columns = valid.data == nullptr ? nullptr : &valid,
            .table_rows    = &rows,
            .full_width    = q.ne[2],
            .column_begin  = begin,
            .width         = count,
            .batch_size    = q.ne[3],
        };
        causal_attention_small_t_fp8_launch_for<Geometry>(
            q, input, positions, scale, cache, invocation, envelope, physical_begin, physical_end,
            true, chunk.acc, chunk.m, chunk.l, result, stream);
        begin += count;
    }
}

} // namespace

void fp8_attention_shard(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                         const Tensor& valid, const Tensor& rows, float scale,
                         PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                         int physical_begin, int physical_end, WorkspaceArena& workspace,
                         Tensor& result, cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        fp8_attention_shard_for<CausalD256H24Kv4>(q, k, v, positions, valid, rows, scale, cache,
                                                  envelope, physical_begin, physical_end, workspace,
                                                  result, stream);
        return;
    }
    fp8_attention_shard_for<CausalD256H16Kv2>(q, k, v, positions, valid, rows, scale, cache,
                                              envelope, physical_begin, physical_end, workspace,
                                              result, stream);
}

std::size_t
fp8_attention_shard_workspace_capacity_bytes(std::int32_t q_heads, KvCacheStorage cache_storage,
                                             std::int32_t batch_size, std::int32_t max_width,
                                             CausalAttentionExecutionEnvelope envelope) {
    if (q_heads != CausalD256H24Kv4::QHeads && q_heads != CausalD256H16Kv2::QHeads) {
        throw std::invalid_argument("fp8_attention_shard workspace: unsupported head geometry");
    }
    if (batch_size < 1 || max_width < 1) {
        throw std::invalid_argument("fp8_attention_shard workspace: invalid call extent");
    }
    // A call forms its chunks left to right at the tile width, so every chunk is either the full
    // tile or a shorter final remainder, and any width below the tile can be that remainder. Each
    // chunk allocates under its own scope, so the call's requirement is the largest single-chunk
    // peak and not the sum over chunks. Sweeping the whole reachable width interval is therefore
    // exact and, at eight iterations, cheap enough to evaluate once at startup.
    const std::int32_t tile = q_heads == CausalD256H24Kv4::QHeads ? 8 : 6;
    std::size_t capacity    = 0;
    for (std::int32_t width = 1; width <= std::min(max_width, tile); ++width) {
        const std::int32_t splits =
            causal_attention_split_capacity(q_heads, width, cache_storage, envelope, batch_size);
        WorkspaceLayoutBuilder layout;
        (void)allocate_fp8_shard_chunk(layout, q_heads, width, splits, batch_size);
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
    return capacity;
}

void fp8_attention_merge(const Tensor& first, const Tensor& second, Tensor& out,
                         cudaStream_t stream) {
    if (out.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_fp8_merge<CausalD256H24Kv4>(first, second, out, stream);
        return;
    }
    launch_fp8_merge<CausalD256H16Kv2>(first, second, out, stream);
}

} // namespace ninfer::ops::detail
