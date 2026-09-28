#pragma once

// Row-scaled E4M3FN-cache causal prompt kernel. Q and cached K use the same fixed D256 rotation;
// SM89+ uses native E4M3 K32 Tensor Cores, while SM86 consumes the represented FP8 values through
// FP16 Tensor Cores. V codes widen to FP16, receive their represented row scale once, and feed
// FP16/FP32 PV MMA.

#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
#include "ops/kernel/tiered_kv_pointer.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptFp8Warps         = 16;
inline constexpr int kCausalPromptFp8Threads       = kCausalPromptFp8Warps * 32;
inline constexpr int kCausalPromptFp8Br            = 64;
inline constexpr int kCausalPromptFp8Bc            = 64;
inline constexpr int kCausalPromptFp8DB16          = kCausalPromptHeadDim / 2;
inline constexpr int kCausalPromptFp8RowTiles      = kCausalPromptFp8Br / 16;
inline constexpr int kCausalPromptFp8ProducerWarps = 2 * kCausalPromptFp8RowTiles;
inline constexpr int kCausalPromptFp8DConsumers = kCausalPromptFp8Warps / kCausalPromptFp8RowTiles;

inline constexpr int kCausalPromptFp8QBytes = kCausalPromptFp8Br * kCausalPromptHeadDim;
inline constexpr int kCausalPromptFp8QScaleBytes =
    kCausalPromptFp8Br * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptFp8KBytes = kCausalPromptFp8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptFp8VBytes = kCausalPromptFp8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptFp8VStageBytes =
    kCausalPromptFp8Bc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptFp8PBytes =
    kCausalPromptFp8Br * kCausalPromptFp8Bc * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptFp8ScaleBytes =
    2 * kCausalPromptFp8Bc * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptFp8StatsBytes =
    7 * kCausalPromptFp8Br * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptFp8SmemBytes =
    kCausalPromptFp8QBytes + kCausalPromptFp8QScaleBytes + kCausalPromptFp8KBytes +
    kCausalPromptFp8VBytes + kCausalPromptFp8VStageBytes + kCausalPromptFp8PBytes +
    kCausalPromptFp8ScaleBytes + kCausalPromptFp8StatsBytes;

static_assert(kCausalPromptFp8DConsumers == 4);
static_assert(kCausalPromptFp8SmemBytes == 92416);
// SM86 has FP16 Tensor Cores but no native FP8 MMA. A 64-row prompt tile reuses each KV tile
// across twice as many queries; Ampere overlays QK and PV staging to stay under its CTA limit.
inline constexpr int kCausalPromptFp8AmpereBr      = 64;
inline constexpr int kCausalPromptFp8AmpereWarps   = 16;
inline constexpr int kCausalPromptFp8AmpereThreads = kCausalPromptFp8AmpereWarps * 32;
inline constexpr int kCausalPromptFp8AmpereSmemBytes =
    kCausalPromptFp8AmpereBr * kCausalPromptHeadDim * static_cast<int>(sizeof(__half)) +
    kCausalPromptFp8AmpereBr * static_cast<int>(sizeof(float)) +
    kCausalPromptFp8Bc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half)) +
    kCausalPromptFp8AmpereBr * kCausalPromptFp8Bc * static_cast<int>(sizeof(__half)) +
    kCausalPromptFp8ScaleBytes + 7 * kCausalPromptFp8AmpereBr * static_cast<int>(sizeof(float));

static_assert(kCausalPromptFp8AmpereSmemBytes == 76032);

__device__ __forceinline__ int4 causal_prompt_fp8_dequant_f16x8(const std::uint8_t* codes8,
                                                                __half scale) {
    const int2 raw         = load_vec<int2>(codes8);
    const std::uint16_t* c = reinterpret_cast<const std::uint16_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 value2 = kv_cache_fp8_dequant_code2_to_half2(c[i], scale);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

template <typename Geometry>
__device__ __forceinline__ void causal_prompt_fp8_write_neutral(float* result, int q_head,
                                                                int row_begin, int row_end, int tid,
                                                                int threads) {
    constexpr int D = kCausalPromptHeadDim;
    for (int row = row_begin; row < row_end; ++row) {
        const std::int64_t component = static_cast<std::int64_t>(D + 1) *
                                       (q_head + static_cast<std::int64_t>(Geometry::QHeads) * row);
        for (int d = tid; d < D; d += threads) result[component + d] = 0.0F;
        if (tid == 0) result[component + D] = -CUDART_INF_F;
    }
}

template <typename Geometry, typename Metadata, bool Tiered, bool Compact = false>
__global__ __maxnreg__(120) void causal_attention_prompt_fp8_kernel(
    const __nv_bfloat16* __restrict__ q, KVPlanePointer<const std::uint8_t, Tiered> cache_k,
    KVPlanePointer<const std::uint8_t, Tiered> cache_v,
    KVPlanePointer<const __half, Tiered> cache_k_scale,
    KVPlanePointer<const __half, Tiered> cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    float* __restrict__ compact_out, std::int32_t width, int physical_begin, int physical_end) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    constexpr bool Ampere = true;
#else
    constexpr bool Ampere = false;
#endif
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = Ampere ? kCausalPromptFp8AmpereBr : kCausalPromptFp8Br;
    constexpr int Bc            = kCausalPromptFp8Bc;
    constexpr int DB16          = D / 2;
    constexpr int Warps         = Ampere ? kCausalPromptFp8AmpereWarps : kCausalPromptFp8Warps;
    constexpr int Threads       = Warps * 32;
    constexpr int RowTiles      = Br / 16;
    constexpr int DConsumers    = Warps / RowTiles;
    constexpr int QKKs          = Ampere ? D / 16 : D / 32;
    constexpr int QKNt          = (Bc / 2) / 8;
    constexpr int PVNtPerWarp   = D / (DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = 2 * RowTiles;
    constexpr int VWorkerWarps  = Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr int QBytes        = Ampere ? Br * D * static_cast<int>(sizeof(__half)) : Br * D;
    constexpr int QScaleBytes   = Br * static_cast<int>(sizeof(float));
    constexpr int KBytes        = Ampere ? Bc * D * static_cast<int>(sizeof(__half)) : Bc * D;
    constexpr int VBytes        = Ampere ? 0 : Bc * D;
    constexpr int VStageBytes   = Ampere ? 0 : Bc * D * static_cast<int>(sizeof(__half));
    constexpr int PBytes        = Br * Bc * static_cast<int>(sizeof(__half));
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffU;
    static_assert(Ampere ? QKKs == 16 : QKKs == 8);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::uint8_t* q_fp8 = reinterpret_cast<std::uint8_t*>(smem_raw);
    float* q_scale      = reinterpret_cast<float*>(q_fp8 + QBytes);
    std::uint8_t* k_fp8 =
        reinterpret_cast<std::uint8_t*>(reinterpret_cast<unsigned char*>(q_scale) + QScaleBytes);
    std::uint8_t* v_fp8 = k_fp8 + KBytes;
    // Ampere reuses the QK K-half tile for PV V-half after a full-block barrier.
    __half* v_f16 =
        Ampere ? reinterpret_cast<__half*>(k_fp8) : reinterpret_cast<__half*>(v_fp8 + VBytes);
    __half* p_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_fp8) + VBytes + VStageBytes);
    __half* k_scale_s  = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + PBytes);
    __half* v_scale_s  = k_scale_s + Bc;
    float* running_m_s = reinterpret_cast<float*>(v_scale_s + Bc);
    float* running_l_s = running_m_s + Br;
    float* partial_m_s = running_l_s + Br;
    float* partial_l_s = partial_m_s + 2 * Br;
    float* alpha_s     = partial_l_s + 2 * Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_fp8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_fp8);
    __half* q_half       = reinterpret_cast<__half*>(q_fp8);
    __half* k_half       = reinterpret_cast<__half*>(k_fp8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) return;
    if (q0 >= tokens) {
        if constexpr (Compact) {
            causal_prompt_fp8_write_neutral<Geometry>(compact_out, q_head, q0, min(q0 + Br, width),
                                                      tid, Threads);
        } else {
            causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                     Threads);
        }
        return;
    }
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows             = min(Br, tokens - q0);
    const int max_query_abs         = positions[q0 + tile_rows - 1];
    const int key_blocks            = max_query_abs / Bc + 1;
    if constexpr (Compact) {
        bool owns_visible_page = false;
        for (int page = 0; page <= (max_query_abs >> kPagedKVPageShift); ++page) {
            const int physical_page = block_table[page];
            owns_visible_page |= physical_page >= physical_begin && physical_page < physical_end;
        }
        if (!owns_visible_page) {
            causal_prompt_fp8_write_neutral<Geometry>(compact_out, q_head, q0, min(q0 + Br, width),
                                                      tid, Threads);
            return;
        }
    }

    for (int row = warp; row < Br; row += Warps) {
        float values[8];
        float local_absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r] =
                row < tile_rows
                    ? __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)])
                    : 0.0F;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) local_absmax = fmaxf(local_absmax, fabsf(values[r]));
        const float absmax = warp_max(local_absmax, FullMask);
        const float qs     = absmax > 0.0F ? absmax / kKVCacheFp8MaxFinite : 0.0F;
        const float inv    = qs > 0.0F ? 1.0F / qs : 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d     = lane + 32 * r;
            const auto code = kv_cache_fp8_quant_code(values[r], inv);
            if constexpr (Ampere) {
                __nv_fp8_e4m3 represented;
                represented.__x                             = code;
                q_half[row * D + causal_prompt_swz(row, d)] = static_cast<__half>(represented);
            } else {
                causal_prompt_store_byte_swizzled(q_fp8, row, d, code);
            }
        }
        if (lane == 0) q_scale[row] = qs;
    }
    if (tid < Br) {
        running_m_s[tid] = -CUDART_INF_F;
        running_l_s[tid] = 0.0F;
    }
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0 = 0.0F;
    float q_scale_r1 = 0.0F;
    if (warp < ProducerWarps) {
        const int row0 = (warp >> 1) * 16 + gid;
        const int row1 = row0 + 8;
        q_scale_r0     = __shfl_sync(FullMask, lid == 0 ? q_scale[row0] : 0.0F, gid * 4);
        q_scale_r1     = __shfl_sync(FullMask, lid == 0 ? q_scale[row1] : 0.0F, gid * 4);
    }

    auto issue_kv_scales = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        const int global_page   = block_table[tile_k0 >> kPagedKVPageShift];
        const bool page_owned   = global_page >= physical_begin && global_page < physical_end;
        const int physical_page = page_owned ? global_page - physical_begin : 0;
        const int page_offset0  = tile_k0 & (kPagedKVPageSize - 1);
        for (int key_l = cooperative_tid; key_l < Bc; key_l += cooperative_threads) {
            const int key = tile_k0 + key_l;
            if (page_owned && key <= max_query_abs) {
                const std::int64_t off = kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head,
                                                                            page_offset0 + key_l);
                k_scale_s[key_l]       = cache_k_scale[off];
                v_scale_s[key_l]       = cache_v_scale[off];
            } else {
                k_scale_s[key_l] = __float2half_rn(0.0F);
                v_scale_s[key_l] = __float2half_rn(0.0F);
            }
        }
    };

    auto issue_kv_codes = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        const int global_page   = block_table[tile_k0 >> kPagedKVPageShift];
        const bool page_owned   = global_page >= physical_begin && global_page < physical_end;
        const int physical_page = page_owned ? global_page - physical_begin : 0;
        const int page_offset0  = tile_k0 & (kPagedKVPageSize - 1);
        if constexpr (Ampere) {
#pragma unroll 1
            for (int chunk = cooperative_tid; chunk < Bc * (D / 8); chunk += cooperative_threads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = tile_k0 + key_l;
                __half* kd      = &k_half[key_l * D + causal_prompt_swz(key_l, d)];
                if (page_owned && key <= max_query_abs) {
                    const std::int64_t off = kv_cache_fp8_code_index<Geometry>(
                        physical_page, kv_head, d, page_offset0 + key_l);
                    store_vec(
                        kd, causal_prompt_fp8_dequant_f16x8(&cache_k[off], __float2half_rn(1.0F)));
                } else {
                    store_vec(kd, make_int4(0, 0, 0, 0));
                }
            }
        } else {
#pragma unroll 1
            for (int chunk = cooperative_tid; chunk < Bc * (D / 16); chunk += cooperative_threads) {
                const int key_l  = chunk / (D / 16);
                const int dc     = chunk - key_l * (D / 16);
                const int d      = dc * 16;
                const int key    = tile_k0 + key_l;
                std::uint8_t* kd = &k_fp8[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2];
                std::uint8_t* vd = &v_fp8[key_l * D + d];
                if (page_owned && key <= max_query_abs) {
                    const std::int64_t off = kv_cache_fp8_code_index<Geometry>(
                        physical_page, kv_head, d, page_offset0 + key_l);
                    cp_async<16, Cache::cg>(kd, &cache_k[off]);
                    cp_async<16, Cache::cg>(vd, &cache_v[off]);
                } else {
                    store_vec(kd, make_int4(0, 0, 0, 0));
                    store_vec(vd, make_int4(0, 0, 0, 0));
                }
            }
            ninfer::ops::cp_commit();
        }
    };

    auto issue_kv_tile = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        issue_kv_scales(tile_k0, cooperative_tid, cooperative_threads);
        issue_kv_codes(tile_k0, cooperative_tid, cooperative_threads);
    };

    issue_kv_tile(0, tid, Threads);
    if constexpr (!Ampere) ninfer::ops::cp_wait<0>();
    __syncthreads();

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
    }
    const float scale_l2 = scale * Log2E;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0                 = kb * Bc;
        const int tile_physical_page = block_table[k0 >> kPagedKVPageShift];
        const bool tile_owned =
            tile_physical_page >= physical_begin && tile_physical_page < physical_end;
        if (warp < ProducerWarps) {
            const int row_base = (warp >> 1) * 16;
            const int col_half = warp & 1;
            const int col_base = col_half * (Bc / 2);
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt)
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;

            for (int kk = 0; kk < QKKs; ++kk) {
                const int acol = kk * 16 + a_coloff;
                unsigned af[4];
                if constexpr (Ampere) {
                    ldmatrix_x4(af[0], af[1], af[2], af[3],
                                smem_addr(&q_half[(row_base + a_rowoff) * D +
                                                  causal_prompt_swz(row_base + a_rowoff, acol)]));
                } else {
                    ldmatrix_x4(af[0], af[1], af[2], af[3],
                                smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                                 causal_prompt_swz(row_base + a_rowoff, acol)]));
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int brow = col_base + nt * 8 + b_rin;
                    const int bcol = kk * 16 + b_koff;
                    unsigned bf[2];
                    if constexpr (Ampere) {
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_half[brow * D + causal_prompt_swz(brow, bcol)]));
                        mma_f16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                                af[1], af[2], af[3], bf[0], bf[1]);
                    } else {
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                        mma_fp8_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                                     af[1], af[2], af[3], bf[0], bf[1]);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int keya = col_base + nt * 8 + 2 * lid;
                const int keyb = keya + 1;
                float ks0      = gid == 0 ? __half2float(k_scale_s[keya]) : 0.0F;
                float ks1      = gid == 0 ? __half2float(k_scale_s[keyb]) : 0.0F;
                ks0            = __shfl_sync(FullMask, ks0, lid);
                ks1            = __shfl_sync(FullMask, ks1, lid);
                score[nt][0] *= q_scale_r0 * ks0;
                score[nt][1] *= q_scale_r0 * ks1;
                score[nt][2] *= q_scale_r1 * ks0;
                score[nt][3] *= q_scale_r1 * ks1;
            }

            const int row0  = row_base + gid;
            const int row1  = row0 + 8;
            const int qabs0 = row0 < tile_rows ? positions[q0 + row0] : -1;
            const int qabs1 = row1 < tile_rows ? positions[q0 + row1] : -1;
            const bool full_score_tile =
                tile_owned && q0 + Br <= tokens && k0 + Bc - 1 <= positions[q0];
            float bm0 = -CUDART_INF_F;
            float bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + col_base + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = tile_owned && key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = tile_owned && key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = tile_owned && key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = tile_owned && key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            if (lid == 0) {
                partial_m_s[col_half * Br + row0] = bm0;
                partial_m_s[col_half * Br + row1] = bm1;
            }
            asm volatile("bar.sync 1, 256;" ::: "memory");
            bm0                     = fmaxf(partial_m_s[row0], partial_m_s[Br + row0]);
            bm1                     = fmaxf(partial_m_s[row1], partial_m_s[Br + row1]);
            const float previous_m0 = running_m_s[row0];
            const float previous_m1 = running_m_s[row1];
            const float nm0         = fmaxf(previous_m0, bm0);
            const float nm1         = fmaxf(previous_m1, bm1);
            const float nm0_scaled  = nm0 * scale_l2;
            const float nm1_scaled  = nm1 * scale_l2;
            const float alpha0 = previous_m0 == -CUDART_INF_F
                                     ? 0.0F
                                     : exp2_approx(__fmaf_rn(previous_m0, scale_l2, -nm0_scaled));
            const float alpha1 = previous_m1 == -CUDART_INF_F
                                     ? 0.0F
                                     : exp2_approx(__fmaf_rn(previous_m1, scale_l2, -nm1_scaled));
            float bl0          = 0.0F;
            float bl1          = 0.0F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = col_base + nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0F;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0F;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0F;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0F;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col0)] = __float2half_rn(p00);
                p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col1)] = __float2half_rn(p01);
                p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col0)] = __float2half_rn(p10);
                p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col1)] = __float2half_rn(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);
            if (lid == 0) {
                partial_l_s[col_half * Br + row0] = bl0;
                partial_l_s[col_half * Br + row1] = bl1;
            }
            asm volatile("bar.sync 1, 256;" ::: "memory");
            if (col_half == 0 && lid == 0) {
                const float tile_l0 = partial_l_s[row0] + partial_l_s[Br + row0];
                const float tile_l1 = partial_l_s[row1] + partial_l_s[Br + row1];
                running_l_s[row0]   = __fmaf_rn(running_l_s[row0], alpha0, tile_l0);
                running_l_s[row1]   = __fmaf_rn(running_l_s[row1], alpha1, tile_l1);
                running_m_s[row0]   = nm0;
                running_m_s[row1]   = nm1;
                alpha_s[row0]       = alpha0;
                alpha_s[row1]       = alpha1;
            }
        } else if (!Ampere && warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
                if (key <= max_query_abs) {
                    store_vec(dst, causal_prompt_fp8_dequant_f16x8(&v_fp8[key_l * D + d],
                                                                   v_scale_s[key_l]));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();
        // Every producer has finished reading K; Ampere can now overwrite that shared tile with V.
        if constexpr (Ampere) {
            const int physical_page = tile_owned ? tile_physical_page - physical_begin : 0;
            const int page_offset0  = k0 & (kPagedKVPageSize - 1);
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
                if (tile_owned && key <= max_query_abs) {
                    const std::int64_t off = kv_cache_fp8_code_index<Geometry>(
                        physical_page, kv_head, d, page_offset0 + key_l);
                    store_vec(dst,
                              causal_prompt_fp8_dequant_f16x8(&cache_v[off], v_scale_s[key_l]));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
            __syncthreads();
        }

        const bool has_next = kb + 1 < key_blocks;
        if constexpr (!Ampere) {
            if (has_next) issue_kv_tile((kb + 1) * Bc, tid, Threads);
        }

        const int row_tile = warp % RowTiles;
        const int d_slice  = warp / RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            if constexpr (Compact) {
                acc[n][0] = alpha0 == 0.0F ? 0.0F : acc[n][0] * alpha0;
                acc[n][1] = alpha0 == 0.0F ? 0.0F : acc[n][1] * alpha0;
                acc[n][2] = alpha1 == 0.0F ? 0.0F : acc[n][2] * alpha1;
                acc[n][3] = alpha1 == 0.0F ? 0.0F : acc[n][3] * alpha1;
            } else {
                acc[n][0] *= alpha0;
                acc[n][1] *= alpha0;
                acc[n][2] *= alpha1;
                acc[n][3] *= alpha1;
            }
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       causal_prompt_p_swz<Bc>(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
        }
        if constexpr (!Ampere) {
            if (has_next) ninfer::ops::cp_wait<0>();
            __syncthreads();
        } else {
            __syncthreads();
            if (has_next) {
                issue_kv_tile((kb + 1) * Bc, tid, Threads);
                __syncthreads();
            }
        }
    }

    const int row_tile = warp % RowTiles;
    const int d_slice  = warp / RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = running_l_s[row0] > 0.0F ? __frcp_rn(running_l_s[row0]) : 0.0F;
    const float inv_l1 = running_l_s[row1] > 0.0F ? __frcp_rn(running_l_s[row1]) : 0.0F;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if constexpr (Compact) {
            if (row0 < tile_rows) {
                const std::int64_t component =
                    static_cast<std::int64_t>(D + 1) *
                    (q_head + static_cast<std::int64_t>(Geometry::QHeads) * (q0 + row0));
                const bool has_mass             = running_l_s[row0] > 0.0F;
                compact_out[component + d0]     = has_mass ? acc[n][0] * inv_l0 : 0.0F;
                compact_out[component + d0 + 1] = has_mass ? acc[n][1] * inv_l0 : 0.0F;
                if (d_slice == 0 && lid == 0 && n == 0) {
                    compact_out[component + D] =
                        running_l_s[row0] > 0.0F
                            ? running_m_s[row0] * scale + logf(running_l_s[row0])
                            : -CUDART_INF_F;
                }
            }
            if (row1 < tile_rows) {
                const std::int64_t component =
                    static_cast<std::int64_t>(D + 1) *
                    (q_head + static_cast<std::int64_t>(Geometry::QHeads) * (q0 + row1));
                const bool has_mass             = running_l_s[row1] > 0.0F;
                compact_out[component + d0]     = has_mass ? acc[n][2] * inv_l1 : 0.0F;
                compact_out[component + d0 + 1] = has_mass ? acc[n][3] * inv_l1 : 0.0F;
                if (d_slice == 0 && lid == 0 && n == 0) {
                    compact_out[component + D] =
                        running_l_s[row1] > 0.0F
                            ? running_m_s[row1] * scale + logf(running_l_s[row1])
                            : -CUDART_INF_F;
                }
            }
        } else {
            if (row0 < tile_rows) {
                *reinterpret_cast<unsigned*>(
                    &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                    pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
            }
            if (row1 < tile_rows) {
                *reinterpret_cast<unsigned*>(
                    &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                    pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
            }
        }
    }
    if constexpr (Compact) {
        causal_prompt_fp8_write_neutral<Geometry>(compact_out, q_head, tokens, min(q0 + Br, width),
                                                  tid, Threads);
    } else {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                                 Threads);
    }
}

} // namespace ninfer::ops
