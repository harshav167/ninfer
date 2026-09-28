#pragma once
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {
void fp8_attention_shard(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                         const Tensor& valid, const Tensor& rows, float scale,
                         PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                         int physical_begin, int physical_end, WorkspaceArena& workspace,
                         Tensor& result, cudaStream_t stream);
void fp8_attention_merge(const Tensor& first, const Tensor& second, Tensor& out,
                         cudaStream_t stream);

// Transient capacity one device's shard consumes over every chunk the launcher can form for a call
// of at most max_width query rows at one exact batch size. It is derived from the same allocation
// recipe the shard launcher runs, so an owner that sizes an arena from this query cannot be short
// when the launcher allocates mid-call.
[[nodiscard]] std::size_t
fp8_attention_shard_workspace_capacity_bytes(std::int32_t q_heads, KvCacheStorage cache_storage,
                                             std::int32_t batch_size, std::int32_t max_width,
                                             CausalAttentionExecutionEnvelope envelope);
} // namespace ninfer::ops::detail
