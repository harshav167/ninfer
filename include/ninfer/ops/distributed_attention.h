#pragma once

#include "ninfer/ops/softmax_attention.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::ops {

struct DistributedAttentionResourceEstimate {
    std::size_t primary_bytes      = 0;
    std::size_t secondary_bytes    = 0;
    std::size_t host_staging_bytes = 0;
};

// Explicit startup-owned resources for serialized calls through one two-device FP8 attention
// route. The object is not thread-safe or reentrant. Neither device owns model weights through
// this object; caller-owned cache planes remain external.
//
// The secondary route stages the query block and metadata through pinned host memory because
// peer access is not assumed. The secondary shard and its cache writes are settled before a
// successful return. Primary work and output remain ordered on `stream` and may still be in flight
// when execute returns. The caller must have primary input producers ordered before the call on
// `stream` or externally synchronized; the primary result follows ordinary stream readiness
// semantics.
// Stream capture is rejected. Once submitted, remote work has no cancellation API. On an
// exception, cleanup attempts to settle both streams; a failed settle poisons this instance.
// Device loss or failed synchronization can still leave hardware work unresolved.
class DistributedAttention {
public:
    // Exact allocation totals for a valid constructor configuration. This query performs no CUDA
    // calls and uses the same sizing recipe as construction.
    [[nodiscard]] static DistributedAttentionResourceEstimate
    estimate_resources(int primary, int secondary, AttentionHeadGeometry geometry,
                       std::int32_t max_width, std::int32_t max_batch, std::int32_t max_table_rows,
                       std::int32_t max_logical_pages);

    // Construction preallocates fixed capacities: max_width is 1..262144, max_batch is 1..8,
    // max_table_rows is at least max_batch, and max_logical_pages is 1..4096. Invalid constructor
    // capacities and execution beyond the declared width, batch, row, or page limits are rejected;
    // allocation failures propagate and capacities never grow lazily.
    DistributedAttention(int primary, int secondary, AttentionHeadGeometry geometry,
                         std::int32_t max_width, std::int32_t max_batch,
                         std::int32_t max_table_rows, std::int32_t max_logical_pages);
    ~DistributedAttention();
    DistributedAttention(const DistributedAttention&)            = delete;
    DistributedAttention& operator=(const DistributedAttention&) = delete;

    // The primary and secondary cache planes partition the physical page IDs:
    // [0, primary_pages) and [primary_pages, primary_pages + secondary_pages). Empty ownership
    // is represented by four null tensors; either shard may be empty and contributes the neutral
    // partial, but at least one group must own pages. The primary block table describes the
    // complete namespace.
    // All other input tensors and output live on primary. q/k/v/out are contiguous
    // BF16 [256,Hq|Hkv,W,B], positions and table rows are I32, and valid_columns is either I32
    // [B] or empty for dense rows. Supported geometries are [256,24,4] and [256,16,2] with
    // scale 1/16; B is 1..8, and B>1 requires W<=16. W must fit the constructor's max_width and
    // every visible prefix must fit `envelope`.
    //
    // Primary cache pages, secondary cache planes, and their scales use row-scaled FP8 E4M3FN
    // storage. Every visible key is resolved by the global block table and contributes once.
    // Cache pages, scales, inputs, output and the table must be pairwise non-overlapping and
    // reside on their declared device. Inputs are not mutated; addressed cache rows are appended
    // on their owning device.
    // Every row's live positions are sequential. Masked live columns form a prefix; masked tails
    // repeat the final live position (or use zero positions for an empty row) and produce exact
    // BF16 zero outputs.
    //
    // No historical K/V is transferred between devices. Each call moves only q/k/v, positions,
    // row/table metadata, and one compact FP32 partial via pinned host staging.
    //
    // If execution throws after secondary work was submitted, cleanup settles the caller and
    // secondary streams before rethrowing when possible. Failed settlement poisons this instance
    // so a later call cannot reuse staging or workspaces while their ownership is uncertain.
    void execute(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                 const Tensor& valid_columns, const Tensor& table_rows, float scale,
                 PagedKVBatchLayerView primary_cache, PagedKVStagedPlanes secondary_cache,
                 CausalAttentionExecutionEnvelope envelope, Tensor& out, cudaStream_t stream);

    // Counts include all object-owned CUDA allocations on each GPU (partials, workspaces, copied
    // inputs/metadata, and neutral stand-ins); caller-owned cache planes are excluded.
    // The pinned host staging allocation is reported separately; streams and events are not bytes.
    [[nodiscard]] std::size_t primary_bytes() const noexcept;
    [[nodiscard]] std::size_t secondary_bytes() const noexcept;
    [[nodiscard]] std::size_t host_staging_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::ops
