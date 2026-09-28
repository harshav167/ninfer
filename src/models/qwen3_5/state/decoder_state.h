#pragma once

#include "core/layout.h"
#include "core/active_kv_stager.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/distributed_attention.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <optional>

namespace ninfer::models::qwen3_5 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    std::uint32_t text_resident_page_groups = 0;
    std::optional<int> text_overflow_device;
    std::uint32_t active_step_capacity = 0;
    std::optional<int> remote_attention_device;
    std::uint32_t remote_full_attention_layers = 0;
    std::int32_t attention_query_heads         = 0;
    std::int32_t max_attention_width           = 0;
    std::int32_t max_attention_batch           = 0;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    PagedKVStorageLayout layer_storage;
    ActiveKVStagingLayout active_staging;
    std::uint32_t active_step_capacity = 0;
    std::int32_t attention_query_heads = 0;
    std::int32_t max_attention_width   = 0;
    std::int32_t max_attention_batch   = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept { return pages.payload_bytes(); }
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
};

class PagedKVCache {
public:
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept { return pages_; }

    [[nodiscard]] const DeviceKVPagePool& page_pool() const noexcept { return pages_; }

    [[nodiscard]] KVExecutionTablePool& execution_tables() noexcept { return execution_tables_; }

    [[nodiscard]] const KVExecutionTablePool& execution_tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;
    [[nodiscard]] PagedKVBatchLayerView prepare_layer(std::uint32_t layer, const Tensor& positions,
                                                      const Tensor& valid_columns,
                                                      const Tensor& table_rows,
                                                      cudaStream_t stream) const;
    void complete_layer(std::uint32_t layer, cudaStream_t stream) const;
    [[nodiscard]] bool uses_remote_attention(std::uint32_t layer) const noexcept;
    [[nodiscard]] PagedKVStagedPlanes secondary_layer_planes(std::uint32_t layer) const;

    [[nodiscard]] ops::DistributedAttention* remote_attention() noexcept {
        return remote_attention_.get();
    }

    [[nodiscard]] ops::DistributedAttention* remote_attention() const noexcept {
        return remote_attention_.get();
    }

    [[nodiscard]] std::uint64_t active_read_bytes() const noexcept {
        return active_stager_ ? active_stager_->read_bytes() : 0;
    }

    [[nodiscard]] std::uint64_t active_write_bytes() const noexcept {
        return active_stager_ ? active_stager_->write_bytes() : 0;
    }

    [[nodiscard]] std::size_t active_staging_bytes() const noexcept {
        return active_staging_bytes_;
    }

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    // Residency may change without changing any logical KV contents or owner.
    mutable DeviceKVPagePool pages_;
    mutable KVExecutionTablePool execution_tables_;
    std::unique_ptr<ops::DistributedAttention> remote_attention_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    PagedKVStorageLayout layer_storage_;
    std::size_t active_staging_bytes_   = 0;
    std::uint32_t active_step_capacity_ = 0;
    mutable std::optional<PinnedHostBuffer> active_metadata_;
    mutable std::vector<std::uint8_t> active_marks_;
    mutable std::vector<std::uint32_t> active_reads_;
    mutable std::vector<std::uint32_t> active_writes_;
    mutable std::uint32_t active_next_layer_ = 0;
    mutable bool active_pipeline_            = false;
    std::unique_ptr<ActiveKVStager> active_stager_;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::models::qwen3_5
