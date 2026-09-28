#include "models/qwen3_5/state/decoder_state.h"

#include "core/device.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, KvCacheStorage storage,
                              std::int32_t table_rows, std::uint32_t physical_page_groups,
                              std::uint32_t resident_page_groups         = 0,
                              std::optional<int> overflow_device         = std::nullopt,
                              std::uint32_t active_step_capacity         = 0,
                              std::optional<int> static_secondary_device = std::nullopt,
                              std::vector<std::uint32_t> static_secondary_plane_indices = {},
                              std::int32_t attention_query_heads                        = 0,
                              std::int32_t max_attention_width                          = 0,
                              std::int32_t max_attention_batch                          = 0) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const PagedKVStorageLayout layer_storage = paged_kv_storage_layout(storage, head_dim);

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    KVPageGeometry geometry;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * layer_storage.planes_per_layer());
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back(
            {layer_storage.key.data_dtype, layer_storage.key.data_leading_extent, kv_heads, 256});
        geometry.planes.push_back({layer_storage.value.data_dtype,
                                   layer_storage.value.data_leading_extent, kv_heads, 256});
        if (layer_storage.key.has_scale()) {
            geometry.planes.push_back({layer_storage.key.scale_dtype,
                                       layer_storage.key.scale_leading_extent, kv_heads, 256});
        }
        if (layer_storage.value.has_scale()) {
            geometry.planes.push_back({layer_storage.value.scale_dtype,
                                       layer_storage.value.scale_leading_extent, kv_heads, 256});
        }
    }
    PagedKVCacheLayout out{
        .pages = plan_device_kv_page_pool(
            builder, DeviceKVPagePoolSpec{.page_group_count          = physical_page_groups,
                                          .geometry                  = std::move(geometry),
                                          .resident_page_group_count = resident_page_groups,
                                          .overflow_device           = overflow_device,
                                          .static_secondary_device   = static_secondary_device,
                                          .static_secondary_plane_indices =
                                              std::move(static_secondary_plane_indices)}),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers                = layers,
        .max_context           = capacity,
        .kv_heads              = kv_heads,
        .layer_storage         = layer_storage,
        .active_step_capacity  = active_step_capacity,
        .attention_query_heads = attention_query_heads,
        .max_attention_width   = max_attention_width,
        .max_attention_batch   = max_attention_batch,
    };
    if (!static_secondary_device) {
        out.active_staging =
            plan_active_kv_staging(builder, out.pages, layer_storage.planes_per_layer());
    }
    return out;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    std::optional<int> remote_device = spec.remote_attention_device;
    std::vector<std::uint32_t> remote_planes;
    if (spec.remote_full_attention_layers != 0) {
        if (!remote_device || spec.remote_full_attention_layers > spec.full_attention_layers) {
            throw std::invalid_argument("remote full-attention layer selection is invalid");
        }
        if (spec.text_overflow_device) {
            throw std::invalid_argument(
                "static remote attention cannot combine with active KV overflow");
        }
        if (spec.kv_storage != KvCacheStorage::Fp8E4M3Row256 || spec.attention_head_dim != 256 ||
            !((spec.attention_query_heads == 24 && spec.kv_heads == 4) ||
              (spec.attention_query_heads == 16 && spec.kv_heads == 2)) ||
            spec.max_attention_width <= 0 ||
            spec.max_attention_width >
                static_cast<std::int32_t>(ops::kCausalAttentionMaximumVisibleKeys) ||
            spec.max_attention_batch <= 0 || spec.max_attention_batch > 8 ||
            spec.kv_table_rows < spec.max_attention_batch) {
            throw std::invalid_argument("remote attention geometry is unsupported");
        }
        if (spec.capacity > ops::kCausalAttentionMaximumVisibleKeys) {
            throw std::invalid_argument("remote attention context exceeds the supported envelope");
        }
        const std::uint32_t first = spec.full_attention_layers - spec.remote_full_attention_layers;
        const std::size_t planes_per_layer =
            paged_kv_storage_layout(spec.kv_storage, spec.attention_head_dim).planes_per_layer();
        remote_planes.reserve(static_cast<std::size_t>(spec.remote_full_attention_layers) *
                              planes_per_layer);
        for (std::uint32_t layer = first; layer < spec.full_attention_layers; ++layer) {
            const std::uint32_t base =
                static_cast<std::uint32_t>(static_cast<std::size_t>(layer) * planes_per_layer);
            for (std::size_t plane = 0; plane < planes_per_layer; ++plane) {
                remote_planes.push_back(base + static_cast<std::uint32_t>(plane));
            }
        }
    } else {
        remote_device.reset();
    }

    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                spec.text_physical_page_groups, spec.text_resident_page_groups,
                                spec.text_overflow_device, spec.active_step_capacity, remote_device,
                                std::move(remote_planes), spec.attention_query_heads,
                                spec.max_attention_width, spec.max_attention_batch);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      layer_storage_(layout.layer_storage) {
    if (pages_.plane_count() !=
        static_cast<std::size_t>(layers_) * layer_storage_.planes_per_layer()) {
        throw std::invalid_argument("Paged KV layer plane inventory is inconsistent");
    }
    if (pages_.has_static_secondary_planes()) {
        if (pages_.overflow_buffer() != nullptr || layout.attention_query_heads <= 0 ||
            layout.max_attention_width <= 0 || layout.max_attention_batch <= 0) {
            throw std::invalid_argument("static remote attention resources are inconsistent");
        }
        int primary_device = 0;
        CUDA_CHECK(cudaGetDevice(&primary_device));
        remote_attention_ = std::make_unique<ops::DistributedAttention>(
            primary_device, pages_.static_secondary_device(),
            ops::AttentionHeadGeometry{layer_storage_.head_dim, layout.attention_query_heads,
                                       kv_heads_},
            layout.max_attention_width, layout.max_attention_batch,
            layout.execution_tables.spec.table_rows,
            static_cast<std::int32_t>((max_context_ + kPagedKVPageSize - 1U) / kPagedKVPageSize));
    }
    if (pages_.overflow_buffer()) {
        if (layout.active_step_capacity == 0) {
            throw std::invalid_argument("active KV metadata capacity is zero");
        }
        active_step_capacity_ = layout.active_step_capacity;
        active_staging_bytes_ = layout.active_staging.bytes;
        active_metadata_.emplace((static_cast<std::size_t>(active_step_capacity_) + 16U) *
                                 sizeof(std::int32_t));
        const auto overflow_pages = pages_.capacity_pages() - pages_.resident_capacity_pages();
        active_marks_.resize(overflow_pages);
        active_reads_.reserve(overflow_pages);
        active_writes_.reserve(overflow_pages);
        active_stager_ = std::make_unique<ActiveKVStager>(backing, layout.active_staging, pages_);
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const std::size_t stride        = layer_storage_.planes_per_layer();
    const std::size_t base          = static_cast<std::size_t>(layer) * stride;
    const std::size_t k_scale_index = base + 2;
    const std::size_t v_scale_index =
        k_scale_index + static_cast<std::size_t>(layer_storage_.key.has_scale());
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = layer_storage_.key.has_scale() ? pages_.plane(k_scale_index) : Tensor(),
        .v_scale_pages = layer_storage_.value.has_scale() ? pages_.plane(v_scale_index) : Tensor(),
        .block_table   = block_table,
        .head_dim      = layer_storage_.head_dim,
        .num_kv_heads  = kv_heads_,
        .storage       = layer_storage_.storage,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    const PagedKVLayerView direct = layer_view(layer, Tensor());
    return PagedKVBatchLayerView{
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
        .staged        = direct.staged,
    };
}

bool PagedKVCache::uses_remote_attention(std::uint32_t layer) const noexcept {
    if (layer >= layers_ || !pages_.has_static_secondary_planes()) { return false; }
    const std::size_t base = static_cast<std::size_t>(layer) * layer_storage_.planes_per_layer();
    return pages_.plane(base).data == nullptr;
}

PagedKVStagedPlanes PagedKVCache::secondary_layer_planes(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    if (!uses_remote_attention(layer)) {
        throw std::invalid_argument("Paged KV layer is not on the static secondary device");
    }
    const std::size_t stride        = layer_storage_.planes_per_layer();
    const std::size_t base          = static_cast<std::size_t>(layer) * stride;
    const std::size_t k_scale_index = base + 2;
    const std::size_t v_scale_index =
        k_scale_index + static_cast<std::size_t>(layer_storage_.key.has_scale());
    return PagedKVStagedPlanes{
        .k = pages_.secondary_plane(base),
        .v = pages_.secondary_plane(base + 1),
        .k_scale =
            layer_storage_.key.has_scale() ? pages_.secondary_plane(k_scale_index) : Tensor(),
        .v_scale =
            layer_storage_.value.has_scale() ? pages_.secondary_plane(v_scale_index) : Tensor(),
    };
}

PagedKVBatchLayerView PagedKVCache::prepare_layer(std::uint32_t layer, const Tensor& positions,
                                                  const Tensor& valid_columns,
                                                  const Tensor& table_rows,
                                                  cudaStream_t stream) const {
    PagedKVBatchLayerView result = batch_layer_view(layer);
    if (!active_stager_) { return result; }
    if (layer != active_next_layer_) { throw std::logic_error("active KV layer order changed"); }
    if (layer == 0) {
        const auto width = positions.ne[0];
        const auto batch = positions.ne[1];
        if (positions.dtype != DType::I32 || !positions.data || !positions.is_contiguous() ||
            width <= 0 || batch <= 0 || batch > 8 || positions.ne[2] != 1 || positions.ne[3] != 1 ||
            positions.numel() > active_step_capacity_ || table_rows.dtype != DType::I32 ||
            !table_rows.data || !table_rows.is_contiguous() || table_rows.numel() < batch ||
            (valid_columns.data &&
             (valid_columns.dtype != DType::I32 || !valid_columns.is_contiguous() ||
              valid_columns.numel() < batch))) {
            throw std::invalid_argument("active KV staging metadata is invalid");
        }
        auto* host_positions = static_cast<std::int32_t*>(active_metadata_->data());
        auto* host_rows      = host_positions + active_step_capacity_;
        auto* host_valid     = host_rows + 8;
        CUDA_CHECK(cudaMemcpyAsync(host_positions, positions.data, positions.bytes(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_rows, table_rows.data, batch * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        if (valid_columns.data) {
            CUDA_CHECK(cudaMemcpyAsync(host_valid, valid_columns.data, batch * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, stream));
        } else {
            std::fill_n(host_valid, batch, width);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        execution_tables_.prioritize_active_residency(pages_, stream);
        std::fill(active_marks_.begin(), active_marks_.end(), 0);
        active_reads_.clear();
        active_writes_.clear();
        const auto resident = pages_.resident_capacity_pages();
        for (std::int32_t b = 0; b < batch; ++b) {
            if (host_valid[b] < 0 || host_valid[b] > width) {
                throw std::invalid_argument("active KV valid columns exceed width");
            }
            if (host_valid[b] == 0) { continue; }
            const auto table     = execution_tables_.host_row(host_rows[b]);
            std::int32_t maximum = -1;
            const auto mark      = [&](std::uint32_t logical, std::uint8_t bits) {
                if (logical >= table.size() || table[logical] < 0 ||
                    static_cast<std::uint32_t>(table[logical]) >= pages_.capacity_pages()) {
                    throw std::logic_error("active KV execution mapping is invalid");
                }
                const auto physical = static_cast<std::uint32_t>(table[logical]);
                if (physical >= resident) { active_marks_[physical - resident] |= bits; }
            };
            for (std::int32_t t = 0; t < host_valid[b]; ++t) {
                const auto position = host_positions[b * width + t];
                if (position < 0 || static_cast<std::uint32_t>(position) >= max_context_) {
                    throw std::invalid_argument("active KV position is outside the context");
                }
                maximum = std::max(maximum, position);
                mark(static_cast<std::uint32_t>(position) / kPagedKVPageSize, 3);
            }
            for (std::uint32_t p = 0; p <= static_cast<std::uint32_t>(maximum) / kPagedKVPageSize;
                 ++p) {
                mark(p, 1);
            }
        }
        for (std::uint32_t p = 0; p < active_marks_.size(); ++p) {
            if (active_marks_[p] & 1U) { active_reads_.push_back(resident + p); }
            if (active_marks_[p] & 2U) { active_writes_.push_back(resident + p); }
        }
        active_pipeline_ = !active_reads_.empty();
        if (active_pipeline_) { active_stager_->begin(active_reads_, active_writes_); }
    }
    if (active_pipeline_) {
        const auto staged = active_stager_->prepare(layer, stream);
        if (staged.size() != 4) {
            throw std::logic_error("quantized staged plane inventory is incomplete");
        }
        result.staged = {staged[0], staged[1], staged[2], staged[3]};
    }
    return result;
}

void PagedKVCache::complete_layer(std::uint32_t layer, cudaStream_t stream) const {
    if (!active_stager_) { return; }
    if (layer != active_next_layer_) {
        throw std::logic_error("active KV completion order changed");
    }
    if (active_pipeline_) { active_stager_->commit(layer, stream); }
    ++active_next_layer_;
    if (active_next_layer_ == layers_) {
        if (active_pipeline_) { active_stager_->finish(); }
        active_next_layer_ = 0;
        active_pipeline_   = false;
    }
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::models::qwen3_5
