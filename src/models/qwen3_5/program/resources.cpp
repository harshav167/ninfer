#include "models/qwen3_5/program/resources.h"

#include "core/startup.h"
#include "models/load_options.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t kAlignment = 256;

std::int32_t dimension(std::uint64_t value) {
    if (value == 0 || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("Qwen3.5 resource dimension is outside the Tensor domain");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

ResourceLayout plan_resources(const ResourceSpec& spec) {
    if (spec.decoder.enable_mtp != (spec.frame.backend == SpeculativeBackend::Mtp) ||
        spec.dflash.has_value() != is_masked_draft_backend(spec.frame.backend) ||
        spec.decoder.kv_table_rows < static_cast<std::int32_t>(spec.frame.batch_capacity)) {
        throw std::invalid_argument("Qwen3.5 context and frame capacities do not agree");
    }
    LayoutBuilder builder;
    ResourceLayout out;
    out.decoder      = plan_decoder_state(builder, spec.decoder);
    out.state_images = plan_state_image_layout(spec.state_images);
    if (spec.replay_records) {
        out.replay_records = plan_gdn_replay_records(builder, *spec.replay_records);
    }
    if (spec.dflash) {
        const auto& draft = *spec.dflash;
        auto& layout      = out.dflash.emplace();
        if (draft.feature_rows <= 0 || !spec.state_images.dflash_local) {
            throw std::invalid_argument("DFlash resources require features and local state");
        }
        if (draft.full_layers) {
            layout.full = plan_paged_kv_cache(builder, draft.full_layers, spec.decoder.capacity,
                                              draft.kv_heads, draft.head_dim,
                                              KvCacheStorage::BFloat16, spec.decoder.kv_table_rows);
        }
        layout.prefill_features = builder.add_tensor(
            DType::BF16, {draft.feature_rows, dimension(spec.frame.prefill_chunk)}, kAlignment,
            "DFlash prefill target features");
        layout.prefill_positions =
            builder.add_tensor(DType::I32, {dimension(spec.frame.prefill_chunk)}, kAlignment,
                               "DFlash prefill target positions");
        layout.pending_features = builder.add_tensor(
            DType::BF16,
            {draft.feature_rows, dimension(std::uint64_t(spec.frame.draft_window) + 1),
             dimension(spec.frame.batch_capacity)},
            kAlignment, "DFlash residency target features");
    }
    if (spec.intervention) {
        const auto& edit = *spec.intervention;
        if (edit.hidden <= 0 || edit.layers <= 0)
            throw std::invalid_argument("Qwen3.5 residual intervention geometry is invalid");
        out.intervention = ResidualInterventionLayout{
            builder.add_tensor(DType::FP32, {edit.hidden, 2, edit.layers}, kAlignment,
                               "residual intervention directions"),
            builder.add_tensor(DType::FP32, {2, edit.layers}, kAlignment,
                               "residual intervention references")};
    }
    out.frame = plan_frames(builder, spec.frame);
    if (!spec.frame.causal_scoring) {
        out.token_counts = builder.add_tensor(
            DType::I32, {spec.frame.public_tokens, dimension(spec.frame.batch_capacity)},
            kAlignment, "residency sampling token counts");
    }
    out.context_bytes      = spec.context_bytes;
    out.host_context_bytes = spec.host_context_bytes;
    out.bytes              = builder.finish(kAlignment, "Qwen3.5 execution resources");
    return out;
}

std::vector<RecordSpec> ResourceLayout::context_records() const {
    std::vector<RecordSpec> records{
        {state_images.images.image_bytes, state_images.images.alignment},
        {decoder.text_kv.pages.page_bytes, decoder.text_kv.pages.alignment}};
    const auto* backend = decoder.mtp_kv           ? &*decoder.mtp_kv
                          : dflash && dflash->full ? &*dflash->full
                                                   : nullptr;
    if (backend) records.push_back({backend->pages.page_bytes, backend->pages.alignment});
    return records;
}

std::size_t ResourceLayout::pinned_bytes() const {
    const auto mapping_bytes = [](const PagedKVCacheLayout& cache) {
        const auto& table = cache.execution_tables.spec;
        return std::size_t(table.logical_page_capacity) * table.table_rows * sizeof(std::int64_t);
    };
    auto mapping = mapping_bytes(decoder.text_kv);
    if (decoder.mtp_kv) mapping += mapping_bytes(*decoder.mtp_kv);
    if (dflash && dflash->full) mapping += mapping_bytes(*dflash->full);
    const auto counts = token_counts ? std::size_t(frame.spec.public_tokens) *
                                           frame.spec.batch_capacity * sizeof(std::int32_t)
                                     : 0;
    return frame.host.bytes + 2 * mapping + counts;
}

DFlashPersistentState::DFlashPersistentState(MemorySpan arena, MemorySpan backing,
                                             const DFlashPersistentLayout& layout,
                                             CyclicKVCacheView& local_state)
    : local(local_state), prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    if (layout.full) full.emplace(arena, backing, *layout.full);
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer,
                                                          const Tensor& offsets) const {
    return local.layer_view(layer, offsets);
}

PagedKVBatchLayerView DFlashPersistentState::full_batch_layer(std::uint32_t layer) const {
    if (!full) throw std::logic_error("DFlash has no full-attention context");
    return full->batch_layer_view(layer);
}

Resources::Resources(DeviceContext& device, const ResourceLayout& layout,
                     const WorkspaceSpec& workspace, const residual::Profile* profile,
                     const StartupObserver& observer)
    : context_records(layout.context_records()), context(device.device(), layout.context_bytes),
      persistent(device.device(), layout.bytes),
      workspace_storage(device.device(), workspace.capacity),
      work(workspace_storage.span().subspan(0, workspace.general_capacity)),
      decoder(context.span(), persistent.span(), layout.decoder),
      state_images(context.span(), layout.state_images), frame(persistent.span(), layout.frame) {
    if (workspace.general_capacity == 0 ||
        (workspace.vision &&
         workspace.vision->general_capacity_bytes != workspace.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 execution workspace layout is inconsistent");
    }
    if (layout.host_context_bytes) {
        StartupPhaseScope phase(observer, StartupPhase::HostContextPin, StartupProgressUnit::Bytes,
                                layout.host_context_bytes);
        host_context = PinnedHostBuffer(layout.host_context_bytes);
        phase.complete(layout.host_context_bytes, layout.host_context_bytes);
    }
    const auto backing = persistent.span();
    if (layout.replay_records) {
        replay_records.emplace(backing, *layout.replay_records);
        replay_fold.emplace(*replay_records, state_images.linear().all_layers_view());
    }
    if (layout.dflash) {
        auto* local = state_images.dflash_local();
        if (!local) throw std::logic_error("DFlash StateImage has no local state");
        dflash.emplace(context.span(), backing, *layout.dflash, *local);
    }
    if (bool(layout.intervention) != bool(profile))
        throw std::invalid_argument("Qwen3.5 residual intervention layout and profile disagree");
    if (profile) {
        const std::size_t hidden = profile->hidden_size;
        const std::size_t count  = profile->layers.size();
        const Tensor directions  = layout.intervention->directions.bind(backing);
        const Tensor references  = layout.intervention->references.bind(backing);
        if (!count || directions.ne[0] != static_cast<std::int32_t>(hidden) ||
            directions.ne[2] != static_cast<std::int32_t>(count) ||
            references.ne[1] != static_cast<std::int32_t>(count))
            throw std::invalid_argument("Qwen3.5 residual intervention geometry disagrees");
        std::vector<float> direction_values(hidden * 2 * count), reference_values(2 * count);
        intervention_layers.reserve(count);
        for (std::size_t slot = 0; slot < count; ++slot) {
            const auto& edit = profile->layers[slot];
            if (edit.refusal.size() != hidden || edit.disclaimer.size() != hidden)
                throw std::invalid_argument("Qwen3.5 residual intervention direction is truncated");
            std::copy(edit.refusal.begin(), edit.refusal.end(),
                      direction_values.begin() + (2 * slot) * hidden);
            std::copy(edit.disclaimer.begin(), edit.disclaimer.end(),
                      direction_values.begin() + (2 * slot + 1) * hidden);
            reference_values[2 * slot]     = edit.refusal_reference;
            reference_values[2 * slot + 1] = edit.disclaimer_reference;
            intervention_layers.push_back(edit.layer);
        }
        CUDA_CHECK(cudaMemcpyAsync(directions.data, direction_values.data(), directions.bytes(),
                                   cudaMemcpyHostToDevice, device.stream()));
        CUDA_CHECK(cudaMemcpyAsync(references.data, reference_values.data(), references.bytes(),
                                   cudaMemcpyHostToDevice, device.stream()));
        CUDA_CHECK(cudaStreamSynchronize(device.stream()));
        intervention.emplace(
            execution::ResidualIntervention{intervention_layers, directions, references});
    }
    if (layout.token_counts) {
        token_counts = layout.token_counts->bind(backing);
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream()));
    }
}

PagedKVCache* Resources::backend_kv() noexcept {
    if (auto* mtp = decoder.mtp_cache()) return mtp;
    return dflash && dflash->full ? &*dflash->full : nullptr;
}

const PagedKVCache* Resources::backend_kv() const noexcept {
    if (const auto* mtp = decoder.mtp_cache()) return mtp;
    return dflash && dflash->full ? &*dflash->full : nullptr;
}

} // namespace ninfer::models::qwen3_5
