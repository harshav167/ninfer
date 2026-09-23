#pragma once

#include "core/arena.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <optional>

namespace ninfer {

// Packed secondary storage: pinned RAM or a second CUDA device. Never an
// attention operand. Synchronous copy methods preserve the inference stream and
// current device; an explicitly owned active-transfer pipeline may borrow staging.
class OffloadBuffer {
public:
    static constexpr std::size_t kStagingBytes = 32U * 1024U * 1024U;
    explicit OffloadBuffer(std::size_t bytes, std::optional<int> secondary_device = std::nullopt);
    ~OffloadBuffer();
    OffloadBuffer(const OffloadBuffer&)            = delete;
    OffloadBuffer& operator=(const OffloadBuffer&) = delete;

    [[nodiscard]] void* data() const noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    [[nodiscard]] bool secondary() const noexcept { return secondary_device_.has_value(); }

    [[nodiscard]] std::size_t staging_bytes() const noexcept;

    [[nodiscard]] int secondary_device() const noexcept { return secondary_device_.value_or(-1); }

    [[nodiscard]] void* staging_data() const noexcept {
        return staging_ ? staging_->data() : nullptr;
    }

    void zero(std::size_t offset, std::size_t bytes, cudaStream_t stream) const;
    void copy_from_buffer_2d(const OffloadBuffer& source, std::size_t source_offset,
                             std::size_t source_pitch, std::size_t destination_offset,
                             std::size_t destination_pitch, std::size_t width, std::size_t height,
                             cudaStream_t stream) const;

    void copy_from_device_2d(std::size_t offset, std::size_t pitch, const void* source,
                             std::size_t source_pitch, std::size_t width, std::size_t height,
                             cudaStream_t stream) const;
    void copy_to_device_2d(std::size_t offset, std::size_t pitch, void* destination,
                           std::size_t destination_pitch, std::size_t width, std::size_t height,
                           cudaStream_t stream) const;

private:
    void copy_2d(bool to_backing, std::size_t offset, std::size_t pitch, void* device_data,
                 std::size_t device_pitch, std::size_t width, std::size_t height,
                 cudaStream_t stream) const;
    void release() noexcept;

    std::size_t bytes_    = 0;
    int inference_device_ = 0;
    std::optional<int> secondary_device_;
    std::optional<PinnedHostBuffer> host_;
    std::optional<PinnedHostBuffer> staging_;
    void* secondary_data_          = nullptr;
    cudaStream_t secondary_stream_ = nullptr;
};

} // namespace ninfer
