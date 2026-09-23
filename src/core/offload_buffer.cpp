#include "core/offload_buffer.h"

#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {
class DeviceScope {
public:
    explicit DeviceScope(int target) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(target));
    }

    ~DeviceScope() { (void)cudaSetDevice(previous_); }

    DeviceScope(const DeviceScope&)            = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
private:
    int previous_ = 0;
};
} // namespace

OffloadBuffer::OffloadBuffer(std::size_t bytes, std::optional<int> secondary_device)
    : bytes_(bytes), secondary_device_(secondary_device) {
    CUDA_CHECK(cudaGetDevice(&inference_device_));
    if (!secondary_device_) {
        if (bytes_ != 0) { host_.emplace(bytes_); }
        return;
    }
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (*secondary_device_ < 0 || *secondary_device_ >= count ||
        *secondary_device_ == inference_device_) {
        throw std::invalid_argument("secondary cache requires a distinct visible CUDA device");
    }
    if (bytes_ == 0) { return; }
    staging_.emplace(std::min(bytes_, kStagingBytes));
    try {
        DeviceScope remote(*secondary_device_);
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        constexpr std::size_t headroom = 1024ULL * 1024ULL * 1024ULL;
        if (free <= headroom || bytes_ > free - headroom) {
            throw std::runtime_error("secondary cache allocation would leave less than 1 GiB free");
        }
        CUDA_CHECK(cudaMalloc(&secondary_data_, bytes_));
        CUDA_CHECK(cudaStreamCreateWithFlags(&secondary_stream_, cudaStreamNonBlocking));
    } catch (...) {
        release();
        throw;
    }
}

OffloadBuffer::~OffloadBuffer() { release(); }

void OffloadBuffer::release() noexcept {
    if (!secondary_data_ && !secondary_stream_) { return; }
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess ||
        cudaSetDevice(*secondary_device_) != cudaSuccess) {
        return;
    }
    if (secondary_stream_) {
        (void)cudaStreamSynchronize(secondary_stream_);
        (void)cudaStreamDestroy(secondary_stream_);
        secondary_stream_ = nullptr;
    }
    if (secondary_data_) {
        (void)cudaFree(secondary_data_);
        secondary_data_ = nullptr;
    }
    (void)cudaSetDevice(previous);
}

void* OffloadBuffer::data() const noexcept {
    return secondary_device_ ? secondary_data_ : (host_ ? host_->data() : nullptr);
}

std::size_t OffloadBuffer::staging_bytes() const noexcept {
    return staging_ ? staging_->size() : 0;
}

void OffloadBuffer::zero(std::size_t offset, std::size_t bytes, cudaStream_t stream) const {
    if (offset > bytes_ || bytes > bytes_ - offset) {
        throw std::out_of_range("offload zero is outside its backing");
    }
    if (bytes == 0) { return; }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (secondary_device_) {
        DeviceScope remote(*secondary_device_);
        CUDA_CHECK(
            cudaMemsetAsync(static_cast<std::byte*>(data()) + offset, 0, bytes, secondary_stream_));
        CUDA_CHECK(cudaStreamSynchronize(secondary_stream_));
    } else {
        std::memset(static_cast<std::byte*>(data()) + offset, 0, bytes);
    }
}

void OffloadBuffer::copy_from_buffer_2d(const OffloadBuffer& source, std::size_t source_offset,
                                        std::size_t source_pitch, std::size_t destination_offset,
                                        std::size_t destination_pitch, std::size_t width,
                                        std::size_t height, cudaStream_t stream) const {
    if (width == 0 || height == 0) { return; }
    const auto check = [&](std::size_t size, std::size_t offset, std::size_t pitch) {
        if (width > pitch || offset > size || width > size - offset ||
            height - 1 > (size - offset - width) / pitch) {
            throw std::out_of_range("offload buffer copy is outside its backing");
        }
    };
    check(source.size(), source_offset, source_pitch);
    check(size(), destination_offset, destination_pitch);
    if (secondary_device_ && source.secondary_device_ &&
        secondary_device_ != source.secondary_device_) {
        throw std::invalid_argument("active and inactive overflow must use the same secondary GPU");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto* destination = static_cast<std::byte*>(data()) + destination_offset;
    const auto* input = static_cast<const std::byte*>(source.data()) + source_offset;
    const int device  = secondary_device_.value_or(source.secondary_device());
    if (device < 0) {
        for (std::size_t row = 0; row < height; ++row) {
            std::memmove(destination + row * destination_pitch, input + row * source_pitch, width);
        }
        return;
    }
    DeviceScope remote(device);
    const cudaStream_t remote_stream =
        secondary_device_ ? secondary_stream_ : source.secondary_stream_;
    const auto kind = secondary_device_ ? (source.secondary_device_ ? cudaMemcpyDeviceToDevice
                                                                    : cudaMemcpyHostToDevice)
                                        : cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy2DAsync(destination, destination_pitch, input, source_pitch, width, height,
                                 kind, remote_stream));
    CUDA_CHECK(cudaStreamSynchronize(remote_stream));
}

void OffloadBuffer::copy_from_device_2d(std::size_t offset, std::size_t pitch, const void* source,
                                        std::size_t source_pitch, std::size_t width,
                                        std::size_t height, cudaStream_t stream) const {
    copy_2d(true, offset, pitch, const_cast<void*>(source), source_pitch, width, height, stream);
}

void OffloadBuffer::copy_to_device_2d(std::size_t offset, std::size_t pitch, void* destination,
                                      std::size_t destination_pitch, std::size_t width,
                                      std::size_t height, cudaStream_t stream) const {
    copy_2d(false, offset, pitch, destination, destination_pitch, width, height, stream);
}

void OffloadBuffer::copy_2d(bool to_backing, std::size_t offset, std::size_t pitch,
                            void* device_data, std::size_t device_pitch, std::size_t width,
                            std::size_t height, cudaStream_t stream) const {
    if (width == 0 || height == 0) { return; }
    if (!device_data || !data() || width > pitch || width > device_pitch || offset > bytes_ ||
        width > bytes_ - offset || (height - 1) > (bytes_ - offset - width) / pitch ||
        (height - 1) > (std::numeric_limits<std::size_t>::max() - width) / device_pitch) {
        throw std::out_of_range("offload copy is outside its backing or has invalid pitches");
    }
    int current = 0;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != inference_device_) {
        throw std::logic_error("offload copy must run on its owning inference device");
    }
    auto* backing = static_cast<std::byte*>(data()) + offset;
    auto* local   = static_cast<std::byte*>(device_data);
    if (!secondary_device_) {
        if (to_backing) {
            CUDA_CHECK(cudaMemcpy2DAsync(backing, pitch, local, device_pitch, width, height,
                                         cudaMemcpyDeviceToHost, stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(local, device_pitch, backing, pitch, width, height,
                                         cudaMemcpyHostToDevice, stream));
        }
        return;
    }
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        throw std::logic_error("secondary checkpoint transfers cannot run inside a CUDA graph");
    }
    // Order all local producers/consumers before switching device or reusing the staging tile.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t x = 0; x < width;) {
        const std::size_t tile_width = std::min(width - x, staging_->size());
        const std::size_t rows       = staging_->size() / tile_width;
        for (std::size_t y = 0; y < height;) {
            const std::size_t tile_height = std::min(height - y, rows);
            void* tile                    = staging_->data();
            if (to_backing) {
                CUDA_CHECK(cudaMemcpy2DAsync(tile, tile_width, local + y * device_pitch + x,
                                             device_pitch, tile_width, tile_height,
                                             cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                DeviceScope remote(*secondary_device_);
                CUDA_CHECK(cudaMemcpy2DAsync(backing + y * pitch + x, pitch, tile, tile_width,
                                             tile_width, tile_height, cudaMemcpyHostToDevice,
                                             secondary_stream_));
                CUDA_CHECK(cudaStreamSynchronize(secondary_stream_));
            } else {
                {
                    DeviceScope remote(*secondary_device_);
                    CUDA_CHECK(cudaMemcpy2DAsync(tile, tile_width, backing + y * pitch + x, pitch,
                                                 tile_width, tile_height, cudaMemcpyDeviceToHost,
                                                 secondary_stream_));
                    CUDA_CHECK(cudaStreamSynchronize(secondary_stream_));
                }
                CUDA_CHECK(cudaMemcpy2DAsync(local + y * device_pitch + x, device_pitch, tile,
                                             tile_width, tile_width, tile_height,
                                             cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            y += tile_height;
        }
        x += tile_width;
    }
}

} // namespace ninfer
