#include "core/offload_buffer.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}

void roundtrip(std::size_t width, std::size_t height, int remote, cudaStream_t stream) {
    const std::size_t source_pitch  = width + 32;
    const std::size_t backing_pitch = width + 64;
    const std::size_t offset        = 256;
    std::vector<std::byte> expected(source_pitch * height);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = std::byte((i * 37U + i / 113U) & 255U);
    }
    ninfer::DeviceBuffer source(expected.size()), destination(expected.size());
    source.copy_from_host(expected.data(), expected.size());
    destination.fill(0);
    ninfer::OffloadBuffer cold(offset + backing_pitch * height, remote);
    require(cold.staging_bytes() <= ninfer::OffloadBuffer::kStagingBytes,
            "secondary storage exceeded bounded staging");
    cold.copy_from_device_2d(offset, backing_pitch, source.p, source_pitch, width, height, stream);
    // Mutate the source after parking: restoration must come from independent remote bytes.
    source.fill(0);
    cold.copy_to_device_2d(offset, backing_pitch, destination.p, source_pitch, width, height,
                           stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::byte> actual(expected.size());
    destination.copy_to_host(actual.data(), actual.size());
    for (std::size_t row = 0; row < height; ++row) {
        require(std::equal(expected.begin() + row * source_pitch,
                           expected.begin() + row * source_pitch + width,
                           actual.begin() + row * source_pitch),
                "secondary payload mismatch");
        require(std::all_of(actual.begin() + row * source_pitch + width,
                            actual.begin() + (row + 1) * source_pitch,
                            [](std::byte b) { return b == std::byte{0}; }),
                "secondary copy overwrote destination padding");
    }
    bool rejected = false;
    try {
        cold.copy_to_device_2d(cold.size() - 1, backing_pitch, destination.p, source_pitch, width,
                               height, stream);
    } catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "secondary out-of-bounds copy accepted");
    rejected = false;
    try {
        cold.copy_from_device_2d(offset, width - 1, destination.p, source_pitch, width, height,
                                 stream);
    } catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "secondary invalid pitch accepted");
    int current = -1;
    CUDA_CHECK(cudaGetDevice(&current));
    require(current == 0, "secondary copy did not restore current device");
}
} // namespace

int main() {
    int count         = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || count < 2) {
        std::cout << "SKIP: two CUDA devices required\n";
        return 77;
    }
    try {
        CUDA_CHECK(status);
        ninfer::DeviceContext device(0);
        bool rejected = false;
        try {
            ninfer::OffloadBuffer invalid(1024, 0);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "same-device secondary allocation accepted");
        rejected = false;
        try {
            ninfer::OffloadBuffer invalid(1024, count);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "out-of-range secondary device accepted");
        roundtrip(17, 4097, 1, device.stream);
        roundtrip(4096, 10000, 1, device.stream);
        roundtrip(ninfer::OffloadBuffer::kStagingBytes + 17, 2, 1, device.stream);
        std::cout << "Secondary GPU staged copy: exact bytes, pitch isolation, multiple tiles, "
                     "wide rows, bounds and device restoration passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
