#pragma once

#include "core/paged_kv_cache.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer {

struct ActiveKVStagingLayout {
    std::array<std::vector<TensorRegion>, 2> buffers;
    std::uint32_t planes_per_group = 0;
    std::uint32_t groups           = 0;
    std::size_t bytes              = 0;
};

[[nodiscard]] ActiveKVStagingLayout plan_active_kv_staging(LayoutBuilder& builder,
                                                           const DeviceKVPagePoolLayout& pages,
                                                           std::uint32_t planes_per_group);

// Raw physical-plane movement only. The caller supplies sorted unique execution
// page IDs to read/write and owns all sequence/frontier/commit semantics. The
// secondary GPU runs no model kernels. Two local plane groups and two bounded
// pinned tiles overlap next-group DMA with the current group's compute.
class ActiveKVStager {
public:
    ActiveKVStager(DeviceSpan backing, const ActiveKVStagingLayout& layout,
                   const DeviceKVPagePool& pool);
    ~ActiveKVStager();
    ActiveKVStager(const ActiveKVStager&)            = delete;
    ActiveKVStager& operator=(const ActiveKVStager&) = delete;

    void begin(std::span<const std::uint32_t> reads, std::span<const std::uint32_t> writes);
    [[nodiscard]] std::span<const Tensor> prepare(std::uint32_t group, cudaStream_t compute);
    void commit(std::uint32_t group, cudaStream_t compute);
    void finish();

    [[nodiscard]] std::uint64_t read_bytes() const noexcept { return read_bytes_.load(); }

    [[nodiscard]] std::uint64_t write_bytes() const noexcept { return write_bytes_.load(); }

private:
    struct Job {
        std::uint32_t group = 0;
        bool write          = false;
    };

    struct Tile {
        void* host         = nullptr;
        cudaEvent_t local  = nullptr;
        cudaEvent_t remote = nullptr;
        int pending_device = -1;
    };

    void submit(Job job);
    void worker() noexcept;
    void process(Job job);
    void copy_region(bool write, std::byte* remote, std::byte* local, std::size_t bytes);
    void wait_tile(Tile& tile);
    void cleanup() noexcept;

    const DeviceKVPagePool& pool_;
    const OffloadBuffer& remote_;
    std::array<std::vector<Tensor>, 2> buffers_;
    std::uint32_t planes_per_group_ = 0;
    std::uint32_t groups_           = 0;
    int local_device_               = 0;
    cudaStream_t local_stream_      = nullptr;
    cudaStream_t remote_stream_     = nullptr;
    std::array<cudaEvent_t, 2> ready_{};
    std::array<cudaEvent_t, 2> consumed_{};
    std::array<Tile, 2> tiles_{};
    std::size_t tile_bytes_ = 0;
    std::size_t next_tile_  = 0;
    std::span<const std::uint32_t> reads_;
    std::span<const std::uint32_t> writes_;
    std::array<Job, 8> jobs_{};
    std::size_t head_  = 0;
    std::size_t count_ = 0;
    std::array<std::int64_t, 2> loaded_{-1, -1};
    std::mutex mutex_;
    std::condition_variable condition_;
    bool busy_                 = false;
    bool stopping_             = false;
    bool open_                 = false;
    std::uint32_t next_commit_ = 0;
    std::exception_ptr failure_;
    std::thread worker_;
    std::atomic<std::uint64_t> read_bytes_{0};
    std::atomic<std::uint64_t> write_bytes_{0};
};

} // namespace ninfer
