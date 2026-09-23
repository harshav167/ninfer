#include "core/active_kv_stager.h"
#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer {
namespace {
class CurrentDevice {
public:
    explicit CurrentDevice(int target) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(target));
    }

    ~CurrentDevice() { (void)cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};

const OffloadBuffer& require_remote(const DeviceKVPagePool& pool) {
    if (!pool.overflow_buffer()) {
        throw std::invalid_argument("active staging requires overflow storage");
    }
    return *pool.overflow_buffer();
}
} // namespace

ActiveKVStagingLayout plan_active_kv_staging(LayoutBuilder& builder,
                                             const DeviceKVPagePoolLayout& pages,
                                             std::uint32_t planes_per_group) {
    ActiveKVStagingLayout out;
    if (pages.overflow_planes.empty()) { return out; }
    if (planes_per_group == 0 || pages.overflow_planes.size() % planes_per_group != 0) {
        throw std::invalid_argument("active KV staging plane groups are inconsistent");
    }
    out.planes_per_group = planes_per_group;
    out.groups = static_cast<std::uint32_t>(pages.overflow_planes.size() / planes_per_group);
    for (auto& buffer : out.buffers) {
        buffer.reserve(planes_per_group);
        for (std::uint32_t p = 0; p < planes_per_group; ++p) {
            const auto& source = pages.overflow_planes[p].storage;
            const auto& shape  = source.shape;
            auto region = builder.add_tensor(source.dtype, {shape[0], shape[1], shape[2], shape[3]},
                                             256, "active KV double-buffer plane");
            out.bytes += region.region.bytes;
            buffer.push_back(region);
        }
    }
    return out;
}

ActiveKVStager::ActiveKVStager(DeviceSpan backing, const ActiveKVStagingLayout& layout,
                               const DeviceKVPagePool& pool)
    : pool_(pool), remote_(require_remote(pool)), planes_per_group_(layout.planes_per_group),
      groups_(layout.groups) {
    if (planes_per_group_ == 0 || groups_ == 0 ||
        pool.plane_count() != static_cast<std::size_t>(planes_per_group_) * groups_) {
        throw std::invalid_argument("active KV staging inventory is invalid");
    }
    CUDA_CHECK(cudaGetDevice(&local_device_));
    for (std::size_t slot = 0; slot < 2; ++slot) {
        if (layout.buffers[slot].size() != planes_per_group_) {
            throw std::invalid_argument("active KV staging buffer is incomplete");
        }
        buffers_[slot].reserve(planes_per_group_);
        for (const auto& plane : layout.buffers[slot]) {
            buffers_[slot].push_back(plane.bind(backing));
        }
    }
    for (std::uint32_t group = 0; group < groups_; ++group) {
        for (std::uint32_t p = 0; p < planes_per_group_; ++p) {
            const auto& source      = pool.overflow_plane(group * planes_per_group_ + p);
            const auto& destination = buffers_[0][p];
            if (source.dtype != destination.dtype || source.bytes() != destination.bytes() ||
                !std::equal(source.nb, source.nb + 4, destination.nb) ||
                !std::equal(source.ne, source.ne + 4, destination.ne)) {
                throw std::invalid_argument("active KV groups must share exact plane geometry");
            }
        }
    }
    tile_bytes_ = remote_.staging_bytes() / 2;
    if (tile_bytes_ == 0 || !remote_.staging_data()) {
        throw std::invalid_argument("active KV transfer has no bounded staging tiles");
    }
    try {
        CUDA_CHECK(cudaStreamCreateWithFlags(&local_stream_, cudaStreamNonBlocking));
        for (std::size_t slot = 0; slot < 2; ++slot) {
            tiles_[slot].host =
                static_cast<std::byte*>(remote_.staging_data()) + slot * tile_bytes_;
            CUDA_CHECK(cudaEventCreateWithFlags(&ready_[slot], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&consumed_[slot], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&tiles_[slot].local, cudaEventDisableTiming));
        }
        {
            CurrentDevice remote(remote_.secondary_device());
            CUDA_CHECK(cudaStreamCreateWithFlags(&remote_stream_, cudaStreamNonBlocking));
            for (auto& tile : tiles_) {
                CUDA_CHECK(cudaEventCreateWithFlags(&tile.remote, cudaEventDisableTiming));
            }
        }
        worker_ = std::thread([this] { worker(); });
    } catch (...) {
        cleanup();
        throw;
    }
}

ActiveKVStager::~ActiveKVStager() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
    cleanup();
}

void ActiveKVStager::cleanup() noexcept {
    int previous = 0;
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(local_device_);
    if (local_stream_) { (void)cudaStreamSynchronize(local_stream_); }
    (void)cudaSetDevice(remote_.secondary_device());
    if (remote_stream_) { (void)cudaStreamSynchronize(remote_stream_); }
    for (auto& tile : tiles_) {
        if (tile.remote) {
            (void)cudaEventDestroy(tile.remote);
            tile.remote = nullptr;
        }
    }
    if (remote_stream_) {
        (void)cudaStreamDestroy(remote_stream_);
        remote_stream_ = nullptr;
    }
    (void)cudaSetDevice(local_device_);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        if (ready_[slot]) {
            (void)cudaEventDestroy(ready_[slot]);
            ready_[slot] = nullptr;
        }
        if (consumed_[slot]) {
            (void)cudaEventDestroy(consumed_[slot]);
            consumed_[slot] = nullptr;
        }
        if (tiles_[slot].local) {
            (void)cudaEventDestroy(tiles_[slot].local);
            tiles_[slot].local = nullptr;
        }
    }
    if (local_stream_) {
        (void)cudaStreamDestroy(local_stream_);
        local_stream_ = nullptr;
    }
    (void)cudaSetDevice(previous);
}

void ActiveKVStager::submit(Job job) {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&] { return count_ < jobs_.size() || failure_ || stopping_; });
    if (failure_) { std::rethrow_exception(failure_); }
    if (stopping_) { throw std::logic_error("active KV worker is stopping"); }
    jobs_[(head_ + count_) % jobs_.size()] = job;
    ++count_;
    condition_.notify_all();
}

void ActiveKVStager::begin(std::span<const std::uint32_t> reads,
                           std::span<const std::uint32_t> writes) {
    if (open_) { throw std::logic_error("active KV pass was not completed"); }
    const auto valid = [&](auto pages) {
        for (std::size_t i = 0; i < pages.size(); ++i) {
            if (pages[i] < pool_.resident_capacity_pages() || pages[i] >= pool_.capacity_pages() ||
                (i && pages[i - 1] >= pages[i])) {
                throw std::invalid_argument(
                    "active KV transfer pages are not sorted unique overflow pages");
            }
        }
    };
    valid(reads);
    valid(writes);
    if (!std::includes(reads.begin(), reads.end(), writes.begin(), writes.end())) {
        throw std::invalid_argument("active KV writes must be included in the staged read set");
    }
    {
        std::lock_guard lock(mutex_);
        if (busy_ || count_) { throw std::logic_error("active KV worker still has a prior pass"); }
        if (failure_) { std::rethrow_exception(failure_); }
        loaded_.fill(-1);
        reads_  = reads;
        writes_ = writes;
    }
    open_        = true;
    next_commit_ = 0;
    submit({0, false});
    if (groups_ > 1) { submit({1, false}); }
}

std::span<const Tensor> ActiveKVStager::prepare(std::uint32_t group, cudaStream_t compute) {
    if (!open_ || group != next_commit_ || group >= groups_) {
        throw std::logic_error("active KV prepare order is invalid");
    }
    const auto slot = group % 2;
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&] { return loaded_[slot] == group || failure_; });
    if (failure_) { std::rethrow_exception(failure_); }
    lock.unlock();
    CUDA_CHECK(cudaStreamWaitEvent(compute, ready_[slot], 0));
    return buffers_[slot];
}

void ActiveKVStager::commit(std::uint32_t group, cudaStream_t compute) {
    if (!open_ || group != next_commit_ || group >= groups_) {
        throw std::logic_error("active KV commit order is invalid");
    }
    CUDA_CHECK(cudaEventRecord(consumed_[group % 2], compute));
    submit({group, true});
    if (group + 2 < groups_) { submit({group + 2, false}); }
    ++next_commit_;
}

void ActiveKVStager::finish() {
    if (!open_ || next_commit_ != groups_) {
        throw std::logic_error("active KV pass is incomplete");
    }
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&] { return (!busy_ && count_ == 0) || failure_; });
    if (failure_) { std::rethrow_exception(failure_); }
    lock.unlock();
    CUDA_CHECK(cudaStreamSynchronize(local_stream_));
    {
        CurrentDevice remote(remote_.secondary_device());
        CUDA_CHECK(cudaStreamSynchronize(remote_stream_));
    }
    open_ = false;
}

void ActiveKVStager::worker() noexcept {
    try {
        CUDA_CHECK(cudaSetDevice(local_device_));
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, [&] { return count_ || stopping_; });
                if (count_ == 0 && stopping_) { return; }
                job   = jobs_[head_];
                head_ = (head_ + 1) % jobs_.size();
                --count_;
                busy_ = true;
                condition_.notify_all();
            }
            process(job);
            {
                std::lock_guard lock(mutex_);
                busy_ = false;
            }
            condition_.notify_all();
        }
    } catch (...) {
        std::lock_guard lock(mutex_);
        failure_ = std::current_exception();
        count_   = 0;
        busy_    = false;
        condition_.notify_all();
    }
}

void ActiveKVStager::process(Job job) {
    const auto slot = job.group % 2;
    if (job.write) { CUDA_CHECK(cudaStreamWaitEvent(local_stream_, consumed_[slot], 0)); }
    const auto pages = job.write ? writes_ : reads_;
    for (std::uint32_t p = 0; p < planes_per_group_; ++p) {
        const auto& remote_plane = pool_.overflow_plane(job.group * planes_per_group_ + p);
        const auto& local_plane  = buffers_[slot][p];
        std::size_t begin        = 0;
        while (begin < pages.size()) {
            std::size_t end = begin + 1;
            while (end < pages.size() && pages[end] == pages[end - 1] + 1) { ++end; }
            const auto page   = pages[begin] - pool_.resident_capacity_pages();
            const auto offset = static_cast<std::size_t>(page) * remote_plane.nb[3];
            const auto bytes  = (end - begin) * remote_plane.nb[3];
            copy_region(job.write, static_cast<std::byte*>(remote_plane.data) + offset,
                        static_cast<std::byte*>(local_plane.data) + offset, bytes);
            if (job.write) {
                write_bytes_.fetch_add(bytes, std::memory_order_relaxed);
            } else {
                read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
            }
            begin = end;
        }
    }
    if (!job.write) {
        CUDA_CHECK(cudaEventRecord(ready_[slot], local_stream_));
        std::lock_guard lock(mutex_);
        loaded_[slot] = job.group;
        condition_.notify_all();
    }
}

void ActiveKVStager::wait_tile(Tile& tile) {
    if (tile.pending_device < 0) { return; }
    CurrentDevice device(tile.pending_device);
    CUDA_CHECK(
        cudaEventSynchronize(tile.pending_device == local_device_ ? tile.local : tile.remote));
    tile.pending_device = -1;
}

void ActiveKVStager::copy_region(bool write, std::byte* remote, std::byte* local,
                                 std::size_t bytes) {
    for (std::size_t offset = 0; offset < bytes;) {
        Tile& tile = tiles_[next_tile_++ % 2];
        wait_tile(tile);
        const auto count = std::min(tile_bytes_, bytes - offset);
        if (write) {
            CUDA_CHECK(cudaMemcpyAsync(tile.host, local + offset, count, cudaMemcpyDeviceToHost,
                                       local_stream_));
            CUDA_CHECK(cudaEventRecord(tile.local, local_stream_));
            {
                CurrentDevice device(remote_.secondary_device());
                CUDA_CHECK(cudaStreamWaitEvent(remote_stream_, tile.local, 0));
                CUDA_CHECK(cudaMemcpyAsync(remote + offset, tile.host, count,
                                           cudaMemcpyHostToDevice, remote_stream_));
                CUDA_CHECK(cudaEventRecord(tile.remote, remote_stream_));
            }
            tile.pending_device = remote_.secondary_device();
        } else {
            {
                CurrentDevice device(remote_.secondary_device());
                CUDA_CHECK(cudaMemcpyAsync(tile.host, remote + offset, count,
                                           cudaMemcpyDeviceToHost, remote_stream_));
                CUDA_CHECK(cudaEventRecord(tile.remote, remote_stream_));
            }
            CUDA_CHECK(cudaStreamWaitEvent(local_stream_, tile.remote, 0));
            CUDA_CHECK(cudaMemcpyAsync(local + offset, tile.host, count, cudaMemcpyHostToDevice,
                                       local_stream_));
            CUDA_CHECK(cudaEventRecord(tile.local, local_stream_));
            tile.pending_device = local_device_;
        }
        offset += count;
    }
}

} // namespace ninfer
