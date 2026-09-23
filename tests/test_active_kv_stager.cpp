#include "core/active_kv_stager.h"
#include "core/device.h"
#include "core/host_kv_arena.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<unsigned char> pattern(std::size_t bytes, std::size_t plane) {
    std::vector<unsigned char> out(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        out[i] = static_cast<unsigned char>((i * 37U + i / 113U + plane * 19U) & 255U);
    }
    return out;
}

void test_active_residency(ninfer::DeviceContext& device) {
    constexpr std::uint32_t capacity = 8, resident = 4;
    ninfer::KVPageGeometry geometry;
    for (unsigned layer = 0; layer < 2; ++layer) {
        geometry.planes.push_back({ninfer::DType::FP8_E4M3FN, 256, 2, 256});
        geometry.planes.push_back({ninfer::DType::FP8_E4M3FN, 256, 2, 256});
        geometry.planes.push_back({ninfer::DType::FP16, 1, 2, 256});
        geometry.planes.push_back({ninfer::DType::FP16, 1, 2, 256});
    }
    ninfer::LayoutBuilder builder;
    const auto layout =
        ninfer::plan_device_kv_page_pool(builder, {.page_group_count          = capacity,
                                                   .geometry                  = geometry,
                                                   .resident_page_group_count = resident,
                                                   .overflow_device           = 1});
    const auto table_layout = ninfer::plan_kv_execution_tables(
        builder, {.logical_page_capacity = capacity, .table_rows = 3});
    ninfer::DeviceBuffer backing(builder.finish(256));
    ninfer::DeviceKVPagePool pool({backing.p, backing.bytes}, layout);
    ninfer::KVExecutionTablePool tables({backing.p, backing.bytes}, table_layout, pool);
    auto reservation = pool.reserve(capacity);
    std::vector<ninfer::DeviceKVPageLease> leases;
    leases.reserve(capacity);
    pool.materialize(*reservation, capacity, leases);
    std::vector<ninfer::DeviceKVPageHandle> handles;
    for (const auto& lease : leases) { handles.push_back(lease.handle()); }
    const auto host_layout = ninfer::plan_host_kv_page_layout(geometry);
    const std::array layouts{host_layout};
    ninfer::HostKVArena host(host_layout.page_stride * capacity * 2, layouts);
    auto source           = host.allocate(host_layout, capacity);
    auto destination      = host.allocate(host_layout, capacity);
    auto source_view      = host.writable_view(*source);
    auto destination_view = host.writable_view(*destination);
    std::vector<std::byte> expected(host_layout.page_stride * capacity, std::byte{0});
    for (std::uint32_t page = 0; page < capacity; ++page) {
        for (std::size_t p = 0; p < geometry.planes.size(); ++p) {
            const auto bytes = pattern(host_layout.planes[p].page_payload_bytes, page * 31 + p);
            std::memcpy(expected.data() + page * host_layout.page_stride +
                            host_layout.planes[p].offset,
                        bytes.data(), bytes.size());
        }
    }
    std::memcpy(source_view.data(), expected.data(), expected.size());
    pool.copy_from_host(host.view(*source), handles, device.stream);
    const auto check_payloads = [&] {
        std::memset(destination_view.data(), 0, expected.size());
        pool.copy_to_host(handles, destination_view, device.stream);
        device.synchronize();
        require(std::memcmp(destination_view.data(), expected.data(), expected.size()) == 0,
                "residency change corrupted a hot, cold, or shared logical page");
    };
    const auto check_gpu_row = [&](const ninfer::KVExecutionRowLease& row, std::size_t count) {
        std::vector<std::int32_t> actual(count);
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), tables.row(row.handle()).data,
                                   count * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                   device.stream));
        device.synchronize();
        require(std::equal(actual.begin(), actual.end(), tables.host_row(row.row_index()).begin()),
                "residency change failed to update the GPU execution table");
    };
    auto active = tables.acquire(0);
    auto other  = tables.acquire(1);
    const std::array hot{handles[4], handles[5], handles[6]};
    const std::array shared{handles[6], handles[0]};
    tables.publish(active.handle(), 0, hot, device.stream);
    tables.publish(other.handle(), 0, shared, device.stream);
    device.synchronize();
    require(tables.prioritize_active_residency(pool, device.stream),
            "cold resident cache prevented promotion of a fitting active working set");
    for (unsigned i = 0; i < hot.size(); ++i) {
        require(tables.host_row(0)[i] < resident,
                "active page remains remote despite fitting locally");
    }
    require(tables.host_row(1)[0] == tables.host_row(0)[2],
            "shared logical page acquired inconsistent physical mappings");
    check_gpu_row(active, hot.size());
    check_gpu_row(other, shared.size());
    check_payloads();
    require(!tables.prioritize_active_residency(pool, device.stream),
            "unchanged active working set caused repeated residency transfers");

    // Overcapacity must not ping-pong one active row against another.
    auto overflowing = tables.acquire(2);
    const std::array extra{handles[1], handles[2], handles[3], handles[7]};
    tables.publish(overflowing.handle(), 0, extra, device.stream);
    device.synchronize();
    require(!tables.prioritize_active_residency(pool, device.stream),
            "overcapacity displaced an already resident active page");
    check_payloads();
    overflowing.release();
    other.release();
    tables.truncate(active.handle(), 2);
    const auto stale = handles[0];
    leases[0].release();
    const std::array resumed{handles[7], handles[4]};
    tables.publish(active.handle(), 0, resumed, device.stream);
    device.synchronize();
    require(tables.prioritize_active_residency(pool, device.stream),
            "resumed context did not promote into free resident space");
    require(tables.host_row(0)[0] < resident && tables.host_row(0)[1] < resident,
            "free resident promotion left active KV remote");
    check_gpu_row(active, resumed.size());
    auto replacement_reservation = pool.reserve(1);
    leases[0]                    = pool.materialize_one(*replacement_reservation);
    handles[0]                   = leases[0].handle();
    bool stale_rejected          = false;
    try {
        pool.zero_pages(std::span(&stale, 1), device.stream);
    } catch (const std::invalid_argument&) { stale_rejected = true; }
    require(stale_rejected, "relocation and reuse revived a stale page capability");
    pool.copy_from_host(host.view(*source), handles, device.stream);
    check_payloads();
    pool.zero_pages(std::span(handles).subspan(1, 1), device.stream);
    std::fill_n(expected.data() + host_layout.page_stride, host_layout.page_stride, std::byte{0});
    check_payloads();
    pool.copy_page(handles[4], handles[2], device.stream);
    std::memcpy(expected.data() + 2 * host_layout.page_stride,
                expected.data() + 4 * host_layout.page_stride, host_layout.page_stride);
    check_payloads();

    // Fragmented free primary pages beat a preferred contiguous remote run.
    active.release();
    leases.clear();
    auto all = pool.reserve(capacity);
    pool.materialize(*all, capacity, leases);
    for (auto p : {1U, 3U, 6U, 7U}) { leases[p].release(); }
    auto growth = pool.reserve(2);
    std::vector<ninfer::DeviceKVPageLease> added;
    added.reserve(2);
    pool.materialize(*growth, 2, added, leases[5].handle());
    auto grown = tables.acquire(0);
    tables.publish(grown.handle(), 0, added, device.stream);
    require(tables.host_row(0)[0] < resident && tables.host_row(0)[1] < resident,
            "contiguous remote allocation overrode free resident fragments");
    check_gpu_row(grown, 2);
    std::cout << "PASS: active-first residency, retained/shared bytes, stable handles, "
                 "free-space promotion, no ping-pong, and resident-first allocation\n";
}

} // namespace

int main() {
    int devices       = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || devices < 2) {
        std::cout << "SKIP: two CUDA devices required\n";
        return 77;
    }
    try {
        CUDA_CHECK(status);
        ninfer::DeviceContext device(0);
        test_active_residency(device);
        constexpr std::uint32_t total = 1600, resident = 128, groups = 3, planes = 4;
        ninfer::KVPageGeometry geometry;
        for (unsigned group = 0; group < groups; ++group) {
            geometry.planes.push_back({ninfer::DType::FP8_E4M3FN, 256, 4, 256});
            geometry.planes.push_back({ninfer::DType::U8, 128, 4, 256});
            geometry.planes.push_back({ninfer::DType::FP16, 1, 4, 256});
            geometry.planes.push_back({ninfer::DType::U8, 16, 4, 256});
        }
        ninfer::LayoutBuilder builder;
        const auto layout =
            ninfer::plan_device_kv_page_pool(builder, {.page_group_count          = total,
                                                       .geometry                  = geometry,
                                                       .resident_page_group_count = resident,
                                                       .overflow_device           = 1});
        const auto stage_layout = ninfer::plan_active_kv_staging(builder, layout, planes);
        ninfer::DeviceBuffer allocation(builder.finish(256));
        ninfer::DeviceKVPagePool pool({allocation.p, allocation.bytes}, layout);
        auto reservation = pool.reserve(total);
        require(reservation.has_value(), "split pool did not reserve its complete budget");
        std::vector<ninfer::DeviceKVPageLease> leases;
        leases.reserve(total);
        pool.materialize(*reservation, total, leases);
        require(!pool.reserve(1), "split pool overcommitted its total capacity");
        const auto* remote = pool.overflow_buffer();
        require(remote && remote->secondary_device() == 1, "split pool has no secondary backing");
        std::size_t largest = 0;
        for (unsigned p = 0; p < groups * planes; ++p)
            largest = std::max(largest, pool.overflow_plane(p).bytes());
        ninfer::DeviceBuffer scratch(largest);
        for (unsigned p = 0; p < groups * planes; ++p) {
            const auto& local      = pool.plane(p);
            const auto local_bytes = pattern(local.bytes(), p + 100);
            CUDA_CHECK(
                cudaMemcpy(local.data, local_bytes.data(), local.bytes(), cudaMemcpyHostToDevice));
            const auto& plane = pool.overflow_plane(p);
            const auto bytes  = pattern(plane.bytes(), p);
            scratch.copy_from_host(bytes.data(), bytes.size());
            const auto offset = static_cast<std::size_t>(static_cast<std::byte*>(plane.data) -
                                                         static_cast<std::byte*>(remote->data()));
            remote->copy_from_device_2d(offset, bytes.size(), scratch.p, bytes.size(), bytes.size(),
                                        1, device.stream);
        }
        std::vector<std::uint32_t> reads(total - resident);
        std::iota(reads.begin(), reads.end(), resident);
        const std::array<std::uint32_t, 3> writes{resident, resident + 47, total - 1};
        ninfer::ActiveKVStager stager({allocation.p, allocation.bytes}, stage_layout, pool);
        stager.begin(reads, writes);
        for (unsigned group = 0; group < groups; ++group) {
            const auto staged = stager.prepare(group, device.stream);
            for (unsigned p = 0; p < planes; ++p) {
                std::vector<unsigned char> observed(staged[p].bytes());
                CUDA_CHECK(cudaMemcpyAsync(observed.data(), staged[p].data, observed.size(),
                                           cudaMemcpyDeviceToHost, device.stream));
                device.synchronize();
                require(observed == pattern(observed.size(), group * planes + p),
                        "prefetched plane differs from independent byte pattern");
                for (auto page : writes) {
                    auto* destination = static_cast<std::byte*>(staged[p].data) +
                                        (page - resident) * staged[p].nb[3];
                    CUDA_CHECK(
                        cudaMemsetAsync(destination, 0xa0 + group, staged[p].nb[3], device.stream));
                }
            }
            stager.commit(group, device.stream);
        }
        stager.finish();
        for (unsigned p = 0; p < groups * planes; ++p) {
            const auto& plane = pool.overflow_plane(p);
            auto expected     = pattern(plane.bytes(), p);
            for (auto page : writes) {
                std::fill_n(expected.data() + (page - resident) * plane.nb[3], plane.nb[3],
                            static_cast<unsigned char>(0xa0 + p / planes));
            }
            const auto offset = static_cast<std::size_t>(static_cast<std::byte*>(plane.data) -
                                                         static_cast<std::byte*>(remote->data()));
            remote->copy_to_device_2d(offset, plane.bytes(), scratch.p, plane.bytes(),
                                      plane.bytes(), 1, device.stream);
            std::vector<unsigned char> observed(plane.bytes());
            scratch.copy_to_host(observed.data(), observed.size());
            require(observed == expected, "writeback changed immutable bytes or lost a dirty page");
            const auto& local = pool.plane(p);
            observed.resize(local.bytes());
            CUDA_CHECK(
                cudaMemcpy(observed.data(), local.data, observed.size(), cudaMemcpyDeviceToHost));
            require(observed == pattern(observed.size(), p + 100),
                    "staging modified resident prefix bytes");
        }
        // Copy-on-write can cross either residency boundary without changing its payload.
        pool.copy_page(leases[resident].handle(), leases[0].handle(), device.stream);
        pool.copy_page(leases[0].handle(), leases[resident + 1].handle(), device.stream);
        pool.copy_page(leases[resident + 1].handle(), leases[resident + 2].handle(), device.stream);
        for (unsigned p = 0; p < groups * planes; ++p) {
            const auto& plane = pool.overflow_plane(p);
            const auto offset = static_cast<std::size_t>(static_cast<std::byte*>(plane.data) -
                                                         static_cast<std::byte*>(remote->data()));
            remote->copy_to_device_2d(offset + 2 * plane.nb[3], plane.nb[3], scratch.p, plane.nb[3],
                                      plane.nb[3], 1, device.stream);
            std::vector<unsigned char> observed(plane.nb[3]);
            scratch.copy_to_host(observed.data(), observed.size());
            require(std::all_of(observed.begin(), observed.end(),
                                [p](auto byte) {
                                    return byte == static_cast<unsigned char>(0xa0 + p / planes);
                                }),
                    "cross-tier copy-on-write changed packed KV");
        }
        require(stager.read_bytes() == layout.overflow_bytes,
                "active read accounting missed a plane");
        require(stager.write_bytes() == static_cast<std::uint64_t>(writes.size()) *
                                            (layout.overflow_bytes / (total - resident)),
                "active write accounting differs from dirty pages");
        int current = -1;
        CUDA_CHECK(cudaGetDevice(&current));
        require(current == 0, "pipeline changed inference CUDA device");
        std::cout
            << "PASS: split active budget, double-buffer prefetch, exact dirty-page writeback, "
               "resident isolation, and cross-tier COW\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
