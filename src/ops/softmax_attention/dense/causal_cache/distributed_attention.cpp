// ninfer::ops - two-device FP8 causal Softmax Attention composition.
//
// Model execution and the merged output stay on the primary device. The secondary device computes
// only the attention contribution of the KV pages it physically owns, so no historical K/V is
// transported between the devices at all. What crosses the link is one query block, its new K/V
// rows, small metadata, the block table, and one compact partial result per call.
//
// Physical page IDs are global. The primary block table resolves logical pages to that global
// namespace, the primary planes own the ID range [0, primary_pages), and the secondary planes own
// [primary_pages, primary_pages + secondary_pages). Both devices run the same shard launcher over
// their own plane base and their own global range, so every visible key is contributed exactly
// once by exactly one device, and a device whose range holds no visible key publishes the neutral
// partial (zero components, -inf logsumexp) through the production path instead of a special case
// in the merge.
//
// Peer access is not assumed. One pinned host staging buffer carries the upload and the download
// of the compact remote partial, with explicit event ordering, explicit device restoration on
// every path, and no per-call allocation.

#include "ninfer/ops/distributed_attention.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include "ops/softmax_attention/dense/causal_cache/distributed_launch.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim              = 256;
constexpr float kExpectedScale               = 0.0625f;
constexpr std::int32_t kMaximumBatchSize     = 8;
constexpr KvCacheStorage kDistributedStorage = KvCacheStorage::Fp8E4M3Row256;

// Each shard publishes the normalized attention components plus their logsumexp, so a compact
// partial holds one FP32 row more than the head dimension.
constexpr std::int32_t kPartialComponents = kHeadDim + 1;

constexpr std::size_t kAlignment = 256;

[[noreturn]] void fail(const char* op, std::string message) {
    throw std::invalid_argument(std::string(op) + ": " + std::move(message));
}

void require(bool condition, const char* op, const char* message) {
    if (!condition) { fail(op, message); }
}

std::int32_t div_up(std::int32_t value, std::int32_t divisor) {
    return value / divisor + (value % divisor != 0);
}

std::size_t checked_size_add(std::size_t lhs, std::size_t rhs) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error("distributed attention allocation size overflows size_t");
    }
    return lhs + rhs;
}

std::size_t checked_size_mul(std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error("distributed attention allocation size overflows size_t");
    }
    return lhs * rhs;
}

std::size_t checked_size_product(std::initializer_list<std::size_t> factors) {
    std::size_t product = 1;
    for (const std::size_t factor : factors) { product = checked_size_mul(product, factor); }
    return product;
}

std::size_t checked_size_sum(std::initializer_list<std::size_t> terms) {
    std::size_t sum = 0;
    for (const std::size_t term : terms) { sum = checked_size_add(sum, term); }
    return sum;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0) { throw std::invalid_argument("alignment must be nonzero"); }
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? value : checked_size_add(value, alignment - remainder);
}

void log_cuda_failure(const char* operation, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA call failed during %s: %s: %s\n", operation,
                     cudaGetErrorName(err), cudaGetErrorString(err));
    }
}

// Restores the caller's device on every path, including exceptions and destructors. Neither the
// constructor nor the destructor may abort or throw: nested guards run inside `Impl`'s destructor,
// so a logged best-effort restore lets an original exception propagate instead of terminating the
// process. A guard that failed to bind reports it through `bound()`, because a call whose
// allocation or dispatch depends on the binding must not continue on an arbitrary device.
class DeviceGuard {
public:
    explicit DeviceGuard(int device) noexcept : device_(device) {
        const cudaError_t read = cudaGetDevice(&previous_);
        if (read != cudaSuccess) {
            log_cuda_failure("cudaGetDevice", read);
            previous_ = kUnknownDevice;
            return;
        }
        if (device_ == previous_) {
            bound_ = true;
            return;
        }
        const cudaError_t set = cudaSetDevice(device_);
        if (set != cudaSuccess) {
            log_cuda_failure("cudaSetDevice", set);
            return;
        }
        bound_ = true;
    }

    ~DeviceGuard() noexcept {
        if (previous_ != kUnknownDevice) {
            log_cuda_failure("cudaSetDevice", cudaSetDevice(previous_));
        }
    }

    DeviceGuard(const DeviceGuard&)            = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    [[nodiscard]] int device() const noexcept { return device_; }

    [[nodiscard]] bool bound() const noexcept { return bound_; }

private:
    static constexpr int kUnknownDevice = -1;

    int device_   = 0;
    int previous_ = kUnknownDevice;
    bool bound_   = false;
};

// Binds `device` and fails loudly when the binding did not take effect.
void require_bound(const DeviceGuard& guard, const char* op) {
    if (!guard.bound()) { fail(op, "could not bind device " + std::to_string(guard.device())); }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        fail(op, std::string(name) + " has an unexpected shape");
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (tensor.data == nullptr) { fail(op, std::string(name) + " must be non-null"); }
    if (!tensor.is_contiguous()) { fail(op, std::string(name) + " must be contiguous"); }
}

// Rejects a pointer that is host memory or provably belongs to another device. Called with the
// pointer's own device current, so a legitimate allocation always resolves cleanly.
void require_device(const void* pointer, int expected, const char* op, const char* name) {
    if (pointer == nullptr) { return; }
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
        (void)cudaGetLastError();
        fail(op, std::string(name) + " is not addressable device memory");
    }
    if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
        fail(op, std::string(name) + " must be device memory");
    }
    if (attributes.device != expected) {
        fail(op, std::string(name) + " is not on the declared device");
    }
}

// Checked host-side facts about one call.
struct CallShape {
    std::int32_t width           = 0;
    std::int32_t batch           = 0;
    std::int32_t primary_pages   = 0;
    std::int32_t secondary_pages = 0;
    std::int32_t total_pages     = 0;
    std::int32_t logical_pages   = 0;
    std::int32_t table_rows      = 0;
};

// One named operand of a live call. An exact pointer identity is not enough, because two views of
// one allocation can start at different offsets and still share bytes.
struct NamedOperand {
    const Tensor* tensor = nullptr;
    const char* name     = nullptr;
};

bool ranges_overlap(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.data == nullptr || rhs.data == nullptr) { return false; }
    const auto lhs_begin        = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin        = reinterpret_cast<std::uintptr_t>(rhs.data);
    const std::size_t lhs_bytes = lhs.bytes();
    const std::size_t rhs_bytes = rhs.bytes();
    // Compare distances rather than adding lengths to addresses, which would wrap for malformed
    // user-supplied views near UINTPTR_MAX.
    return lhs_begin <= rhs_begin ? rhs_begin - lhs_begin < lhs_bytes
                                  : lhs_begin - rhs_begin < rhs_bytes;
}

void require_non_overlapping(std::span<const NamedOperand> operands, const char* op) {
    for (std::size_t first = 0; first < operands.size(); ++first) {
        for (std::size_t second = first + 1; second < operands.size(); ++second) {
            if (ranges_overlap(*operands[first].tensor, *operands[second].tensor)) {
                fail(op, std::string(operands[first].name) + " overlaps " + operands[second].name);
            }
        }
    }
}

// Byte offsets of the one-page stand-in plane group a device uses when its global range is empty.
// The FP8 planes stay structurally valid, so the neutral partial comes from the production shard
// path rather than from a private fill kernel.
struct SparePlaneLayout {
    std::size_t k       = 0;
    std::size_t v       = 0;
    std::size_t k_scale = 0;
    std::size_t v_scale = 0;
    std::size_t total   = 0;
};

SparePlaneLayout spare_plane_layout(std::int32_t kv_heads) {
    const std::size_t codes = static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize * kv_heads;
    const std::size_t scales =
        static_cast<std::size_t>(kPagedKVPageSize) * kv_heads * sizeof(__half);
    SparePlaneLayout layout;
    std::size_t cursor = 0;
    layout.k           = cursor;
    cursor             = align_up(cursor + codes, kAlignment);
    layout.v           = cursor;
    cursor             = align_up(cursor + codes, kAlignment);
    layout.k_scale     = cursor;
    cursor             = align_up(cursor + scales, kAlignment);
    layout.v_scale     = cursor;
    cursor             = align_up(cursor + scales, kAlignment);
    layout.total       = cursor;
    return layout;
}

PagedKVStagedPlanes make_spare_planes(void* base, std::int32_t kv_heads) {
    auto* const bytes = static_cast<std::uint8_t*>(base);
    const auto layout = spare_plane_layout(kv_heads);
    const auto at     = [&](std::size_t offset) { return bytes + offset; };
    return {
        Tensor(at(layout.k), DType::FP8_E4M3FN, {kHeadDim, kPagedKVPageSize, kv_heads, 1}),
        Tensor(at(layout.v), DType::FP8_E4M3FN, {kHeadDim, kPagedKVPageSize, kv_heads, 1}),
        Tensor(at(layout.k_scale), DType::FP16, {1, kPagedKVPageSize, kv_heads, 1}),
        Tensor(at(layout.v_scale), DType::FP16, {1, kPagedKVPageSize, kv_heads, 1}),
    };
}

PagedKVBatchLayerView make_shard_view(const PagedKVStagedPlanes& planes, const Tensor& block_tables,
                                      std::int32_t kv_heads) {
    return {
        .k_pages       = planes.k,
        .v_pages       = planes.v,
        .k_scale_pages = planes.k_scale,
        .v_scale_pages = planes.v_scale,
        .block_tables  = block_tables,
        .head_dim      = kHeadDim,
        .num_kv_heads  = kv_heads,
        .storage       = kDistributedStorage,
        .staged        = {},
    };
}

Tensor partial_view(const DeviceBuffer& buffer, std::int32_t q_heads, std::int32_t width,
                    std::int32_t batch) {
    return Tensor(buffer.p, DType::FP32, {kPartialComponents, q_heads, width, batch});
}

// Every block-table entry addressed by a live prefix must resolve to exactly one of the two
// physical page ranges. Validate the row index even for an empty contribution because the kernel
// reads it before taking its neutral-work path. Position structure is also part of this Op's
// input contract: live columns are sequential, and inert masked tails repeat the final live
// position (empty rows use zero positions).
void validate_live_table(const std::int32_t* table, const std::int32_t* rows,
                         const std::int32_t* positions, const std::int32_t* valid_columns,
                         const CallShape& shape, CausalAttentionExecutionEnvelope envelope,
                         const char* op) {
    for (std::int32_t row = 0; row < shape.batch; ++row) {
        const std::int32_t live      = valid_columns == nullptr ? shape.width : valid_columns[row];
        const std::int32_t table_row = rows[row];
        require(table_row >= 0 && table_row < shape.table_rows, op,
                "a block table row is outside the declared pool");

        const std::size_t row_offset = static_cast<std::size_t>(row) * shape.width;
        if (live == 0) {
            for (std::int32_t column = 0; column < shape.width; ++column) {
                require(positions[row_offset + column] == 0, op,
                        "an empty row must use zero positions");
            }
            continue;
        }

        const std::int32_t first_position = positions[row_offset];
        require(first_position >= 0, op, "live positions must be nonnegative");
        std::int32_t last_position = first_position;
        for (std::int32_t column = 0; column < live; ++column) {
            const std::int32_t position = positions[row_offset + column];
            const std::int64_t expected = static_cast<std::int64_t>(first_position) + column;
            require(position >= 0, op, "live positions must be nonnegative");
            require(static_cast<std::uint32_t>(position) < envelope.max_visible_keys, op,
                    "a live position exceeds the visible-key envelope");
            require(position == expected, op, "live positions must be sequential");
            last_position = position;
        }
        require(static_cast<std::uint32_t>(last_position) + 1 <= envelope.max_visible_keys, op,
                "the live prefix exceeds the visible-key envelope");

        if (valid_columns != nullptr) {
            for (std::int32_t column = live; column < shape.width; ++column) {
                require(positions[row_offset + column] == last_position, op,
                        "masked tail positions must repeat the final live position");
            }
        }

        const std::int32_t live_pages = div_up(last_position + 1, kPagedKVPageSize);
        const std::int32_t* entries =
            table + static_cast<std::int64_t>(table_row) * shape.logical_pages;
        for (std::int32_t page = 0; page < live_pages; ++page) {
            require(entries[page] >= 0 && entries[page] < shape.total_pages, op,
                    "a live block table entry is outside the primary and secondary planes");
        }
    }
}

struct ResourceLayout {
    std::int32_t query_heads    = 0;
    std::int32_t kv_heads       = 0;
    std::size_t workspace_bytes = 0;
    std::size_t partial_bytes   = 0;
    std::size_t query_bytes     = 0;
    std::size_t kv_bytes        = 0;
    std::size_t position_bytes  = 0;
    std::size_t row_bytes       = 0;
    std::size_t table_bytes     = 0;
    std::size_t upload_bytes    = 0;
    std::size_t staging_bytes   = 0;
    SparePlaneLayout spare_planes;
    DistributedAttentionResourceEstimate estimate;
};

ResourceLayout resource_layout(int primary, int secondary, AttentionHeadGeometry geometry,
                               std::int32_t max_width, std::int32_t max_batch,
                               std::int32_t max_table_rows, std::int32_t max_logical_pages) {
    constexpr const char* op = "DistributedAttention";
    require(primary >= 0 && secondary >= 0, op, "device indices must be nonnegative");
    require(primary != secondary, op, "two-device attention needs two distinct devices");
    require(valid_attention_head_geometry(geometry) && geometry.head_dim == kHeadDim &&
                ((geometry.query_heads == 24 && geometry.kv_heads == 4) ||
                 (geometry.query_heads == 16 && geometry.kv_heads == 2)),
            op, "unsupported head geometry");
    require(max_width >= 1 &&
                max_width <= static_cast<std::int32_t>(kCausalAttentionMaximumVisibleKeys),
            op, "the maximum query width is outside the causal domain");
    require(max_batch >= 1 && max_batch <= kMaximumBatchSize, op,
            "the maximum batch is out of range");
    require(max_table_rows >= max_batch, op,
            "the table pool must hold at least one row per request");
    require(max_logical_pages >= 1 &&
                max_logical_pages <=
                    div_up(static_cast<std::int32_t>(kCausalAttentionMaximumVisibleKeys),
                           kPagedKVPageSize),
            op, "the logical page capacity exceeds the causal domain");

    ResourceLayout layout;
    layout.query_heads = geometry.query_heads;
    layout.kv_heads    = geometry.kv_heads;

    // Size both arenas from the launcher's own allocation recipe at the widest envelope the route
    // admits. The shard launcher forms one chunk per tile width and allocates under a per-chunk
    // scope, so the requirement is the largest single-chunk peak over reachable batch sizes.
    const CausalAttentionExecutionEnvelope widest{1u, kCausalAttentionMaximumVisibleKeys};
    for (std::int32_t batch = 1; batch <= max_batch; ++batch) {
        layout.workspace_bytes =
            std::max(layout.workspace_bytes,
                     detail::fp8_attention_shard_workspace_capacity_bytes(
                         layout.query_heads, kDistributedStorage, batch, max_width, widest));
    }
    require(layout.workspace_bytes > 0, op,
            "the small-tile shard route reported no transient capacity");

    layout.partial_bytes = checked_size_product(
        {static_cast<std::size_t>(kPartialComponents), static_cast<std::size_t>(layout.query_heads),
         static_cast<std::size_t>(max_width), static_cast<std::size_t>(max_batch), sizeof(float)});
    layout.query_bytes = checked_size_product(
        {static_cast<std::size_t>(kHeadDim), static_cast<std::size_t>(layout.query_heads),
         static_cast<std::size_t>(max_width), static_cast<std::size_t>(max_batch),
         sizeof(std::uint16_t)});
    layout.kv_bytes = checked_size_product(
        {static_cast<std::size_t>(kHeadDim), static_cast<std::size_t>(layout.kv_heads),
         static_cast<std::size_t>(max_width), static_cast<std::size_t>(max_batch),
         sizeof(std::uint16_t)});
    layout.position_bytes =
        checked_size_product({static_cast<std::size_t>(max_width),
                              static_cast<std::size_t>(max_batch), sizeof(std::int32_t)});
    layout.row_bytes =
        checked_size_product({static_cast<std::size_t>(max_batch), sizeof(std::int32_t)});
    layout.table_bytes =
        checked_size_product({static_cast<std::size_t>(max_logical_pages),
                              static_cast<std::size_t>(max_table_rows), sizeof(std::int32_t)});
    // Each staged region begins at an independently aligned offset; the byte count is the sum of
    // those aligned regions, not one alignment applied to the total.
    const auto aligned  = [](std::size_t bytes) { return align_up(bytes, kAlignment); };
    layout.upload_bytes = checked_size_sum(
        {aligned(layout.query_bytes), aligned(layout.kv_bytes), aligned(layout.kv_bytes),
         aligned(layout.position_bytes), aligned(layout.row_bytes), aligned(layout.row_bytes),
         aligned(layout.table_bytes)});
    layout.staging_bytes = std::max(layout.upload_bytes, aligned(layout.partial_bytes));
    layout.spare_planes  = spare_plane_layout(layout.kv_heads);
    layout.estimate.primary_bytes =
        checked_size_sum({layout.partial_bytes, layout.partial_bytes, layout.workspace_bytes,
                          layout.spare_planes.total});
    layout.estimate.secondary_bytes =
        checked_size_sum({layout.partial_bytes, layout.workspace_bytes, layout.query_bytes,
                          layout.kv_bytes, layout.kv_bytes, layout.position_bytes, layout.row_bytes,
                          layout.row_bytes, layout.table_bytes, layout.spare_planes.total});
    layout.estimate.host_staging_bytes = layout.staging_bytes;
    return layout;
}

} // namespace

struct DistributedAttention::Impl {
    int primary   = 0;
    int secondary = 0;
    AttentionHeadGeometry geometry{};
    std::int32_t max_width         = 0;
    std::int32_t max_batch         = 1;
    std::int32_t max_table_rows    = 1;
    std::int32_t max_logical_pages = 1;

    // Worst-case shard decomposition and the resources derived from it. All of these are startup
    // quantities: no call allocates, copies host metadata into a cache, or grows a buffer.
    std::size_t workspace_bytes = 0;
    std::size_t upload_bytes    = 0;
    std::size_t staging_bytes   = 0;

    // Primary-owned device resources.
    DeviceBuffer primary_partial;        // local shard partial [257,Hq,maxW,maxB] FP32
    DeviceBuffer primary_remote_partial; // remote shard partial brought back for the merge
    DeviceBuffer primary_workspace;
    DeviceBuffer primary_spare_plane;

    // Secondary-owned device resources: the uploaded operands are private to this device, and the
    // remote planes are read in place, so no historical K/V is ever copied.
    DeviceBuffer secondary_partial;
    DeviceBuffer secondary_workspace;
    DeviceBuffer secondary_query;
    DeviceBuffer secondary_key;
    DeviceBuffer secondary_value;
    DeviceBuffer secondary_positions;
    DeviceBuffer secondary_valid;
    DeviceBuffer secondary_rows;
    DeviceBuffer secondary_table;
    DeviceBuffer secondary_spare_plane;

    // PinnedHostBuffer owns exactly one cudaMallocHost and has no default constructor, so the
    // staging buffer is an optional member filled once at startup.
    std::optional<PinnedHostBuffer> staging;
    cudaStream_t secondary_stream       = nullptr;
    cudaEvent_t primary_inputs_consumed = nullptr;
    cudaEvent_t primary_settled         = nullptr;
    cudaEvent_t secondary_partial_ready = nullptr;
    bool primary_pending                = false;
    bool poisoned                       = false;

    std::size_t primary_bytes   = 0;
    std::size_t secondary_bytes = 0;

    // Reproduces the documented post-call settle while an exception unwinds a call that already
    // gave the secondary stream work. Defined below, where the settle policy can use every member.
    struct UnwindGuard;

    ~Impl();
};

DistributedAttention::Impl::~Impl() {
    {
        DeviceGuard guard(primary);
        if (primary_pending) {
            log_cuda_failure("cudaEventSynchronize(primary settle)",
                             cudaEventSynchronize(primary_settled));
            primary_pending = false;
        }
        primary_partial        = DeviceBuffer{};
        primary_remote_partial = DeviceBuffer{};
        primary_workspace      = DeviceBuffer{};
        primary_spare_plane    = DeviceBuffer{};
        if (primary_inputs_consumed != nullptr) {
            log_cuda_failure("cudaEventDestroy(primary inputs consumed)",
                             cudaEventDestroy(primary_inputs_consumed));
            primary_inputs_consumed = nullptr;
        }
        if (primary_settled != nullptr) {
            log_cuda_failure("cudaEventDestroy(primary settled)",
                             cudaEventDestroy(primary_settled));
            primary_settled = nullptr;
        }
    }

    {
        DeviceGuard guard(secondary);
        // This stream can still own DMA reads from pinned staging or cache-plane work after an
        // exceptional execute path. Settle it before freeing any secondary resource or allowing
        // the optional pinned buffer to destruct below.
        if (secondary_stream != nullptr) {
            log_cuda_failure("cudaStreamSynchronize(secondary teardown)",
                             cudaStreamSynchronize(secondary_stream));
        }
        secondary_partial     = DeviceBuffer{};
        secondary_workspace   = DeviceBuffer{};
        secondary_query       = DeviceBuffer{};
        secondary_key         = DeviceBuffer{};
        secondary_value       = DeviceBuffer{};
        secondary_positions   = DeviceBuffer{};
        secondary_valid       = DeviceBuffer{};
        secondary_rows        = DeviceBuffer{};
        secondary_table       = DeviceBuffer{};
        secondary_spare_plane = DeviceBuffer{};
        if (secondary_partial_ready != nullptr) {
            log_cuda_failure("cudaEventDestroy(secondary partial ready)",
                             cudaEventDestroy(secondary_partial_ready));
            secondary_partial_ready = nullptr;
        }
        if (secondary_stream != nullptr) {
            log_cuda_failure("cudaStreamDestroy(secondary)", cudaStreamDestroy(secondary_stream));
            secondary_stream = nullptr;
        }
    }
}

// Reproduces the documented post-call settle when control leaves `execute` by an exception after
// the secondary stream already holds remote work. Without it, the next call would overwrite the
// pinned staging buffer and reset the secondary arena while that work still reads them. The
// destructor is `noexcept` by construction: every call below only logs, and `DeviceGuard` cannot
// throw, so an original exception propagates instead of terminating.
struct DistributedAttention::Impl::UnwindGuard {
    UnwindGuard(Impl* impl, cudaStream_t caller) noexcept : impl_(impl), caller_(caller) {}

    ~UnwindGuard() noexcept {
        if (!armed_) { return; }
        const auto settle = [&](const char* operation, int device, cudaStream_t stream) {
            DeviceGuard guard(device);
            if (!guard.bound()) {
                impl_->poisoned = true;
                return;
            }
            const cudaError_t error = cudaStreamSynchronize(stream);
            if (error != cudaSuccess) {
                impl_->poisoned = true;
                log_cuda_failure(operation, error);
            }
        };
        // The caller stream owns the primary shard/merge and the host-to-device read of the
        // compact remote result. The secondary stream owns every remote staging read/write.
        settle("cudaStreamSynchronize(caller)", impl_->primary, caller_);
        settle("cudaStreamSynchronize(secondary)", impl_->secondary, impl_->secondary_stream);
    }

    void arm() noexcept { armed_ = true; }

    void disarm() noexcept { armed_ = false; }

private:
    Impl* impl_          = nullptr;
    cudaStream_t caller_ = nullptr;
    bool armed_          = false;
};

DistributedAttention::DistributedAttention(int primary, int secondary,
                                           AttentionHeadGeometry geometry, std::int32_t max_width,
                                           std::int32_t max_batch, std::int32_t max_table_rows,
                                           std::int32_t max_logical_pages)
    : impl_(std::make_unique<Impl>()) {
    constexpr const char* op = "DistributedAttention";
    Impl& self               = *impl_;
    self.primary             = primary;
    self.secondary           = secondary;
    self.geometry            = geometry;
    self.max_width           = max_width;
    self.max_batch           = max_batch;
    self.max_table_rows      = max_table_rows;
    self.max_logical_pages   = max_logical_pages;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    require(primary >= 0 && primary < device_count, op, "the primary device is not available");
    require(secondary >= 0 && secondary < device_count, op,
            "the secondary device is not available");
    const ResourceLayout sizes = resource_layout(primary, secondary, geometry, max_width, max_batch,
                                                 max_table_rows, max_logical_pages);
    self.workspace_bytes       = sizes.workspace_bytes;
    self.upload_bytes          = sizes.upload_bytes;
    self.staging_bytes         = sizes.staging_bytes;
    {
        DeviceGuard guard(primary);
        // cudaMalloc allocates on the device that is current, so a binding that did not take
        // effect would place this device's buffers on the wrong one.
        require_bound(guard, op);
        self.primary_partial        = DeviceBuffer(sizes.partial_bytes);
        self.primary_remote_partial = DeviceBuffer(sizes.partial_bytes);
        self.primary_workspace      = DeviceBuffer(sizes.workspace_bytes);
        self.primary_spare_plane    = DeviceBuffer(sizes.spare_planes.total);
        // A stand-in plane group is only ever handed to a shard whose global range is empty, so
        // it should never be decoded. Zeroing keeps a codec read of it numerically neutral rather
        // than undefined, at a one-off startup cost of one page.
        self.primary_spare_plane.fill(0);
        CUDA_CHECK(cudaEventCreateWithFlags(&self.primary_inputs_consumed, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&self.primary_settled, cudaEventDisableTiming));
    }
    {
        DeviceGuard guard(secondary);
        require_bound(guard, op);
        self.secondary_partial     = DeviceBuffer(sizes.partial_bytes);
        self.secondary_workspace   = DeviceBuffer(sizes.workspace_bytes);
        self.secondary_query       = DeviceBuffer(sizes.query_bytes);
        self.secondary_key         = DeviceBuffer(sizes.kv_bytes);
        self.secondary_value       = DeviceBuffer(sizes.kv_bytes);
        self.secondary_positions   = DeviceBuffer(sizes.position_bytes);
        self.secondary_valid       = DeviceBuffer(sizes.row_bytes);
        self.secondary_rows        = DeviceBuffer(sizes.row_bytes);
        self.secondary_table       = DeviceBuffer(sizes.table_bytes);
        self.secondary_spare_plane = DeviceBuffer(sizes.spare_planes.total);
        self.secondary_spare_plane.fill(0);
        CUDA_CHECK(cudaEventCreateWithFlags(&self.secondary_partial_ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(&self.secondary_stream, cudaStreamNonBlocking));
    }

    self.staging.emplace(sizes.staging_bytes);
    self.primary_bytes =
        checked_size_sum({self.primary_partial.bytes, self.primary_remote_partial.bytes,
                          self.primary_workspace.bytes, self.primary_spare_plane.bytes});
    self.secondary_bytes = checked_size_sum(
        {self.secondary_partial.bytes, self.secondary_workspace.bytes, self.secondary_query.bytes,
         self.secondary_key.bytes, self.secondary_value.bytes, self.secondary_positions.bytes,
         self.secondary_valid.bytes, self.secondary_rows.bytes, self.secondary_table.bytes,
         self.secondary_spare_plane.bytes});
}

DistributedAttentionResourceEstimate DistributedAttention::estimate_resources(
    int primary, int secondary, AttentionHeadGeometry geometry, std::int32_t max_width,
    std::int32_t max_batch, std::int32_t max_table_rows, std::int32_t max_logical_pages) {
    return resource_layout(primary, secondary, geometry, max_width, max_batch, max_table_rows,
                           max_logical_pages)
        .estimate;
}

DistributedAttention::~DistributedAttention() = default;

void DistributedAttention::execute(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& positions, const Tensor& valid_columns,
                                   const Tensor& table_rows, float scale,
                                   PagedKVBatchLayerView primary_cache,
                                   PagedKVStagedPlanes secondary_cache,
                                   CausalAttentionExecutionEnvelope envelope, Tensor& out,
                                   cudaStream_t stream) {
    constexpr const char* op = "DistributedAttention::execute";
    Impl& self               = *impl_;
    require(!self.poisoned, op,
            "the instance cannot be reused after a CUDA stream failed to settle");
    const std::int32_t q_heads  = self.geometry.query_heads;
    const std::int32_t kv_heads = self.geometry.kv_heads;

    // ---- Shape, layout and resource validation. -------------------------------------
    require(q.dtype == DType::BF16 && k.dtype == DType::BF16 && v.dtype == DType::BF16 &&
                out.dtype == DType::BF16,
            op, "q/k/v/out must be BF16");
    require(positions.dtype == DType::I32, op, "positions must be I32");
    require(table_rows.dtype == DType::I32, op, "table rows must be I32");
    const bool masked = valid_columns.data != nullptr;
    require(!masked || valid_columns.dtype == DType::I32, op, "valid columns must be I32");
    require(std::isfinite(scale) && std::abs(scale - kExpectedScale) <= 1.0e-6f, op,
            "scale must be 1/sqrt(256)");

    CallShape shape;
    shape.width = q.ne[2];
    shape.batch = q.ne[3];
    require(shape.width >= 1 && shape.width <= self.max_width, op,
            "the query width exceeds the declared maximum");
    require(shape.batch >= 1 && shape.batch <= self.max_batch, op,
            "the batch exceeds the declared maximum");
    require(shape.batch == 1 || shape.width <= 16, op,
            "batches larger than one support query widths only through 16");
    require_shape(q, kHeadDim, q_heads, shape.width, shape.batch, op, "q");
    require_shape(k, kHeadDim, kv_heads, shape.width, shape.batch, op, "k");
    require_shape(v, kHeadDim, kv_heads, shape.width, shape.batch, op, "v");
    require_shape(out, kHeadDim, q_heads, shape.width, shape.batch, op, "out");
    require_shape(positions, shape.width, shape.batch, 1, 1, op, "positions");
    require_shape(table_rows, shape.batch, 1, 1, 1, op, "table rows");
    if (masked) { require_shape(valid_columns, shape.batch, 1, 1, 1, op, "valid columns"); }
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(table_rows, op, "table rows");
    if (masked) { require_contiguous_nonnull(valid_columns, op, "valid columns"); }
    require_contiguous_nonnull(out, op, "out");

    require(primary_cache.storage == kDistributedStorage, op,
            "the two-device route stores row-scaled FP8 E4M3 KV");
    validate_staged_kv_planes(primary_cache.staged, kDistributedStorage, kv_heads);
    require(!primary_cache.staged.enabled(), op,
            "the primary cache view must hold only primary-resident pages");
    require(primary_cache.head_dim == kHeadDim && primary_cache.num_kv_heads == kv_heads, op,
            "the primary cache head geometry does not match");
    require(primary_cache.block_tables.dtype == DType::I32, op, "block tables must be I32");
    require_contiguous_nonnull(primary_cache.block_tables, op, "block tables");
    shape.logical_pages = primary_cache.block_tables.ne[0];
    shape.table_rows    = primary_cache.block_tables.ne[1];
    require(shape.logical_pages >= 1 && shape.logical_pages <= self.max_logical_pages, op,
            "the block table length exceeds the declared logical page capacity");
    require(shape.table_rows <= self.max_table_rows, op,
            "the block table pool exceeds the declared row capacity");
    require(shape.table_rows >= shape.batch, op, "the block table pool is shorter than the batch");

    const bool primary_planes_present = primary_cache.k_pages.data != nullptr;
    if (primary_planes_present) {
        require(primary_cache.k_pages.dtype == DType::FP8_E4M3FN &&
                    primary_cache.v_pages.dtype == DType::FP8_E4M3FN &&
                    primary_cache.k_scale_pages.dtype == DType::FP16 &&
                    primary_cache.v_scale_pages.dtype == DType::FP16,
                op, "the primary cache planes have an unexpected dtype");
        shape.primary_pages = primary_cache.k_pages.ne[3];
        require(shape.primary_pages >= 1, op, "the primary cache plane is empty");
        require_shape(primary_cache.k_pages, kHeadDim, kPagedKVPageSize, kv_heads,
                      shape.primary_pages, op, "primary cache k pages");
        require_shape(primary_cache.v_pages, kHeadDim, kPagedKVPageSize, kv_heads,
                      shape.primary_pages, op, "primary cache v pages");
        require_shape(primary_cache.k_scale_pages, 1, kPagedKVPageSize, kv_heads,
                      shape.primary_pages, op, "primary cache k scale pages");
        require_shape(primary_cache.v_scale_pages, 1, kPagedKVPageSize, kv_heads,
                      shape.primary_pages, op, "primary cache v scale pages");
        require_contiguous_nonnull(primary_cache.k_pages, op, "primary cache k pages");
        require_contiguous_nonnull(primary_cache.v_pages, op, "primary cache v pages");
        require_contiguous_nonnull(primary_cache.k_scale_pages, op, "primary cache k scale pages");
        require_contiguous_nonnull(primary_cache.v_scale_pages, op, "primary cache v scale pages");
    } else {
        require(primary_cache.v_pages.data == nullptr &&
                    primary_cache.k_scale_pages.data == nullptr &&
                    primary_cache.v_scale_pages.data == nullptr,
                op, "an absent primary plane must be absent in every plane");
    }

    validate_staged_kv_planes(secondary_cache, kDistributedStorage, kv_heads);
    if (secondary_cache.enabled()) { shape.secondary_pages = secondary_cache.k.ne[3]; }
    require(primary_planes_present || shape.secondary_pages > 0, op,
            "at least one device must own cache pages");
    require(shape.primary_pages <= std::numeric_limits<std::int32_t>::max() - shape.secondary_pages,
            op, "the global physical page namespace exceeds I32 indexing");
    shape.total_pages = shape.primary_pages + shape.secondary_pages;
    require(envelope.min_visible_keys >= 1 &&
                envelope.min_visible_keys <= envelope.max_visible_keys,
            op, "invalid execution envelope");
    require(envelope.max_visible_keys <= kCausalAttentionMaximumVisibleKeys, op,
            "the visible-key envelope exceeds the causal limit");
    require(static_cast<std::uint64_t>(envelope.max_visible_keys) <=
                static_cast<std::uint64_t>(shape.logical_pages) * kPagedKVPageSize,
            op, "the visible-key envelope exceeds the logical page capacity");
    require(masked || envelope.max_visible_keys >= static_cast<std::uint32_t>(shape.width), op,
            "a dense call needs an envelope at least as wide as its query block");

    // The shared causal contract makes every input, the output, and each cache plane and table
    // pairwise non-overlapping. Both plane groups are listed, which also subsumes the weaker
    // identity-only rule that the two groups must not share storage: aliasing them would let
    // one device read the other's bytes and contribute the same visible key twice. The
    // workspace is private to this object and the stand-in planes are owned here, so the
    // caller-visible operands are what can alias.
    {
        constexpr std::size_t kMaximumOperands = 16;
        std::array<NamedOperand, kMaximumOperands> operands{};
        std::size_t count = 0;
        const auto add    = [&](const Tensor& tensor, const char* name) {
            if (tensor.data == nullptr) { return; }
            require(count < kMaximumOperands, op, "the operand list overflowed");
            operands[count++] = NamedOperand{&tensor, name};
        };
        add(q, "q");
        add(k, "k");
        add(v, "v");
        add(positions, "positions");
        add(valid_columns, "valid columns");
        add(table_rows, "table rows");
        add(out, "out");
        add(primary_cache.k_pages, "primary cache k pages");
        add(primary_cache.v_pages, "primary cache v pages");
        add(primary_cache.k_scale_pages, "primary cache k scale pages");
        add(primary_cache.v_scale_pages, "primary cache v scale pages");
        add(primary_cache.block_tables, "block tables");
        add(secondary_cache.k, "secondary cache k pages");
        add(secondary_cache.v, "secondary cache v pages");
        add(secondary_cache.k_scale, "secondary cache k scale pages");
        add(secondary_cache.v_scale, "secondary cache v scale pages");
        require_non_overlapping({operands.data(), count}, op);
    }

    // ---- Device binding and ordering. -----------------------------------------------
    // Every input is read through host staging, so the host must observe everything already
    // submitted for it. The caller's stream is this call's producer; the previous call's tail
    // is settled first so the single pinned buffer is never overwritten while the primary
    // stream may still be reading it.
    DeviceGuard guard(self.primary);
    require_bound(guard, op);
    // The upload reads its inputs back through host memory, which a captured stream cannot
    // express, so a capture attempt is refused here instead of failing with an opaque CUDA
    // error mid-capture.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    require(capture == cudaStreamCaptureStatusNone, op,
            "the two-device route stages operands through host memory and cannot be captured");
    if (self.primary_pending) {
        const cudaError_t settled = cudaEventSynchronize(self.primary_settled);
        if (settled != cudaSuccess) {
            self.poisoned = true;
            CUDA_CHECK(settled);
        }
        self.primary_pending = false;
    }
    CUDA_CHECK(cudaEventRecord(self.primary_inputs_consumed, stream));
    const cudaError_t inputs_ready = cudaEventSynchronize(self.primary_inputs_consumed);
    if (inputs_ready != cudaSuccess) {
        self.poisoned = true;
        CUDA_CHECK(inputs_ready);
    }

    // Both devices' plane groups are read in place, so every plane a shard will decode must
    // resolve to the device that shard runs on. The primary group is checked on the primary
    // device here; the secondary group is checked by the guard below, because a pointer's
    // attributes only resolve cleanly while its own device is current.
    require_device(q.data, self.primary, op, "q");
    require_device(k.data, self.primary, op, "k");
    require_device(v.data, self.primary, op, "v");
    require_device(positions.data, self.primary, op, "positions");
    require_device(table_rows.data, self.primary, op, "table rows");
    require_device(out.data, self.primary, op, "out");
    require_device(primary_cache.k_pages.data, self.primary, op, "primary cache k pages");
    require_device(primary_cache.v_pages.data, self.primary, op, "primary cache v pages");
    require_device(primary_cache.k_scale_pages.data, self.primary, op,
                   "primary cache k scale pages");
    require_device(primary_cache.v_scale_pages.data, self.primary, op,
                   "primary cache v scale pages");
    require_device(primary_cache.block_tables.data, self.primary, op, "block tables");
    if (masked) { require_device(valid_columns.data, self.primary, op, "valid columns"); }
    {
        DeviceGuard secondary_guard(self.secondary);
        require_bound(secondary_guard, op);
        require_device(secondary_cache.k.data, self.secondary, op, "secondary cache k pages");
        require_device(secondary_cache.v.data, self.secondary, op, "secondary cache v pages");
        require_device(secondary_cache.k_scale.data, self.secondary, op,
                       "secondary cache k scales");
        require_device(secondary_cache.v_scale.data, self.secondary, op,
                       "secondary cache v scales");
    }

    // ---- Upload through pinned host staging. ----------------------------------------
    // Synchronous device-to-host copies are ordered against the legacy default stream as well,
    // so a producer that used it is covered too. Only one query block and small metadata move.
    auto* const stage  = static_cast<std::uint8_t*>(self.staging->data());
    std::size_t cursor = 0;
    const std::size_t query_bytes =
        checked_size_product({static_cast<std::size_t>(kHeadDim), static_cast<std::size_t>(q_heads),
                              static_cast<std::size_t>(shape.width),
                              static_cast<std::size_t>(shape.batch), sizeof(std::uint16_t)});
    const std::size_t kv_bytes = checked_size_product(
        {static_cast<std::size_t>(kHeadDim), static_cast<std::size_t>(kv_heads),
         static_cast<std::size_t>(shape.width), static_cast<std::size_t>(shape.batch),
         sizeof(std::uint16_t)});
    const std::size_t position_bytes =
        checked_size_product({static_cast<std::size_t>(shape.width),
                              static_cast<std::size_t>(shape.batch), sizeof(std::int32_t)});
    const std::size_t row_bytes =
        checked_size_product({static_cast<std::size_t>(shape.batch), sizeof(std::int32_t)});
    const std::size_t table_bytes =
        checked_size_product({static_cast<std::size_t>(shape.logical_pages),
                              static_cast<std::size_t>(shape.table_rows), sizeof(std::int32_t)});
    const auto upload = [&](const void* source, std::size_t bytes) {
        if (bytes == 0) { return; }
        require(cursor <= self.staging_bytes, op, "host staging offset exceeds its capacity");
        const std::size_t reserved = align_up(bytes, kAlignment);
        require(reserved <= self.staging_bytes - cursor, op,
                "host staging upload exceeds its startup capacity");
        CUDA_CHECK(cudaMemcpy(stage + cursor, source, bytes, cudaMemcpyDeviceToHost));
        cursor = checked_size_add(cursor, reserved);
    };
    upload(q.data, query_bytes);
    upload(k.data, kv_bytes);
    upload(v.data, kv_bytes);
    const std::size_t begin_positions = cursor;
    upload(positions.data, position_bytes);
    const std::size_t begin_valid = cursor;
    if (masked) { upload(valid_columns.data, row_bytes); }
    const std::size_t begin_rows = cursor;
    upload(table_rows.data, row_bytes);
    const std::size_t begin_table = cursor;
    upload(primary_cache.block_tables.data, table_bytes);

    const auto* const host_positions =
        reinterpret_cast<const std::int32_t*>(stage + begin_positions);
    const auto* const host_valid =
        masked ? reinterpret_cast<const std::int32_t*>(stage + begin_valid) : nullptr;
    const auto* const host_rows  = reinterpret_cast<const std::int32_t*>(stage + begin_rows);
    const auto* const host_table = reinterpret_cast<const std::int32_t*>(stage + begin_table);
    if (masked) {
        for (std::int32_t row = 0; row < shape.batch; ++row) {
            require(host_valid[row] >= 0 && host_valid[row] <= shape.width, op,
                    "a live-column count is outside [0,width]");
        }
    }
    validate_live_table(host_table, host_rows, host_positions, host_valid, shape, envelope, op);

    // A device without its own pages still receives a structurally valid one-page plane group,
    // so the shard launcher produces the neutral partial on the normal path.
    const PagedKVStagedPlanes primary_planes =
        primary_planes_present
            ? PagedKVStagedPlanes{primary_cache.k_pages, primary_cache.v_pages,
                                  primary_cache.k_scale_pages, primary_cache.v_scale_pages}
            : make_spare_planes(self.primary_spare_plane.p, kv_heads);
    const PagedKVStagedPlanes secondary_planes =
        shape.secondary_pages > 0 ? secondary_cache
                                  : make_spare_planes(self.secondary_spare_plane.p, kv_heads);

    Tensor secondary_tables(self.secondary_table.p, DType::I32,
                            {shape.logical_pages, shape.table_rows});

    // ---- Submit the secondary shard first so both devices compute concurrently. -----
    // It depends only on the uploaded metadata, never on the primary shard.
    //
    // From here the secondary stream owns the pinned staging buffer and the secondary arena.
    // Anything thrown before the normal tail settle below would leave that work live while the
    // next call overwrites both, so the guard reproduces the settle on the unwind path too.
    Impl::UnwindGuard unwind(&self, stream);
    unwind.arm();
    {
        DeviceGuard secondary_guard(self.secondary);
        require_bound(secondary_guard, op);
        std::size_t offset = 0;
        const auto down    = [&](const DeviceBuffer& destination, std::size_t bytes) {
            if (bytes == 0) { return; }
            require(offset <= self.upload_bytes, op, "host staging offset exceeds its capacity");
            const std::size_t reserved = align_up(bytes, kAlignment);
            require(reserved <= self.upload_bytes - offset && bytes <= destination.bytes, op,
                    "secondary upload exceeds its startup buffer capacity");
            CUDA_CHECK(cudaMemcpyAsync(destination.p, stage + offset, bytes, cudaMemcpyHostToDevice,
                                       self.secondary_stream));
            offset = checked_size_add(offset, reserved);
        };
        down(self.secondary_query, query_bytes);
        down(self.secondary_key, kv_bytes);
        down(self.secondary_value, kv_bytes);
        down(self.secondary_positions, position_bytes);
        if (masked) { down(self.secondary_valid, row_bytes); }
        down(self.secondary_rows, row_bytes);
        down(self.secondary_table, table_bytes);

        const Tensor secondary_valid =
            masked ? Tensor(self.secondary_valid.p, DType::I32, {shape.batch}) : Tensor{};
        const Tensor secondary_q(self.secondary_query.p, DType::BF16,
                                 {kHeadDim, q_heads, shape.width, shape.batch});
        const Tensor secondary_k(self.secondary_key.p, DType::BF16,
                                 {kHeadDim, kv_heads, shape.width, shape.batch});
        const Tensor secondary_v(self.secondary_value.p, DType::BF16,
                                 {kHeadDim, kv_heads, shape.width, shape.batch});
        const Tensor secondary_positions(self.secondary_positions.p, DType::I32,
                                         {shape.width, shape.batch});
        const Tensor secondary_rows(self.secondary_rows.p, DType::I32, {shape.batch});

        DeviceArena secondary_workspace(
            DeviceSpan{self.secondary_workspace.p, self.secondary_workspace.bytes});
        secondary_workspace.reset();
        Tensor secondary_partial =
            partial_view(self.secondary_partial, q_heads, shape.width, shape.batch);
        require(secondary_partial.bytes() <= self.secondary_partial.bytes &&
                    secondary_partial.bytes() <= self.staging_bytes,
                op, "the secondary partial exceeds its preallocated buffer capacity");
        detail::fp8_attention_shard(secondary_q, secondary_k, secondary_v, secondary_positions,
                                    secondary_valid, secondary_rows, scale,
                                    make_shard_view(secondary_planes, secondary_tables, kv_heads),
                                    envelope, shape.primary_pages, shape.total_pages,
                                    secondary_workspace, secondary_partial, self.secondary_stream);
        CUDA_CHECK(cudaMemcpyAsync(stage, self.secondary_partial.p, secondary_partial.bytes(),
                                   cudaMemcpyDeviceToHost, self.secondary_stream));
        CUDA_CHECK(cudaEventRecord(self.secondary_partial_ready, self.secondary_stream));
    }

    DeviceArena primary_workspace(
        DeviceSpan{self.primary_workspace.p, self.primary_workspace.bytes});
    primary_workspace.reset();
    Tensor primary_partial = partial_view(self.primary_partial, q_heads, shape.width, shape.batch);
    require(primary_partial.bytes() <= self.primary_partial.bytes, op,
            "the primary partial exceeds its preallocated buffer capacity");
    detail::fp8_attention_shard(
        q, k, v, positions, valid_columns, table_rows, scale,
        make_shard_view(primary_planes, primary_cache.block_tables, kv_heads), envelope, 0,
        shape.primary_pages, primary_workspace, primary_partial, stream);

    // ---- Bring the compact remote partial back and merge on the primary. ------------
    CUDA_CHECK(cudaStreamWaitEvent(stream, self.secondary_partial_ready, 0));
    Tensor remote_partial =
        partial_view(self.primary_remote_partial, q_heads, shape.width, shape.batch);
    require(remote_partial.bytes() <= self.primary_remote_partial.bytes &&
                remote_partial.bytes() <= self.staging_bytes,
            op, "the remote partial exceeds its preallocated buffer capacity");
    CUDA_CHECK(cudaMemcpyAsync(self.primary_remote_partial.p, stage, remote_partial.bytes(),
                               cudaMemcpyHostToDevice, stream));
    detail::fp8_attention_merge(primary_partial, remote_partial, out, stream);
    CUDA_CHECK(cudaEventRecord(self.primary_settled, stream));
    self.primary_pending = true;

    // The secondary planes are mutated by the shard, which appends the new K/V rows it owns, so
    // its stream is settled before returning: the caller may inspect or reuse those planes at
    // once.
    {
        DeviceGuard secondary_guard(self.secondary);
        require_bound(secondary_guard, op);
        const cudaError_t settled = cudaStreamSynchronize(self.secondary_stream);
        if (settled != cudaSuccess) { self.poisoned = true; }
        CUDA_CHECK(settled);
    }
    // Both sides are settled and the pinned buffer is free, so an unwind from here on has
    // nothing left to protect.
    unwind.disarm();
}

std::size_t DistributedAttention::primary_bytes() const noexcept { return impl_->primary_bytes; }

std::size_t DistributedAttention::secondary_bytes() const noexcept {
    return impl_->secondary_bytes;
}

std::size_t DistributedAttention::host_staging_bytes() const noexcept {
    return impl_->staging_bytes;
}

} // namespace ninfer::ops
