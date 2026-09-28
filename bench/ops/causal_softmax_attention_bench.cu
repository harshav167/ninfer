// Public-Op benchmark for causal cache Softmax Attention.

// The ordinary entries exercise public append-and-attend and cached-only calls. The distributed
// entry compares context-sharded and complete-remote attention through
// DistributedAttention::execute. Decode, small-T, prompt, split-KV, and kernel selection remain
// private production details.

#include "ninfer/ops/distributed_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/kv_cache_append.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim   = 256;
constexpr float kScale            = 0.0625F;
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;


enum class Entry : std::uint8_t { Append, Cached, Both, Distributed };
enum class GeometryChoice : std::uint8_t { H24Kv4, H16Kv2, All };
enum class KvChoice : std::uint8_t { Bf16, Int8, Fp8, Nvfp4, K8V4, All };
enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };
enum class PageMapping : std::uint8_t { Identity, Fragmented };
// Two-device placement candidates compared through the same DistributedAttention Op.
// Shard: split the visible key range by physical page between primary and secondary.
// Layer: entire attention for the selected call reads only secondary-resident pages,
//        with a nominal primary plane that the block table still spans (contract: every
//        visible key exactly once, empty contributions numerically neutral).
enum class Placement : std::uint8_t { Shard, Layer, Both };

struct Geometry {
    const char* name;
    std::int32_t query_heads;
    std::int32_t kv_heads;
};

constexpr Geometry kH24Kv4{"d256-h24-kv4", 24, 4};
constexpr Geometry kH16Kv2{"d256-h16-kv2", 16, 2};

struct Options {
    Entry entry             = Entry::Both;
    GeometryChoice geometry = GeometryChoice::All;
    KvChoice kv             = KvChoice::All;
    Execution execution     = Execution::Graph;
    bool execution_set      = false;
    CacheMode cache         = CacheMode::Cold;
    bool graph_calls_set    = false;
    bool placement_set      = false;
    bool contexts_set       = false;
    PageMapping mapping     = PageMapping::Identity;
    Placement placement     = Placement::Both;
    // Two-device entry. The secondary device is explicit; the primary is the device the
    // benchmark runs on. Residency is a page split: --primary-pages is an exact per-row
    // logical page count, --primary-fraction converts to one. Layer placement ignores
    // both and keeps every visible page on the secondary plane.
    // Present iff one of the two-device residency flags was supplied, so the ordinary
    // entries can reject a misplaced placement option instead of measuring something else.
    bool distributed_option_set = false;
    bool fraction_set           = false;
    int secondary_device        = 1;
    double primary_fraction     = 0.5;
    std::int32_t primary_pages  = -1;
    std::vector<std::int32_t> batches{1};
    std::vector<std::int32_t> tokens{1, 2, 4, 6, 8, 12, 16, 1024};
    std::vector<std::int32_t> contexts{0, 128, 2048, 8192};
    std::vector<std::int32_t> row_contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    int graph_calls = 1;
    int warmup      = 5;
    int repeat      = 30;
    bool profile    = false;
    std::string csv_out;
};

struct DeviceMemorySnapshot {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
};

struct Result {
    Entry entry;
    Geometry geometry;
    KvCacheStorage storage;
    Execution execution;
    CacheState cache;
    PageMapping mapping;
    std::int32_t batch;
    std::int32_t tokens;
    std::string row_contexts;
    std::string valid_columns;
    std::string table_rows;
    std::size_t workspace_bytes;
    double logical_bytes;
    double physical_bytes;
    double logical_cache_bytes;
    double key_vector_bytes;
    double value_vector_bytes;
    double physical_cache_bytes;
    double qk_flops;
    double pv_flops;
    double unique_kv_bytes;
    bench::ColdTiming timing;
    std::size_t graph_nodes = 0, workspace_peak = 0;
    int graph_calls = 1;
    // Wall timing is host-observed through complete remote settlement. `stream_timing` is only
    // the primary CUDA-event span, not host wall time or an independently measured remote stage.
    Placement placement              = Placement::Shard;
    int secondary_device             = -1;
    int primary_device               = -1;
    std::int32_t logical_pages       = 0;
    std::int32_t primary_plane_pages = 0;
    std::int32_t remote_plane_pages  = 0;
    std::int32_t appended_keys       = 0;
    std::size_t op_primary_bytes     = 0;
    std::size_t op_secondary_bytes   = 0;
    double primary_plane_bytes       = 0.0;
    double remote_plane_bytes        = 0.0;
    double primary_unique_bytes      = 0.0;
    double remote_unique_bytes       = 0.0;
    // Appended cache keys (rows) each plane holds. A page belongs to one plane, so the
    // two counts sum to the batch's total appended keys with no double counting. This is
    // the same "unique KV" convention as the ordinary entry's unique_kv_bytes.
    std::int64_t primary_resident_keys = 0;
    std::int64_t remote_resident_keys  = 0;
    std::string residency;
    std::size_t primary_input_bytes          = 0;
    std::size_t op_host_staging_bytes        = 0;
    std::size_t primary_setup_device_bytes   = 0;
    std::size_t secondary_setup_device_bytes = 0;
    DeviceMemorySnapshot primary_memory{};
    DeviceMemorySnapshot secondary_memory{};
    std::size_t secondary_flush_bytes = 0;
    bench::ColdTiming stream_timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(
        stderr,
        "error: %s\n"
        "usage: ninfer_causal_softmax_attention_bench "
        "[--entry append|cached|both|distributed] "
        "[--geometry d256-h24-kv4|d256-h16-kv2|all] "
        "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|all] [--batch B,...] [--tokens W,...] "
        "[--context L,...] [--row-contexts L0,...] [--valid-columns V0,...] "
        "[--table-rows R0,...] "
        "[--execution eager|graph|both] [--cache cold|warm|both] "
        "[--mapping identity|fragmented] "
        "[--warmup N] [--repeat N] [--graph-calls N] [--profile] [--csv-out PATH]\n"
        "\n"
        "distributed entry (--entry distributed) options:\n"
        "  --placement shard|layer|both  compare page-sharded attention with complete-remote\n"
        "                                attention; both candidates use the same public Op\n"
        "  --secondary-device N          CUDA device for remote FP8 KV (default 1; must differ\n"
        "                                from the benchmark's primary device)\n"
        "  --primary-pages P             exact per-row logical pages resident on primary (shard)\n"
        "  --primary-fraction F          primary share of each row's pages if pages is absent,\n"
        "                                in [0,1] (default 0.5)\n"
        "\n"
        "Distributed mode requires one geometry, dtype, B, W, and either one --context or a\n"
        "--row-contexts profile. Inputs and logical FP8 cache values are identical between\n"
        "placements; each cache is independently initialized through the public append codec.\n"
        "The public DistributedAttention::execute call is eager-only because remote work uses\n"
        "pinned-host staging and is not graph capturable. --execution eager is allowed;\n"
        "graph/both and any explicit --graph-calls are rejected. No peer-access assumption\n"
        "is made.\n"
        "Each wall sample times one complete call through remote completion; primary CUDA\n"
        "event timing is reported separately and is not host wall time or a remote-stage\n"
        "measurement. Cold samples evict L2 on both devices before timing; warm samples do\n"
        "not. Resource output includes cache planes, call inputs/outputs, Op-reported device\n"
        "resources, L2-flush buffers, pinned-host staging, and post-setup CUDA free memory.\n"
        "Setup totals count persistent benchmark-owned device buffers; temporary codec population\n"
        "buffers are freed before the post-setup snapshots. Free-memory snapshots are device-wide\n"
        "and reflect other GPU users.\n"
        "It does not claim combined bytes/second or token throughput. Layer placement uses\n"
        "zero primary-resident pages for this call; this is not a full transformer layer.\n"
        "\n"
        "Examples (bounded single-call workloads):\n"
        "  ninfer_causal_softmax_attention_bench --entry distributed --placement both \\\n"
        "    --geometry d256-h24-kv4 --kv-dtype fp8 --batch 1 --tokens 1 --context 8192 \\\n"
        "    --primary-fraction 0.5 --cache both --warmup 5 --repeat 30 --csv-out "
        "/tmp/near-kv.csv\n"
        "  ninfer_causal_softmax_attention_bench --entry distributed --placement both \\\n"
        "    --geometry d256-h16-kv2 --kv-dtype fp8 --batch 2 --tokens 4 \\\n"
        "    --row-contexts 4096,8192 --primary-pages 16 --cache cold --warmup 3 --repeat 20\n",
        message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::int32_t> parse_list(const char* text, std::int32_t minimum, std::int32_t maximum,
                                     const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_i32(item, minimum, maximum, flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--entry") {
            const std::string_view value(next("--entry requires a value"));
            if (value == "append")
                options.entry = Entry::Append;
            else if (value == "cached")
                options.entry = Entry::Cached;
            else if (value == "both")
                options.entry = Entry::Both;
            else if (value == "distributed")
                options.entry = Entry::Distributed;
            else
                usage("--entry expects append, cached, both, or distributed");
        } else if (argument == "--placement") {
            const std::string_view value(next("--placement requires a value"));
            if (value == "shard")
                options.placement = Placement::Shard;
            else if (value == "layer")
                options.placement = Placement::Layer;
            else if (value == "both")
                options.placement = Placement::Both;
            else
                usage("--placement expects shard, layer, or both");
            options.placement_set = true;
        } else if (argument == "--secondary-device") {
            options.secondary_device =
                parse_i32(next("--secondary-device requires a value"), 0, 64, "--secondary-device");
            options.distributed_option_set = true;
        } else if (argument == "--primary-pages") {
            options.primary_pages =
                parse_i32(next("--primary-pages requires a value"), 0, 262144, "--primary-pages");
            options.distributed_option_set = true;
        } else if (argument == "--primary-fraction") {
            const std::string_view text(next("--primary-fraction requires a value"));
            const std::string value(text);
            errno               = 0;
            char* end           = nullptr;
            const double parsed = std::strtod(value.c_str(), &end);
            if (errno != 0 || end == value.c_str() || *end != '\0' || !std::isfinite(parsed) ||
                parsed < 0.0 || parsed > 1.0) {
                usage("--primary-fraction expects a value in [0,1]");
            }
            options.primary_fraction       = parsed;
            options.fraction_set           = true;
            options.distributed_option_set = true;
        } else if (argument == "--geometry") {
            const std::string_view value(next("--geometry requires a value"));
            if (value == "d256-h24-kv4")
                options.geometry = GeometryChoice::H24Kv4;
            else if (value == "d256-h16-kv2")
                options.geometry = GeometryChoice::H16Kv2;
            else if (value == "all")
                options.geometry = GeometryChoice::All;
            else
                usage("--geometry expects d256-h24-kv4, d256-h16-kv2, or all");
        } else if (argument == "--kv-dtype") {
            const std::string_view value(next("--kv-dtype requires a value"));
            if (value == "bf16")
                options.kv = KvChoice::Bf16;
            else if (value == "int8")
                options.kv = KvChoice::Int8;
            else if (value == "fp8")
                options.kv = KvChoice::Fp8;
            else if (value == "nvfp4")
                options.kv = KvChoice::Nvfp4;
            else if (value == "k8v4")
                options.kv = KvChoice::K8V4;
            else if (value == "all")
                options.kv = KvChoice::All;
            else
                usage("--kv-dtype expects bf16, int8, fp8, nvfp4, k8v4, or all");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), 1, 262144, "--tokens");
        } else if (argument == "--batch") {
            options.batches = parse_list(next("--batch requires a value"), 1, 8, "--batch");
        } else if (argument == "--context") {
            options.contexts =
                parse_list(next("--context requires a value"), 0, 262144, "--context");
            options.contexts_set = true;
        } else if (argument == "--row-contexts") {
            options.row_contexts =
                parse_list(next("--row-contexts requires a value"), 0, 262144, "--row-contexts");
        } else if (argument == "--valid-columns") {
            options.valid_columns =
                parse_list(next("--valid-columns requires a value"), 0, 16, "--valid-columns");
        } else if (argument == "--table-rows") {
            options.table_rows =
                parse_list(next("--table-rows requires a value"), 0, 7, "--table-rows");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else if (value == "both")
                options.execution = Execution::Both;
            else
                usage("--execution expects eager, graph, or both");
            options.execution_set = true;
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else if (value == "both")
                options.cache = CacheMode::Both;
            else
                usage("--cache expects cold, warm, or both");
        } else if (argument == "--mapping") {
            const std::string_view value(next("--mapping requires a value"));
            if (value == "identity")
                options.mapping = PageMapping::Identity;
            else if (value == "fragmented")
                options.mapping = PageMapping::Fragmented;
            else
                usage("--mapping expects identity or fragmented");
        } else if (argument == "--graph-calls") {
            options.graph_calls =
                parse_i32(next("--graph-calls requires a value"), 1, 128, "--graph-calls");
            options.graph_calls_set = true;
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out requires a path");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    // The two-device entry owns its own --graph-calls rule below, because its samples are
    // host-observed complete calls rather than per-call graph spans.
    if (options.graph_calls > 1 && options.entry != Entry::Distributed &&
        (options.cache != CacheMode::Warm || options.execution != Execution::Graph))
        usage("repeated graphs require --cache warm --execution graph");
    for (const std::int32_t tokens : options.tokens) {
        for (const std::int32_t context : options.contexts) {
            if (context > std::numeric_limits<std::int32_t>::max() - tokens) {
                usage("context + tokens exceeds int32");
            }
        }
    }
    const bool exact_profile = !options.row_contexts.empty() || !options.valid_columns.empty() ||
                               !options.table_rows.empty();
    if (options.primary_pages >= 0 && options.fraction_set) {
        usage("--primary-pages and --primary-fraction are mutually exclusive");
    }
    // Layer placement means zero referenced primary pages by definition, so an explicit
    // residency request on it would silently disagree with what the run measures.
    if (options.placement == Placement::Layer &&
        (options.primary_pages >= 0 || options.fraction_set)) {
        usage("--placement layer fixes zero primary pages and takes no residency flag");
    }
    // The distributed entry compares placement candidates at real model geometry, so it
    // takes one geometry, one cache storage, one B and one W, and accepts row context
    // profiles for unequal-depth rows. It is dense-only, so it rejects column/table
    // profile knobs the two-device contract does not expose.
    if (options.entry == Entry::Distributed) {
        if (options.geometry == GeometryChoice::All || options.kv == KvChoice::All ||
            options.batches.size() != 1 || options.tokens.size() != 1) {
            usage("--entry distributed requires one --geometry, one --kv-dtype, one B, and one W");
        }
        if (options.kv != KvChoice::Fp8) {
            usage("--entry distributed requires the FP8 row-scaled --kv-dtype fp8 cache");
        }
        if (options.row_contexts.empty()) {
            if (!options.contexts_set || options.contexts.size() != 1) {
                usage("--entry distributed requires one --context or a --row-contexts profile");
            }
        } else if (options.contexts_set) {
            usage("--context and --row-contexts are mutually exclusive in distributed mode");
        }
        if (!options.valid_columns.empty() || !options.table_rows.empty()) {
            usage("--entry distributed does not accept --valid-columns or --table-rows");
        }
        if (options.execution_set && options.execution != Execution::Eager) {
            usage(
                "--entry distributed only accepts --execution eager; graph capture is unsupported");
        }
        if (options.graph_calls_set) {
            usage("--entry distributed is not graph capturable and does not accept --graph-calls");
        }
        if (options.profile) { usage("--entry distributed does not accept --profile"); }
        int primary_device = 0;
        CUDA_CHECK(cudaGetDevice(&primary_device));
        if (options.secondary_device == primary_device) {
            usage("--secondary-device must differ from the benchmark's primary device");
        }
        if (options.placement == Placement::Both &&
            ((options.primary_pages == 0) ||
             (options.fraction_set && options.primary_fraction == 0.0))) {
            usage("zero primary residency duplicates layer placement; select --placement layer");
        }
    } else if (options.distributed_option_set || options.placement_set) {
        usage("distributed placement/residency options require --entry distributed");
    }
    if (exact_profile && options.entry != Entry::Append && options.entry != Entry::Distributed) {
        usage("exact row profiles require --entry append");
    }
    if (exact_profile && (options.batches.size() != 1 || options.tokens.size() != 1)) {
        usage("exact row profiles require one batch size and one W");
    }
    if (exact_profile) {
        const auto batch = static_cast<std::size_t>(options.batches.front());
        if ((!options.row_contexts.empty() && options.row_contexts.size() != batch) ||
            (!options.valid_columns.empty() && options.valid_columns.size() != batch) ||
            (!options.table_rows.empty() && options.table_rows.size() != batch)) {
            usage("exact row profile length must equal B");
        }
        for (const std::int32_t valid : options.valid_columns) {
            if (valid > options.tokens.front()) { usage("valid column count exceeds W"); }
        }
        if (!options.table_rows.empty()) {
            std::vector<bool> seen(batch, false);
            for (const std::int32_t row : options.table_rows) {
                if (row < 0 || static_cast<std::size_t>(row) >= batch || seen[row]) {
                    usage("table rows must be a permutation of [0,B)");
                }
                seen[row] = true;
            }
        }
    }
    for (const std::int32_t batch : options.batches) {
        if (batch > 1 && std::any_of(options.tokens.begin(), options.tokens.end(),
                                     [](std::int32_t width) { return width > 16; })) {
            usage("B>1 only supports W<=16");
        }
    }
    if (options.entry == Entry::Cached &&
        std::any_of(options.batches.begin(), options.batches.end(),
                    [](std::int32_t batch) { return batch > 1; })) {
        usage("cached entry is B=1 only");
    }
    if (options.profile &&
        (options.entry == Entry::Both || options.geometry == GeometryChoice::All ||
         options.kv == KvChoice::All || options.batches.size() != 1 || options.tokens.size() != 1 ||
         (options.row_contexts.empty() && options.contexts.size() != 1) ||
         options.execution == Execution::Both || options.cache == CacheMode::Both)) {
        usage("--profile requires one entry, geometry, dtype, B, W, row profile, execution, and "
              "cache");
    }
    return options;
}

std::int32_t align_context(std::int32_t visible) { return ((visible + 127) / 128) * 128; }

std::size_t data_plane_bytes(const Geometry& geometry, const PagedKVVectorLayout& layout,
                             std::int32_t physical_pages) {
    return static_cast<std::size_t>(layout.data_leading_extent) * geometry.kv_heads *
           kPagedKVPageSize * physical_pages * dtype_size(layout.data_dtype);
}

std::size_t scale_plane_bytes(const Geometry& geometry, const PagedKVVectorLayout& layout,
                              std::int32_t physical_pages) {
    return static_cast<std::size_t>(layout.scale_leading_extent) * geometry.kv_heads *
           kPagedKVPageSize * physical_pages * dtype_size(layout.scale_dtype);
}

PagedKVLayerView make_cache_view(DeviceBuffer& k, DeviceBuffer& v, DeviceBuffer& k_scale,
                                 DeviceBuffer& v_scale, DeviceBuffer& block_table,
                                 const Geometry& geometry, KvCacheStorage storage,
                                 std::int32_t padded, std::int32_t physical_pages) {
    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kHeadDim);
    const std::int32_t logical_pages  = padded / kPagedKVPageSize;
    return {
        .k_pages = Tensor(
            k.p, layout.key.data_dtype,
            {layout.key.data_leading_extent, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .v_pages = Tensor(v.p, layout.value.data_dtype,
                          {layout.value.data_leading_extent, kPagedKVPageSize, geometry.kv_heads,
                           physical_pages}),
        .k_scale_pages = layout.key.has_scale()
                             ? Tensor(k_scale.p, layout.key.scale_dtype,
                                      {layout.key.scale_leading_extent, kPagedKVPageSize,
                                       geometry.kv_heads, physical_pages})
                             : Tensor(),
        .v_scale_pages = layout.value.has_scale()
                             ? Tensor(v_scale.p, layout.value.scale_dtype,
                                      {layout.value.scale_leading_extent, kPagedKVPageSize,
                                       geometry.kv_heads, physical_pages})
                             : Tensor(),
        .block_table   = Tensor(block_table.p, DType::I32, {logical_pages}),
        .head_dim      = kHeadDim,
        .num_kv_heads  = geometry.kv_heads,
        .storage       = storage,
    };
}

PagedKVBatchLayerView make_batch_cache_view(const PagedKVLayerView& cache) {
    return single_row_paged_kv_batch_view(cache);
}

PagedKVBatchLayerView make_batch_cache_view(DeviceBuffer& k, DeviceBuffer& v, DeviceBuffer& k_scale,
                                            DeviceBuffer& v_scale, DeviceBuffer& block_tables,
                                            const Geometry& geometry, KvCacheStorage storage,
                                            std::int32_t padded, std::int32_t physical_pages,
                                            std::int32_t table_rows) {
    const PagedKVLayerView direct = make_cache_view(k, v, k_scale, v_scale, block_tables, geometry,
                                                    storage, padded, physical_pages);
    PagedKVBatchLayerView result  = make_batch_cache_view(direct);
    result.block_tables =
        Tensor(block_tables.p, DType::I32, {padded / kPagedKVPageSize, table_rows});
    return result;
}

std::size_t workspace_capacity(const Geometry& geometry, KvCacheStorage storage,
                               std::int32_t tokens, std::int32_t batch, std::int32_t visible) {
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(visible),
                                                         static_cast<std::uint32_t>(visible)};
    return ops::causal_softmax_attention_workspace_capacity_bytes(
        {kHeadDim, geometry.query_heads, geometry.kv_heads}, storage, envelope, batch, tokens,
        tokens);
}

std::int32_t profile_visible(std::span<const std::int32_t> contexts,
                             std::span<const std::int32_t> valid_columns) {
    std::int32_t visible = 1;
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        visible = std::max(visible, contexts[row] + valid_columns[row]);
    }
    return visible;
}

__global__ void initialize_values(__nv_bfloat16* data, std::size_t count, unsigned seed,
                                  float scale, std::size_t index_offset) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned value = static_cast<unsigned>(i + index_offset) + seed;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    data[i] = __float2bfloat16_rn((float(value >> 8) * (2.f / 16777216.f) - 1.f) * scale);
}

DeviceBuffer varied_values(std::size_t count, unsigned seed, float scale,
                           std::size_t index_offset = 0) {
    DeviceBuffer result(count * 2);
    initialize_values<<<(count + 255) / 256, 256>>>(static_cast<__nv_bfloat16*>(result.p), count,
                                                    seed, scale, index_offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

class Case {
public:
    Case(Geometry geometry, KvCacheStorage storage, std::int32_t tokens,
         std::span<const std::int32_t> contexts, std::span<const std::int32_t> valid_columns,
         std::span<const std::int32_t> table_rows, PageMapping mapping)
        : storage_layout_(paged_kv_storage_layout(storage, kHeadDim)),
          batch_(static_cast<std::int32_t>(contexts.size())),
          masked_(std::any_of(valid_columns.begin(), valid_columns.end(),
                              [tokens](std::int32_t valid) { return valid != tokens; })),
          visible_(profile_visible(contexts, valid_columns)), padded_(align_context(visible_)),
          mapping_(mapping), logical_pages_(padded_ / kPagedKVPageSize),
          physical_pages_(mapping == PageMapping::Identity ? batch_ * logical_pages_
                                                           : 2 * batch_ * logical_pages_ + 1),
          q_(varied_values(static_cast<std::size_t>(kHeadDim) * geometry.query_heads * tokens *
                               batch_,
                           101, .25f)),
          k_(varied_values(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens * batch_,
                           103, .25f)),
          v_(varied_values(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens * batch_,
                           107, 1.f)),
          positions_(static_cast<std::size_t>(tokens) * batch_ * sizeof(std::int32_t)),
          valid_columns_(static_cast<std::size_t>(batch_) * sizeof(std::int32_t)),
          table_rows_(static_cast<std::size_t>(batch_) * sizeof(std::int32_t)),
          cache_k_(
              bench::make_zeros(data_plane_bytes(geometry, storage_layout_.key, physical_pages_))),
          cache_v_(bench::make_zeros(
              data_plane_bytes(geometry, storage_layout_.value, physical_pages_))),
          cache_k_scale_(bench::make_zeros(
              storage_layout_.key.has_scale()
                  ? scale_plane_bytes(geometry, storage_layout_.key, physical_pages_)
                  : std::size_t{1})),
          cache_v_scale_(bench::make_zeros(
              storage_layout_.value.has_scale()
                  ? scale_plane_bytes(geometry, storage_layout_.value, physical_pages_)
                  : std::size_t{1})),
          block_table_(static_cast<std::size_t>(logical_pages_) * batch_ * sizeof(std::int32_t)),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * geometry.query_heads *
                                    tokens * batch_ * 2)),
          workspace_bytes_(workspace_capacity(geometry, storage, tokens, batch_, visible_)),
          workspace_(std::max<std::size_t>(workspace_bytes_, 1)),
          q_tensor_(q_.p, DType::BF16, {kHeadDim, geometry.query_heads, tokens, batch_}),
          k_tensor_(k_.p, DType::BF16, {kHeadDim, geometry.kv_heads, tokens, batch_}),
          v_tensor_(v_.p, DType::BF16, {kHeadDim, geometry.kv_heads, tokens, batch_}),
          positions_tensor_(positions_.p, DType::I32, {tokens, batch_}),
          valid_columns_tensor_(valid_columns_.p, DType::I32, {batch_}),
          table_rows_tensor_(table_rows_.p, DType::I32, {batch_}),
          output_tensor_(output_.p, DType::BF16, {kHeadDim, geometry.query_heads, tokens, batch_}),
          cache_view_(make_cache_view(cache_k_, cache_v_, cache_k_scale_, cache_v_scale_,
                                      block_table_, geometry, storage, padded_, physical_pages_)),
          batch_cache_view_(make_batch_cache_view(cache_k_, cache_v_, cache_k_scale_,
                                                  cache_v_scale_, block_table_, geometry, storage,
                                                  padded_, physical_pages_, batch_)),
          envelope_{static_cast<std::uint32_t>(visible_), static_cast<std::uint32_t>(visible_)} {
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens) * batch_, 0);
        for (std::int32_t row = 0; row < batch_; ++row) {
            const std::int32_t valid = valid_columns[static_cast<std::size_t>(row)];
            for (std::int32_t token = 0; token < valid; ++token) {
                host_positions[static_cast<std::size_t>(row) * tokens + token] =
                    contexts[static_cast<std::size_t>(row)] + token;
            }
            const std::int32_t padding_position =
                valid == 0 ? 0 : contexts[static_cast<std::size_t>(row)] + valid - 1;
            for (std::int32_t token = valid; token < tokens; ++token) {
                host_positions[static_cast<std::size_t>(row) * tokens + token] = padding_position;
            }
        }
        std::vector<std::int32_t> host_table(static_cast<std::size_t>(logical_pages_) * batch_);
        for (std::int32_t row = 0; row < batch_; ++row) {
            for (std::int32_t page = 0; page < logical_pages_; ++page) {
                const std::int32_t linear = row * logical_pages_ + page;
                host_table[static_cast<std::size_t>(row) * logical_pages_ + page] =
                    mapping_ == PageMapping::Identity ? linear : 2 * linear + 1;
            }
        }
        CUDA_CHECK(cudaMemcpy(positions_.p, host_positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(block_table_.p, host_table.data(), block_table_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(valid_columns_.p, valid_columns.data(), valid_columns_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(table_rows_.p, table_rows.data(), table_rows_.bytes,
                              cudaMemcpyHostToDevice));
        // Populate every represented cache row through the public codec, outside measurement.
        // A row view selects its own table while all physical planes remain shared.
        std::vector<std::int32_t> initial_positions(padded_);
        for (int pos = 0; pos < padded_; ++pos) initial_positions[pos] = pos;
        DeviceBuffer dp(std::size_t(padded_) * 4);
        CUDA_CHECK(cudaMemcpy(dp.p, initial_positions.data(), dp.bytes, cudaMemcpyHostToDevice));
        Tensor positions(dp.p, DType::I32, {padded_});
        for (int row = 0; row < batch_; ++row) {
            auto initial_k =
                varied_values(std::size_t(kHeadDim) * geometry.kv_heads * padded_, 211 + row, .25f);
            auto initial_v =
                varied_values(std::size_t(kHeadDim) * geometry.kv_heads * padded_, 311 + row, 1.f);
            Tensor k(initial_k.p, DType::BF16, {kHeadDim, geometry.kv_heads, padded_});
            Tensor v(initial_v.p, DType::BF16, {kHeadDim, geometry.kv_heads, padded_});
            auto row_cache = cache_view_;
            row_cache.block_table =
                Tensor(static_cast<std::int32_t*>(block_table_.p) + row * logical_pages_,
                       DType::I32, {logical_pages_});
            ops::kv_cache_append(k, v, positions, row_cache, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    void launch(Entry entry, cudaStream_t stream) {
        if (entry == Entry::Append) {
            const Tensor validity = masked_ ? valid_columns_tensor_ : Tensor{};
            ops::causal_softmax_attention(
                q_tensor_, k_tensor_, v_tensor_, positions_tensor_, validity, table_rows_tensor_,
                {kHeadDim, q_tensor_.ne[1], k_tensor_.ne[1]}, kScale, batch_cache_view_, envelope_,
                workspace_, output_tensor_, stream);
        } else {
            ops::causal_softmax_attention_cached(
                q_tensor_, positions_tensor_, {kHeadDim, q_tensor_.ne[1], cache_view_.num_kv_heads},
                kScale, cache_view_, envelope_, workspace_, output_tensor_, stream);
        }
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

    [[nodiscard]] std::size_t workspace_peak() const {
        if (workspace_.used() != 0 || workspace_.peak_used() > workspace_bytes_)
            throw std::runtime_error("attention workspace query mismatch");
        return workspace_.peak_used();
    }

private:
    PagedKVStorageLayout storage_layout_;
    std::int32_t batch_;
    bool masked_;
    std::int32_t visible_;
    std::int32_t padded_;
    PageMapping mapping_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer valid_columns_;
    DeviceBuffer table_rows_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer cache_k_scale_;
    DeviceBuffer cache_v_scale_;
    DeviceBuffer block_table_;
    DeviceBuffer output_;
    std::size_t workspace_bytes_;
    WorkspaceArena workspace_;
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    Tensor valid_columns_tensor_;
    Tensor table_rows_tensor_;
    Tensor output_tensor_;
    PagedKVLayerView cache_view_;
    PagedKVBatchLayerView batch_cache_view_;
    ops::CausalAttentionExecutionEnvelope envelope_;
};

// ---------------------------------------------------------------------------
// Two-device entry: complete causal attention through DistributedAttention.
//
// The fixture owns both cache planes. Global physical page IDs partition the
// namespace: [0, primary_plane_pages) is the primary plane and
// [primary_plane_pages, +remote_plane_pages) is the secondary plane. Each row
// keeps its leading logical pages in the primary plane and the remainder in the
// secondary plane, so a page's plane decides which device computes its
// attention contribution. The block table holds global IDs and is the only
// routing input the Op sees.
//
// Two placement candidates share this one interface:
//   shard  pages split per row by --primary-pages/--primary-fraction.
//   layer  no primary pages; every referenced global ID is secondary-resident, so the
//          primary contributes the neutral empty-shard result through the same Op.
// Every row is populated through the public append codec, so the Op decodes
// legitimately represented FP8 rows rather than uninitialized bytes. All plane
// population, allocation and synchronization happen in the constructor, outside
// every timed interval.
// ---------------------------------------------------------------------------

class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        if (previous_ != device) { CUDA_CHECK(cudaSetDevice(device)); }
    }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

DeviceMemorySnapshot device_memory_snapshot(int device) {
    const ScopedDevice target(device);
    DeviceMemorySnapshot snapshot;
    CUDA_CHECK(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes));
    return snapshot;
}

// One device's physical cache planes. Buffers are allocated on the device that is
// current when the object is built and must be released while it is current again.
struct CachePlanes {
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer k_scale;
    DeviceBuffer v_scale;
    std::int32_t pages = 0;
};

PagedKVStagedPlanes bind_planes(const CachePlanes& planes, const Geometry& geometry,
                                KvCacheStorage storage) {
    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kHeadDim);
    return {
        Tensor(planes.k.p, layout.key.data_dtype,
               {layout.key.data_leading_extent, kPagedKVPageSize, geometry.kv_heads, planes.pages}),
        Tensor(
            planes.v.p, layout.value.data_dtype,
            {layout.value.data_leading_extent, kPagedKVPageSize, geometry.kv_heads, planes.pages}),
        layout.key.has_scale() ? Tensor(planes.k_scale.p, layout.key.scale_dtype,
                                        {layout.key.scale_leading_extent, kPagedKVPageSize,
                                         geometry.kv_heads, planes.pages})
                               : Tensor(),
        layout.value.has_scale() ? Tensor(planes.v_scale.p, layout.value.scale_dtype,
                                          {layout.value.scale_leading_extent, kPagedKVPageSize,
                                           geometry.kv_heads, planes.pages})
                                 : Tensor(),
    };
}

CachePlanes allocate_planes(const Geometry& geometry, KvCacheStorage storage, std::int32_t pages) {
    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kHeadDim);
    CachePlanes planes;
    planes.pages   = pages;
    planes.k       = DeviceBuffer(data_plane_bytes(geometry, layout.key, pages));
    planes.v       = DeviceBuffer(data_plane_bytes(geometry, layout.value, pages));
    planes.k_scale = DeviceBuffer(
        layout.key.has_scale() ? scale_plane_bytes(geometry, layout.key, pages) : std::size_t{1});
    planes.v_scale =
        DeviceBuffer(layout.value.has_scale() ? scale_plane_bytes(geometry, layout.value, pages)
                                              : std::size_t{1});
    return planes;
}

// Plane-local page permutation for the fragmented mapping. The stride must be coprime with
// the plane's page count, otherwise the walk would revisit pages and alias distinct logical
// pages onto one physical page. Small planes have no useful stride other than 1, and 1 is
// coprime with every capacity, so the search always terminates with a valid stride.
std::int64_t fragment_stride(std::int32_t capacity) {
    if (capacity <= 2) { return 1; }
    const auto gcd = [](std::int64_t a, std::int64_t b) {
        while (b != 0) {
            const std::int64_t next = a % b;
            a                       = b;
            b                       = next;
        }
        return a;
    };
    const auto start = static_cast<std::int64_t>(capacity) / 2 + 1;
    for (std::int64_t attempt = 0; attempt < capacity; ++attempt) {
        const std::int64_t candidate = 1 + (start + attempt) % capacity;
        if (gcd(candidate, capacity) == 1) { return candidate; }
    }
    return 1;
}

class DistributedCase {
public:
    DistributedCase(const Geometry& geometry, KvCacheStorage storage, std::int32_t tokens,
                    std::vector<std::int32_t> contexts,
                    std::vector<std::int32_t> primary_pages_per_row, PageMapping mapping,
                    int secondary_device)
        : geometry_(geometry), storage_(storage), tokens_(tokens),
          batch_(static_cast<std::int32_t>(contexts.size())), contexts_(std::move(contexts)),
          mapping_(mapping), secondary_device_(secondary_device) {
        if (batch_ < 1 || batch_ > 8) {
            throw std::invalid_argument("distributed entry requires B in 1..8");
        }
        if (storage_ != KvCacheStorage::Fp8E4M3Row256) {
            throw std::invalid_argument("distributed entry requires the FP8 row-scaled cache");
        }
        if (geometry_.kv_heads != 4 && geometry_.kv_heads != 2) {
            throw std::invalid_argument("distributed entry requires the registered KV head count");
        }
        if (batch_ > 1 && tokens_ > 16) {
            throw std::invalid_argument("distributed entry requires W<=16 when B>1");
        }

        // One logical page capacity and one envelope span every row, exactly as the
        // production batch view presents them.
        std::int32_t max_total      = 0;
        std::int32_t appended_total = 0;
        for (const std::int32_t context : contexts_) {
            max_total = std::max(max_total, context + tokens_);
            appended_total += context + tokens_;
        }
        appended_keys_ = appended_total;
        logical_pages_ = align_context(max_total) / kPagedKVPageSize;
        if (logical_pages_ <= 0) {
            throw std::invalid_argument("distributed entry requires positive logical pages");
        }

        live_pages_.resize(static_cast<std::size_t>(batch_));
        primary_pages_.resize(static_cast<std::size_t>(batch_));
        if (!primary_pages_per_row.empty() &&
            static_cast<std::int32_t>(primary_pages_per_row.size()) != batch_) {
            throw std::invalid_argument("distributed residency list must have one entry per row");
        }
        for (std::int32_t row = 0; row < batch_; ++row) {
            const std::int32_t live =
                align_context(contexts_[static_cast<std::size_t>(row)] + tokens_) /
                kPagedKVPageSize;
            live_pages_[static_cast<std::size_t>(row)] = live;
            const std::int32_t primary = primary_pages_per_row.empty()
                                             ? 0
                                             : primary_pages_per_row[static_cast<std::size_t>(row)];
            if (primary < 0 || primary > live) {
                throw std::invalid_argument("distributed primary residency is outside row pages");
            }
            primary_pages_[static_cast<std::size_t>(row)] = primary;
        }

        // Split planes. A layer-placement call supplies zero residency for every row;
        // the primary plane then keeps one nominal page that no table entry uses.
        std::int64_t primary_pages_sum = 0;
        std::int64_t remote_pages_sum  = 0;
        for (std::int32_t row = 0; row < batch_; ++row) {
            primary_pages_sum += primary_pages_[static_cast<std::size_t>(row)];
            remote_pages_sum += live_pages_[static_cast<std::size_t>(row)] -
                                primary_pages_[static_cast<std::size_t>(row)];
        }
        // Zero primary pages is a supported configuration: the primary plane is absent and
        // every global ID belongs to the secondary plane. The local shard is then empty, and
        // the Op publishes its neutral contribution through the production path. That is
        // complete remote attention as it appears at one attention call.
        primary_plane_absent_ = primary_pages_sum == 0;
        primary_plane_pages_ =
            primary_plane_absent_ ? 0 : static_cast<std::int32_t>(primary_pages_sum);
        remote_plane_pages_ = static_cast<std::int32_t>(remote_pages_sum);
        // The two-device route needs pages somewhere; a request that leaves every page
        // primary is the ordinary all-local route, which this entry does not measure.
        if (remote_plane_pages_ <= 0) {
            throw std::invalid_argument(
                "distributed entry requires at least one remote page; use --entry append for the "
                "all-local route");
        }
        primary_offsets_.resize(static_cast<std::size_t>(batch_), 0);
        remote_offsets_.resize(static_cast<std::size_t>(batch_), 0);
        for (std::int32_t row = 1; row < batch_; ++row) {
            primary_offsets_[static_cast<std::size_t>(row)] =
                primary_offsets_[static_cast<std::size_t>(row - 1)] +
                primary_pages_[static_cast<std::size_t>(row - 1)];
            remote_offsets_[static_cast<std::size_t>(row)] =
                remote_offsets_[static_cast<std::size_t>(row - 1)] +
                (live_pages_[static_cast<std::size_t>(row - 1)] -
                 primary_pages_[static_cast<std::size_t>(row - 1)]);
        }

        q_ = varied_values(static_cast<std::size_t>(kHeadDim) * geometry_.query_heads * tokens_ *
                               batch_,
                           101, .25f);
        k_ = varied_values(
            static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads * tokens_ * batch_, 103, .25f);
        v_ = varied_values(
            static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads * tokens_ * batch_, 107, 1.f);
        positions_ =
            DeviceBuffer(static_cast<std::size_t>(tokens_) * batch_ * sizeof(std::int32_t));
        table_rows_ = DeviceBuffer(static_cast<std::size_t>(batch_) * sizeof(std::int32_t));
        global_table_ =
            DeviceBuffer(static_cast<std::size_t>(logical_pages_) * batch_ * sizeof(std::int32_t));
        output_ = bench::make_zeros(static_cast<std::size_t>(kHeadDim) * geometry_.query_heads *
                                    tokens_ * batch_ * 2);

        q_tensor_ = Tensor(q_.p, DType::BF16, {kHeadDim, geometry_.query_heads, tokens_, batch_});
        k_tensor_ = Tensor(k_.p, DType::BF16, {kHeadDim, geometry_.kv_heads, tokens_, batch_});
        v_tensor_ = Tensor(v_.p, DType::BF16, {kHeadDim, geometry_.kv_heads, tokens_, batch_});
        positions_tensor_  = Tensor(positions_.p, DType::I32, {tokens_, batch_});
        table_rows_tensor_ = Tensor(table_rows_.p, DType::I32, {batch_});
        output_tensor_ =
            Tensor(output_.p, DType::BF16, {kHeadDim, geometry_.query_heads, tokens_, batch_});
        // Envelope spans the longest row's live keys, matching the production batch view:
        // the Op's envelope is a launch/workspace promise over the batch maximum.
        envelope_ = {static_cast<std::uint32_t>(tokens_), static_cast<std::uint32_t>(max_total)};

        // Dense rows: every live column attends its populated history [0, position].
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens_) * batch_, 0);
        std::vector<std::int32_t> host_rows(static_cast<std::size_t>(batch_), 0);
        std::vector<std::int32_t> host_table(static_cast<std::size_t>(logical_pages_) * batch_, 0);
        for (std::int32_t row = 0; row < batch_; ++row) {
            const std::int32_t context = contexts_[static_cast<std::size_t>(row)];
            for (std::int32_t token = 0; token < tokens_; ++token) {
                host_positions[static_cast<std::size_t>(row) * tokens_ + token] = context + token;
            }
            host_rows[static_cast<std::size_t>(row)] = row;
            for (std::int32_t page = 0; page < logical_pages_; ++page) {
                // Pages past the row's live range are never referenced by a position,
                // but the contract requires a materialized entry: repeat the last page.
                const std::int32_t live = live_pages_[static_cast<std::size_t>(row)];
                host_table[static_cast<std::size_t>(row) * logical_pages_ + page] =
                    global_page_id(row, std::min(page, live - 1));
            }
        }
        CUDA_CHECK(cudaMemcpy(positions_.p, host_positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(table_rows_.p, host_rows.data(), table_rows_.bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(global_table_.p, host_table.data(), global_table_.bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaGetDevice(&primary_device_));
        if (!primary_plane_absent_) {
            primary_planes_ = allocate_planes(geometry_, storage_, primary_plane_pages_);
        }
        try {
            const ScopedDevice remote(secondary_device_);
            remote_planes_ = allocate_planes(geometry_, storage_, remote_plane_pages_);
        } catch (...) {
            release_planes();
            throw;
        }
        remote_staged_ = bind_planes(remote_planes_, geometry_, storage_);
        primary_view_  = batch_view();
        try {
            populate();
            op_.emplace(
                primary_device_, secondary_device_,
                ops::AttentionHeadGeometry{kHeadDim, geometry_.query_heads, geometry_.kv_heads},
                tokens_, batch_, batch_, logical_pages_);
        } catch (...) {
            op_.reset();
            release_planes();
            throw;
        }
    }

    ~DistributedCase() {
        op_.reset();
        release_planes();
    }

    DistributedCase(const DistributedCase&)            = delete;
    DistributedCase& operator=(const DistributedCase&) = delete;

    void execute(cudaStream_t stream) {
        op_->execute(q_tensor_, k_tensor_, v_tensor_, positions_tensor_, Tensor{},
                     table_rows_tensor_, kScale, primary_view_, remote_staged_, envelope_,
                     output_tensor_, stream);
    }

    [[nodiscard]] std::int32_t logical_pages() const noexcept { return logical_pages_; }

    [[nodiscard]] std::int32_t appended_keys() const noexcept { return appended_keys_; }

    [[nodiscard]] std::int32_t primary_plane_pages() const noexcept { return primary_plane_pages_; }

    [[nodiscard]] std::int32_t remote_plane_pages() const noexcept { return remote_plane_pages_; }

    [[nodiscard]] int secondary_device() const noexcept { return secondary_device_; }

    [[nodiscard]] int primary_device() const noexcept { return primary_device_; }

    [[nodiscard]] std::size_t op_primary_bytes() const noexcept { return op_->primary_bytes(); }

    [[nodiscard]] std::size_t op_secondary_bytes() const noexcept { return op_->secondary_bytes(); }

    [[nodiscard]] std::size_t op_host_staging_bytes() const noexcept {
        return op_->host_staging_bytes();
    }

    [[nodiscard]] std::size_t primary_plane_allocation_bytes() const noexcept {
        return plane_allocation_bytes(primary_planes_);
    }

    [[nodiscard]] std::size_t remote_plane_allocation_bytes() const noexcept {
        return plane_allocation_bytes(remote_planes_);
    }

    [[nodiscard]] std::size_t primary_input_bytes() const noexcept {
        return q_.bytes + k_.bytes + v_.bytes + positions_.bytes + table_rows_.bytes +
               global_table_.bytes + output_.bytes;
    }

    [[nodiscard]] double primary_plane_bytes() const noexcept {
        return plane_bytes(primary_planes_);
    }

    [[nodiscard]] double remote_plane_bytes() const noexcept { return plane_bytes(remote_planes_); }

    // Appended cache keys and unique cache bytes each plane owns. The split is by page, so
    // a row contributes whole pages to exactly one plane and every appended key is counted
    // once across the two totals.
    [[nodiscard]] std::int64_t primary_resident_keys() const noexcept {
        std::int64_t total = 0;
        for (std::int32_t row = 0; row < batch_; ++row) { total += primary_keys_in_row(row); }
        return total;
    }

    [[nodiscard]] std::int64_t remote_resident_keys() const noexcept {
        std::int64_t total = 0;
        for (std::int32_t row = 0; row < batch_; ++row) {
            total += row_appended_keys(row) - primary_keys_in_row(row);
        }
        return total;
    }

    // Per-row primary residency actually applied after clamping to each row's own pages.
    [[nodiscard]] std::string primary_residency_profile() const {
        std::string result;
        for (std::int32_t row = 0; row < batch_; ++row) {
            if (row != 0) { result.push_back(';'); }
            result += std::to_string(primary_pages_[static_cast<std::size_t>(row)]);
        }
        return result;
    }

    [[nodiscard]] double primary_unique_bytes() const noexcept {
        return unique_bytes(static_cast<double>(primary_resident_keys()));
    }

    [[nodiscard]] double remote_unique_bytes() const noexcept {
        return unique_bytes(static_cast<double>(remote_resident_keys()));
    }

private:
    static std::size_t plane_allocation_bytes(const CachePlanes& planes) noexcept {
        return planes.k.bytes + planes.v.bytes + planes.k_scale.bytes + planes.v_scale.bytes;
    }

    static double plane_bytes(const CachePlanes& planes) {
        return static_cast<double>(plane_allocation_bytes(planes));
    }

    double unique_bytes(double keys) const {
        const PagedKVStorageLayout layout = paged_kv_storage_layout(storage_, kHeadDim);
        return keys * geometry_.kv_heads *
               static_cast<double>(layout.physical_bytes_per_token_head());
    }

    // Cache keys this row appends: its populated history plus the live query tokens.
    [[nodiscard]] std::int32_t row_appended_keys(std::int32_t row) const noexcept {
        return contexts_[static_cast<std::size_t>(row)] + tokens_;
    }

    [[nodiscard]] std::int32_t primary_keys_in_row(std::int32_t row) const noexcept {
        const std::int32_t page_keys =
            primary_pages_[static_cast<std::size_t>(row)] * kPagedKVPageSize;
        return std::min(row_appended_keys(row), page_keys);
    }

    // Physical page identity. The primary plane holds every row's leading pages and
    // the secondary plane the remainder, so a global ID alone tells both devices
    // which plane owns a page and where it lives inside that plane. An absent primary
    // plane has no local IDs, so this is only reachable when it owns at least one page.
    [[nodiscard]] std::int32_t primary_local_page(std::int32_t row, std::int32_t page) const {
        const std::int32_t linear = primary_offsets_[static_cast<std::size_t>(row)] + page;
        if (mapping_ == PageMapping::Identity) { return linear; }
        const std::int64_t stride = fragment_stride(primary_plane_pages_);
        return static_cast<std::int32_t>((static_cast<std::int64_t>(linear) * stride) %
                                         primary_plane_pages_);
    }

    [[nodiscard]] std::int32_t remote_local_page(std::int32_t row, std::int32_t page) const {
        const std::int32_t local  = page - primary_pages_[static_cast<std::size_t>(row)];
        const std::int32_t linear = remote_offsets_[static_cast<std::size_t>(row)] + local;
        if (mapping_ == PageMapping::Identity) { return linear; }
        const std::int64_t stride = fragment_stride(remote_plane_pages_);
        return static_cast<std::int32_t>((static_cast<std::int64_t>(linear) * stride) %
                                         remote_plane_pages_);
    }

    [[nodiscard]] std::int32_t global_page_id(std::int32_t row, std::int32_t page) const {
        if (page < primary_pages_[static_cast<std::size_t>(row)]) {
            return primary_local_page(row, page);
        }
        return primary_plane_pages_ + remote_local_page(row, page);
    }

    [[nodiscard]] PagedKVBatchLayerView batch_view() const {
        // An empty local shard means no primary planes at all: all four stay null while the
        // block table still spans the complete global namespace on the primary device. The
        // Op accepts that configuration and produces the neutral local contribution.
        const PagedKVStagedPlanes planes = primary_plane_absent_
                                               ? PagedKVStagedPlanes{}
                                               : bind_planes(primary_planes_, geometry_, storage_);
        return {
            .k_pages       = planes.k,
            .v_pages       = planes.v,
            .k_scale_pages = planes.k_scale,
            .v_scale_pages = planes.v_scale,
            .block_tables  = Tensor(global_table_.p, DType::I32, {logical_pages_, batch_}),
            .head_dim      = kHeadDim,
            .num_kv_heads  = geometry_.kv_heads,
            .storage       = storage_,
        };
    }

    // Fill both planes through the public append codec. A row's history is cut at a
    // page boundary, so no page is written twice and every live page is populated.
    // Each device's operands are allocated, written and released while that device is
    // current: the secondary device holds no primary-owned pointers.
    void populate() {
        // One row's leading and trailing histories are contiguous position runs.
        std::int32_t max_primary_tokens = 0;
        std::int32_t max_remote_tokens  = 0;
        for (std::int32_t row = 0; row < batch_; ++row) {
            const std::int32_t live_keys    = row_appended_keys(row);
            const std::int32_t primary_keys = primary_keys_in_row(row);
            max_primary_tokens              = std::max(max_primary_tokens, primary_keys);
            max_remote_tokens               = std::max(max_remote_tokens, live_keys - primary_keys);
        }

        // Plane-local table: each entry is the page's index inside its owning plane,
        // which is what the append codec's physical addressing expects.
        std::vector<std::int32_t> local_table(static_cast<std::size_t>(logical_pages_) * batch_, 0);
        for (std::int32_t row = 0; row < batch_; ++row) {
            for (std::int32_t page = 0; page < logical_pages_; ++page) {
                local_table[static_cast<std::size_t>(row) * logical_pages_ + page] =
                    page < primary_pages_[static_cast<std::size_t>(row)]
                        ? primary_local_page(row, page)
                        : remote_local_page(row, page);
            }
        }

        const auto append = [&](const CachePlanes& planes, DeviceBuffer& table,
                                DeviceBuffer& positions, std::int32_t row, std::int32_t first_key,
                                std::int32_t key_count, unsigned seed) {
            if (key_count <= 0) { return; }
            std::vector<std::int32_t> host_positions(static_cast<std::size_t>(key_count));
            for (std::int32_t i = 0; i < key_count; ++i) {
                host_positions[static_cast<std::size_t>(i)] = first_key + i;
            }
            CUDA_CHECK(cudaMemcpy(positions.p, host_positions.data(),
                                  host_positions.size() * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice));
            auto history_k = varied_values(
                static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads * key_count, seed, .25f,
                static_cast<std::size_t>(first_key) * kHeadDim * geometry_.kv_heads);
            auto history_v = varied_values(
                static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads * key_count, seed + 97u,
                1.f, static_cast<std::size_t>(first_key) * kHeadDim * geometry_.kv_heads);
            const PagedKVStagedPlanes planes_view = bind_planes(planes, geometry_, storage_);
            Tensor hk(history_k.p, DType::BF16, {kHeadDim, geometry_.kv_heads, key_count});
            Tensor hv(history_v.p, DType::BF16, {kHeadDim, geometry_.kv_heads, key_count});
            Tensor hp(positions.p, DType::I32, {key_count});
            Tensor row_table = Tensor(static_cast<std::int32_t*>(table.p) + row * logical_pages_,
                                      DType::I32, {logical_pages_});
            ops::kv_cache_append(hk, hv, hp,
                                 PagedKVLayerView{.k_pages       = planes_view.k,
                                                  .v_pages       = planes_view.v,
                                                  .k_scale_pages = planes_view.k_scale,
                                                  .v_scale_pages = planes_view.v_scale,
                                                  .block_table   = row_table,
                                                  .head_dim      = kHeadDim,
                                                  .num_kv_heads  = geometry_.kv_heads,
                                                  .storage       = storage_},
                                 nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
        };

        // Primary resident pages. An absent primary plane owns no page, so there is nothing
        // to populate and no plane buffer may be bound.
        if (!primary_plane_absent_) {
            const ScopedDevice target(primary_device_);
            DeviceBuffer table(global_table_.bytes);
            DeviceBuffer positions(std::max<std::size_t>(max_primary_tokens, 1) *
                                   sizeof(std::int32_t));
            CUDA_CHECK(
                cudaMemcpy(table.p, local_table.data(), table.bytes, cudaMemcpyHostToDevice));
            for (std::int32_t row = 0; row < batch_; ++row) {
                append(primary_planes_, table, positions, row, 0, primary_keys_in_row(row),
                       211u + static_cast<unsigned>(row));
            }
        }

        // Secondary resident pages.
        {
            const ScopedDevice target(secondary_device_);
            DeviceBuffer table(global_table_.bytes);
            DeviceBuffer positions(std::max<std::size_t>(max_remote_tokens, 1) *
                                   sizeof(std::int32_t));
            CUDA_CHECK(
                cudaMemcpy(table.p, local_table.data(), table.bytes, cudaMemcpyHostToDevice));
            for (std::int32_t row = 0; row < batch_; ++row) {
                const std::int32_t primary_keys = primary_keys_in_row(row);
                append(remote_planes_, table, positions, row, primary_keys,
                       row_appended_keys(row) - primary_keys, 211u + static_cast<unsigned>(row));
            }
        }
    }

    void release_planes() noexcept {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        remote_staged_ = PagedKVStagedPlanes{};
        (void)cudaSetDevice(secondary_device_);
        remote_planes_ = CachePlanes{};
        (void)cudaSetDevice(primary_device_);
        primary_planes_ = CachePlanes{};
        (void)cudaSetDevice(previous);
    }

    Geometry geometry_;
    KvCacheStorage storage_;
    std::int32_t tokens_;
    std::int32_t batch_;
    std::vector<std::int32_t> contexts_;
    PageMapping mapping_;
    int secondary_device_;
    int primary_device_               = 0;
    std::int32_t appended_keys_       = 0;
    std::int32_t logical_pages_       = 0;
    std::int32_t primary_plane_pages_ = 0;
    bool primary_plane_absent_        = false;
    std::int32_t remote_plane_pages_  = 0;
    std::vector<std::int32_t> live_pages_;
    std::vector<std::int32_t> primary_pages_;
    std::vector<std::int32_t> primary_offsets_;
    std::vector<std::int32_t> remote_offsets_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer table_rows_;
    DeviceBuffer global_table_;
    DeviceBuffer output_;
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    Tensor table_rows_tensor_;
    Tensor output_tensor_;
    ops::CausalAttentionExecutionEnvelope envelope_;
    CachePlanes primary_planes_;
    CachePlanes remote_planes_;
    PagedKVStagedPlanes remote_staged_;
    PagedKVBatchLayerView primary_view_;
    std::optional<ops::DistributedAttention> op_;
};

struct DistributedTiming {
    bench::ColdTiming wall;
    bench::ColdTiming primary_event;
    std::size_t secondary_flush_bytes = 0;
    DeviceMemorySnapshot primary_memory{};
    DeviceMemorySnapshot secondary_memory{};
};

// The secondary device's L2 flush, owned for the lifetime of one distributed run. A cold
// sample must start with every plane's cache lines evicted, and the secondary plane is read
// by the secondary device, so flushing the primary L2 alone would leave the remote pages
// resident in the other device's cache and report a fabricated cold latency. The buffer is
// allocated, and released, under the secondary device so it lives where it is written.
class SecondaryFlush {
public:
    explicit SecondaryFlush(int device) : device_(device) {
        const ScopedDevice target(device_);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        try {
            buffer_ = DeviceBuffer(kFlushBytes);
        } catch (...) {
            CUDA_CHECK(cudaStreamDestroy(stream_));
            stream_ = nullptr;
            throw;
        }
    }

    ~SecondaryFlush() {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device_);
        buffer_ = DeviceBuffer{};
        if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
        (void)cudaSetDevice(previous);
    }

    SecondaryFlush(const SecondaryFlush&)            = delete;
    SecondaryFlush& operator=(const SecondaryFlush&) = delete;

    // Completes before returning, so the caller's timed interval opens on an evicted cache.
    void flush() {
        const ScopedDevice target(device_);
        bench::flush_l2(buffer_, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    [[nodiscard]] std::size_t bytes() const noexcept { return buffer_.bytes; }

private:
    int device_ = 0;
    DeviceBuffer buffer_;
    cudaStream_t stream_ = nullptr;
};

// Wall is the host-observed completion latency of the complete call; primary_event is the
// CUDA-event span on the primary stream. The two are distinct scopes, not a stage decomposition.
// Cold samples evict L2 on both devices before timing.
DistributedTiming measure_distributed(DistributedCase& data, CacheState cache, int secondary_device,
                                      DeviceBuffer& flush, cudaStream_t stream, int warmup,
                                      int repeat) {
    if (repeat <= 0 || warmup < 0) {
        throw std::invalid_argument("benchmark requires nonnegative warmup and positive repeat");
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::optional<SecondaryFlush> secondary_flush;
    if (cache == CacheState::Cold) { secondary_flush.emplace(secondary_device); }

    // Snapshot memory after every benchmark-owned allocation, outside timed intervals. In cold
    // mode this includes each device's 256 MiB L2-flush buffer.
    const DeviceMemorySnapshot primary_memory   = device_memory_snapshot(data.primary_device());
    const DeviceMemorySnapshot secondary_memory = device_memory_snapshot(secondary_device);

    const auto flush_cold = [&] {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        secondary_flush->flush();
    };

    for (int index = 0; index < warmup; ++index) {
        if (cache == CacheState::Cold) { flush_cold(); }
        data.execute(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> wall_samples;
    std::vector<double> primary_event_samples;
    wall_samples.reserve(static_cast<std::size_t>(repeat));
    primary_event_samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        if (cache == CacheState::Cold) { flush_cold(); }
        const auto begin = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(start, stream));
        data.execute(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        // execute completes every secondary read and write before returning.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto end     = std::chrono::steady_clock::now();
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        primary_event_samples.push_back(static_cast<double>(milliseconds) * 1000.0);
        wall_samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {bench::summarize_timings(std::move(wall_samples)),
            bench::summarize_timings(std::move(primary_event_samples)),
            secondary_flush ? secondary_flush->bytes() : 0, primary_memory, secondary_memory};
}

const char* placement_name(Placement placement) {
    return placement == Placement::Shard ? "shard" : "layer";
}

const char* entry_name(Entry entry) {
    switch (entry) {
    case Entry::Append:
        return "append";
    case Entry::Cached:
        return "cached";
    case Entry::Distributed:
        return "distributed";
    case Entry::Both:
        break;
    }
    return "both";
}

const char* operation_name(Entry entry) {
    switch (entry) {
    case Entry::Append:
        return "causal_softmax_attention";
    case Entry::Cached:
        return "causal_softmax_attention_cached";
    case Entry::Distributed:
        return "DistributedAttention::execute";
    case Entry::Both:
        break;
    }
    return "both";
}

const char* storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

const char* mapping_name(PageMapping mapping) {
    return mapping == PageMapping::Identity ? "identity" : "fragmented";
}

std::string profile_name(std::span<const std::int32_t> values) {
    std::string result;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) { result.push_back(';'); }
        result += std::to_string(values[index]);
    }
    return result;
}

double cache_vector_bytes(const PagedKVVectorLayout& layout) {
    return static_cast<double>(layout.physical_bytes());
}

double causal_key_sum(std::int32_t tokens, std::int32_t context) {
    const double t = static_cast<double>(tokens);
    return t * context + t * static_cast<double>(tokens + 1) * 0.5;
}

double causal_key_sum(std::span<const std::int32_t> contexts,
                      std::span<const std::int32_t> valid_columns) {
    double total = 0.0;
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        total += causal_key_sum(valid_columns[row], contexts[row]);
    }
    return total;
}

double modeled_bytes(Entry entry, const Geometry& geometry, KvCacheStorage storage,
                     std::span<const std::int32_t> contexts,
                     std::span<const std::int32_t> valid_columns, bool physical_cache) {
    const auto layout              = paged_kv_storage_layout(storage, kHeadDim);
    std::int64_t valid_token_count = 0;
    for (const std::int32_t valid : valid_columns) { valid_token_count += valid; }
    const double valid_tokens = static_cast<double>(valid_token_count);
    const double q_and_output = 2.0 * kHeadDim * geometry.query_heads * valid_tokens * 2.0;
    const double key_bytes =
        physical_cache ? cache_vector_bytes(layout.key) : layout.logical_vector_bytes();
    const double value_bytes =
        physical_cache ? cache_vector_bytes(layout.value) : layout.logical_vector_bytes();
    const double cache_reads =
        causal_key_sum(contexts, valid_columns) * geometry.kv_heads * (key_bytes + value_bytes);
    if (entry == Entry::Cached) { return q_and_output + cache_reads; }
    const double input_kv     = 2.0 * kHeadDim * geometry.kv_heads * valid_tokens * 2.0;
    const double cache_writes = geometry.kv_heads * valid_tokens * (key_bytes + value_bytes);
    return q_and_output + cache_reads + input_kv + cache_writes;
}

double visible_cache_vectors(std::span<const std::int32_t> contexts,
                             std::span<const std::int32_t> valid_columns) {
    double visible_rows = 0.0;
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        if (valid_columns[row] > 0) { visible_rows += contexts[row] + valid_columns[row]; }
    }
    return visible_rows;
}

double cache_footprint_bytes(const Geometry& geometry, KvCacheStorage storage,
                             std::span<const std::int32_t> contexts,
                             std::span<const std::int32_t> valid_columns, bool physical) {
    const auto layout = paged_kv_storage_layout(storage, kHeadDim);
    const double key_bytes =
        physical ? cache_vector_bytes(layout.key) : layout.logical_vector_bytes();
    const double value_bytes =
        physical ? cache_vector_bytes(layout.value) : layout.logical_vector_bytes();
    return visible_cache_vectors(contexts, valid_columns) * geometry.kv_heads *
           (key_bytes + value_bytes);
}

double contraction_flops(const Geometry& geometry, std::span<const std::int32_t> contexts,
                         std::span<const std::int32_t> valid_columns) {
    return 2.0 * kHeadDim * geometry.query_heads * causal_key_sum(contexts, valid_columns);
}

double unique_kv_bytes(const Geometry& geometry, KvCacheStorage storage,
                       std::span<const std::int32_t> contexts,
                       std::span<const std::int32_t> valid_columns) {
    return cache_footprint_bytes(geometry, storage, contexts, valid_columns, true);
}

bench::ColdTiming measure(Case& data, Entry entry, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, DeviceBuffer& flush, cudaStream_t stream,
                          int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); };
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, stream, warmup, repeat)
                   : bench::measure_launch(launch, stream, warmup, repeat);
    }
    return cache == CacheState::Cold
               ? bench::measure_cold_graph(*graph, flush, stream, warmup, repeat)
               : bench::measure_graph(*graph, stream, warmup, repeat);
}

void report_distributed(const Result& result) {
    std::printf("entry=distributed op=DistributedAttention::execute placement=%-5s geometry=%-14s "
                "kv=%-6s mapping=%-10s cache=%-4s execution=eager graph=unsupported "
                "B=%d W=%d contexts=%s residency=%s logical_pages=%d primary_dev=%d "
                "secondary_dev=%d "
                "wall_host_median=%10.3f us wall_host_min=%10.3f us wall_host_p95=%10.3f us "
                "primary_event_median=%10.3f us primary_event_min=%10.3f us "
                "primary_event_p95=%10.3f us\n",
                placement_name(result.placement), result.geometry.name,
                storage_name(result.storage), mapping_name(result.mapping),
                cache_name(result.cache), result.batch, result.tokens, result.row_contexts.c_str(),
                result.residency.c_str(), result.logical_pages, result.primary_device,
                result.secondary_device, result.timing.median_us, result.timing.min_us,
                result.timing.p95_us, result.stream_timing.median_us, result.stream_timing.min_us,
                result.stream_timing.p95_us);
    std::printf("  planes primary=%d remote=%d plane_bytes primary=%.0f remote=%.0f "
                "op_device_bytes primary=%zu secondary=%zu op_host_staging_bytes=%zu "
                "primary_input_bytes=%zu setup_device_bytes primary=%zu secondary=%zu "
                "cold_flush_bytes primary=%zu secondary=%zu "
                "device_free_after_setup primary=%zu/%zu secondary=%zu/%zu "
                "resident_keys primary=%lld remote=%lld appended_total=%d "
                "unique_cache_bytes primary=%.0f remote=%.0f\n",
                result.primary_plane_pages, result.remote_plane_pages, result.primary_plane_bytes,
                result.remote_plane_bytes, result.op_primary_bytes, result.op_secondary_bytes,
                result.op_host_staging_bytes, result.primary_input_bytes,
                result.primary_setup_device_bytes, result.secondary_setup_device_bytes, kFlushBytes,
                result.secondary_flush_bytes, result.primary_memory.free_bytes,
                result.primary_memory.total_bytes, result.secondary_memory.free_bytes,
                result.secondary_memory.total_bytes,
                static_cast<long long>(result.primary_resident_keys),
                static_cast<long long>(result.remote_resident_keys), result.appended_keys,
                result.primary_unique_bytes, result.remote_unique_bytes);
}

void report(const Result& result) {
    const double seconds       = result.timing.median_us * 1.0e-6;
    const double logical_gbps  = result.logical_bytes / seconds / 1.0e9;
    const double physical_gbps = result.physical_bytes / seconds / 1.0e9;
    const double cache_gbps    = result.unique_kv_bytes / seconds / 1.0e9;
    const double tflops        = (result.qk_flops + result.pv_flops) / seconds / 1.0e12;
    const double qk_tflops     = result.qk_flops / seconds / 1.0e12;
    const double pv_tflops     = result.pv_flops / seconds / 1.0e12;
    std::printf(
        "entry=%-6s geometry=%-14s kv=%-6s mapping=%-10s execution=%-5s cache=%-4s "
        "B=%d W=%d contexts=%s valid=%s rows=%s "
        "workspace=%9zu peak=%9zu nodes=%zu calls=%d median=%10.3f us min=%10.3f us p95=%10.3f us "
        "logical_payload=%8.1f GB/s physical_payload=%8.1f GB/s math=%7.2f TFLOP/s\n",
        entry_name(result.entry), result.geometry.name, storage_name(result.storage),
        mapping_name(result.mapping), execution_name(result.execution), cache_name(result.cache),
        result.batch, result.tokens, result.row_contexts.c_str(), result.valid_columns.c_str(),
        result.table_rows.c_str(), result.workspace_bytes, result.workspace_peak,
        result.graph_nodes, result.graph_calls, result.timing.median_us, result.timing.min_us,
        result.timing.p95_us, logical_gbps, physical_gbps, tflops);
    std::printf("  vectors K=%.0f V=%.0f bytes cache logical=%.0f physical=%.0f bytes "
                "qk_flops=%.0f pv_flops=%.0f qk_full_op=%7.2f TFLOP/s "
                "pv_full_op=%7.2f TFLOP/s unique_kv=%.0f bytes "
                "unique_cache_rate=%8.1f GB/s\n",
                result.key_vector_bytes, result.value_vector_bytes, result.logical_cache_bytes,
                result.physical_cache_bytes, result.qk_flops, result.pv_flops, qk_tflops, pv_tflops,
                result.unique_kv_bytes, cache_gbps);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,geometry,kv_dtype,mapping,execution,cache,B,W,row_contexts,valid_columns,"
              "table_rows,workspace_bytes,logical_bytes,physical_bytes,logical_cache_bytes,"
              "key_vector_bytes,value_vector_bytes,physical_cache_bytes,qk_flops,pv_flops,"
              "qk_full_op_tflops,pv_full_op_tflops,unique_kv_bytes,unique_kv_gbps,median_us,"
              "min_us,p95_us,graph_nodes,workspace_peak_bytes,graph_calls,"
              "placement,secondary_device,residency,logical_pages,primary_plane_pages,"
              "remote_plane_pages,appended_keys,op_primary_bytes,op_secondary_bytes,"
              "primary_plane_bytes,remote_plane_bytes,primary_unique_bytes,remote_unique_bytes,"
              "primary_resident_keys,remote_resident_keys,wall_median_us,wall_min_us,"
              "wall_p95_us,stream_median_us,stream_min_us,stream_p95_us,"
              "operation,timing_scope,primary_stream_timing_scope,transport,primary_device,"
              "primary_flush_bytes,secondary_flush_bytes,primary_setup_device_bytes,"
              "secondary_setup_device_bytes,primary_free_after_setup_bytes,"
              "primary_total_device_bytes,secondary_free_after_setup_bytes,secondary_total_device_"
              "bytes\n";
    for (const Result& result : results) {
        const bool distributed = result.entry == Entry::Distributed;
        output << entry_name(result.entry) << ',' << result.geometry.name << ','
               << storage_name(result.storage) << ',' << mapping_name(result.mapping) << ','
               << execution_name(result.execution) << ',' << cache_name(result.cache) << ','
               << result.batch << ',' << result.tokens << ',' << result.row_contexts << ','
               << result.valid_columns << ',' << result.table_rows;
        const double seconds = result.timing.median_us * 1.0e-6;
        if (distributed) {
            // Single-device payload/rate fields have no distributed interpretation. Per-device
            // resident bytes and the complete-call timing are reported separately below.
            output << ",,,,,,,," << result.qk_flops << ',' << result.pv_flops << ",,,,,";
        } else {
            output << ',' << result.workspace_bytes << ',' << result.logical_bytes << ','
                   << result.physical_bytes << ',' << result.logical_cache_bytes << ','
                   << result.key_vector_bytes << ',' << result.value_vector_bytes << ','
                   << result.physical_cache_bytes << ',' << result.qk_flops << ','
                   << result.pv_flops << ',' << result.qk_flops / seconds / 1.0e12 << ','
                   << result.pv_flops / seconds / 1.0e12 << ',' << result.unique_kv_bytes << ','
                   << result.unique_kv_bytes / seconds / 1.0e9 << ',';
        }
        output << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << ',' << result.graph_nodes << ',' << result.workspace_peak
               << ',' << result.graph_calls << ',';
        if (distributed) {
            output << placement_name(result.placement) << ',' << result.secondary_device << ','
                   << '"' << result.residency << '"' << ',' << result.logical_pages << ','
                   << result.primary_plane_pages << ',' << result.remote_plane_pages << ','
                   << result.appended_keys << ',' << result.op_primary_bytes << ','
                   << result.op_secondary_bytes << ',' << result.primary_plane_bytes << ','
                   << result.remote_plane_bytes << ',' << result.primary_unique_bytes << ','
                   << result.remote_unique_bytes << ',' << result.primary_resident_keys << ','
                   << result.remote_resident_keys << ',' << result.timing.median_us << ','
                   << result.timing.min_us << ',' << result.timing.p95_us << ','
                   << result.stream_timing.median_us << ',' << result.stream_timing.min_us << ','
                   << result.stream_timing.p95_us;
        } else {
            output << ",,,,,,,,,,,,,,,,,,,,";
        }
        output << ',' << operation_name(result.entry) << ','
               << (distributed ? "host_wall_complete_call" : "primary_cuda_event_per_call") << ','
               << (distributed ? "primary_cuda_event_span" : "not_applicable") << ','
               << (distributed ? "pinned_host_staging" : "same_device");
        output << ',';
        if (distributed) { output << result.primary_device; }
        output << ',';
        if (distributed) {
            output << result.primary_input_bytes << ',' << result.op_host_staging_bytes << ','
                   << kFlushBytes << ',' << result.secondary_flush_bytes << ','
                   << result.primary_setup_device_bytes << ','
                   << result.secondary_setup_device_bytes << ',' << result.primary_memory.free_bytes
                   << ',' << result.primary_memory.total_bytes << ','
                   << result.secondary_memory.free_bytes << ','
                   << result.secondary_memory.total_bytes;
        } else {
            output << ",,,,,,,,,";
        }
        output << '\n';
    }
}

void profile(Case& data, Entry entry, const Geometry& geometry, KvCacheStorage storage,
             const Options& options, std::int32_t batch, std::int32_t width,
             std::string_view contexts, std::string_view valid_columns, std::string_view table_rows,
             DeviceBuffer& flush, cudaStream_t stream) {
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(entry, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream, [&](cudaStream_t launch_stream) {
            for (int call = 0; call < options.graph_calls; ++call)
                data.launch(entry, launch_stream);
        });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(stream); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(entry, stream); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf(
        "PROFILE entry=%s geometry=%s kv=%s mapping=%s dispatch=public execution=%s cache=%s "
        "B=%d W=%d contexts=%.*s valid=%.*s rows=%.*s graph_calls=%d\n",
        entry_name(entry), geometry.name, storage_name(storage), mapping_name(options.mapping),
        execution_name(execution), cache_name(cache), batch, width,
        static_cast<int>(contexts.size()), contexts.data(), static_cast<int>(valid_columns.size()),
        valid_columns.data(), static_cast<int>(table_rows.size()), table_rows.data(),
        options.graph_calls);
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(stream);
    else
        data.launch(entry, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::vector<Geometry> selected_geometries(GeometryChoice choice) {
    if (choice == GeometryChoice::H24Kv4) { return {kH24Kv4}; }
    if (choice == GeometryChoice::H16Kv2) { return {kH16Kv2}; }
    return {kH24Kv4, kH16Kv2};
}

std::vector<KvCacheStorage> selected_storages(KvChoice choice) {
    if (choice == KvChoice::Bf16) { return {KvCacheStorage::BFloat16}; }
    if (choice == KvChoice::Int8) { return {KvCacheStorage::Int8Group64}; }
    if (choice == KvChoice::Fp8) { return {KvCacheStorage::Fp8E4M3Row256}; }
    if (choice == KvChoice::Nvfp4) { return {KvCacheStorage::Nvfp4Group16}; }
    if (choice == KvChoice::K8V4) { return {KvCacheStorage::Fp8KeyNvfp4Value}; }
    return {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
            KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value};
}

struct RowProfile {
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
};

RowProfile make_row_profile(const Options& options, std::int32_t batch, std::int32_t width,
                            std::int32_t uniform_context) {
    RowProfile profile;
    profile.contexts =
        options.row_contexts.empty()
            ? std::vector<std::int32_t>(static_cast<std::size_t>(batch), uniform_context)
            : options.row_contexts;
    profile.valid_columns = options.valid_columns.empty()
                                ? std::vector<std::int32_t>(static_cast<std::size_t>(batch), width)
                                : options.valid_columns;
    if (options.table_rows.empty()) {
        profile.table_rows.resize(static_cast<std::size_t>(batch));
        for (std::int32_t row = 0; row < batch; ++row) {
            profile.table_rows[static_cast<std::size_t>(row)] = row;
        }
    } else {
        profile.table_rows = options.table_rows;
    }
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t context = profile.contexts[static_cast<std::size_t>(row)];
        const std::int32_t valid   = profile.valid_columns[static_cast<std::size_t>(row)];
        if (valid < 0 || valid > width || context < 0 ||
            context > static_cast<std::int32_t>(ops::kCausalAttentionMaximumVisibleKeys) - valid) {
            throw std::invalid_argument("row profile exceeds the public causal-attention domain");
        }
    }
    return profile;
}

// Primary residency for one row, from --primary-pages when present, else from
// --primary-fraction over that row's aligned logical page count. Clamped to the
// row's own live pages: a shorter row cannot host more resident pages than it has.
std::vector<std::int32_t> primary_residency(const Options& options,
                                            std::span<const std::int32_t> contexts,
                                            std::int32_t tokens) {
    std::vector<std::int32_t> result(contexts.size(), 0);
    for (std::size_t row = 0; row < contexts.size(); ++row) {
        const auto live = align_context(contexts[row] + tokens) / kPagedKVPageSize;
        if (options.primary_pages >= 0) {
            result[row] = std::min(options.primary_pages, live);
            continue;
        }
        result[row] =
            static_cast<std::int32_t>(static_cast<double>(live) * options.primary_fraction);
    }
    return result;
}

void run_distributed(const Options& options, const Geometry& geometry, KvCacheStorage storage,
                     std::int32_t batch, std::int32_t tokens,
                     std::span<const std::int32_t> contexts, DeviceBuffer& flush,
                     cudaStream_t stream, std::vector<Result>& results) {
    if (options.secondary_device >= 0) {
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (options.secondary_device >= devices) {
            throw std::invalid_argument("--secondary-device is not a usable CUDA device");
        }
    }
    std::vector<std::int32_t> valid(contexts.size(), tokens);
    const std::vector<std::int32_t> residency = primary_residency(options, contexts, tokens);
    const std::vector<Placement> placements =
        options.placement == Placement::Both
            ? std::vector<Placement>{Placement::Shard, Placement::Layer}
            : std::vector<Placement>{options.placement};
    if (options.placement == Placement::Both &&
        std::all_of(residency.begin(), residency.end(),
                    [](std::int32_t pages) { return pages == 0; })) {
        throw std::invalid_argument(
            "this profile gives shard and layer identical zero-primary placement; increase "
            "primary residency or select --placement layer");
    }

    for (const Placement placement : placements) {
        // Both candidates call DistributedAttention::execute with the same shapes, logical KV,
        // query/K/V inputs, and cache state; only the physical page split changes. Layer uses no
        // primary pages, so the primary publishes the Op's neutral empty-shard contribution.
        const std::vector<std::int32_t> split = placement == Placement::Shard
                                                    ? residency
                                                    : std::vector<std::int32_t>(contexts.size(), 0);
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm)) {
                continue;
            }
            DistributedCase data(geometry, storage, tokens,
                                 std::vector<std::int32_t>(contexts.begin(), contexts.end()), split,
                                 options.mapping, options.secondary_device);
            const DistributedTiming timing =
                measure_distributed(data, cache, options.secondary_device, flush, stream,
                                    options.warmup, options.repeat);
            Result result{};
            result.entry         = Entry::Distributed;
            result.geometry      = geometry;
            result.storage       = storage;
            result.execution     = Execution::Eager;
            result.cache         = cache;
            result.mapping       = options.mapping;
            result.batch         = batch;
            result.tokens        = tokens;
            result.row_contexts  = profile_name(contexts);
            result.valid_columns = profile_name(valid);
            result.table_rows.clear();
            for (std::int32_t row = 0; row < batch; ++row) {
                if (row != 0) { result.table_rows.push_back(';'); }
                result.table_rows += std::to_string(row);
            }
            // Distributed cache-read byte rates are intentionally not aggregated across devices.
            // Per-device resident-byte values and exact setup allocations are reported separately.
            result.workspace_bytes      = 0;
            result.logical_bytes        = 0.0;
            result.physical_bytes       = 0.0;
            result.logical_cache_bytes  = 0.0;
            result.key_vector_bytes     = 0.0;
            result.value_vector_bytes   = 0.0;
            result.physical_cache_bytes = 0.0;
            result.qk_flops             = contraction_flops(geometry, contexts, valid);
            result.pv_flops             = contraction_flops(geometry, contexts, valid);
            result.unique_kv_bytes      = 0.0;
            result.timing               = timing.wall;
            result.stream_timing        = timing.primary_event;
            result.placement            = placement;
            result.secondary_device     = data.secondary_device();
            result.primary_device       = data.primary_device();
            result.logical_pages        = data.logical_pages();
            // An absent primary plane reports zero plane pages, so the residency profile
            // and the plane counts agree about an empty local shard.
            result.residency                    = data.primary_residency_profile();
            result.primary_plane_pages          = data.primary_plane_pages();
            result.remote_plane_pages           = data.remote_plane_pages();
            result.appended_keys                = data.appended_keys();
            result.op_primary_bytes             = data.op_primary_bytes();
            result.op_secondary_bytes           = data.op_secondary_bytes();
            result.primary_plane_bytes          = data.primary_plane_bytes();
            result.remote_plane_bytes           = data.remote_plane_bytes();
            result.primary_unique_bytes         = data.primary_unique_bytes();
            result.remote_unique_bytes          = data.remote_unique_bytes();
            result.primary_resident_keys        = data.primary_resident_keys();
            result.remote_resident_keys         = data.remote_resident_keys();
            result.graph_calls                  = 0;
            result.secondary_flush_bytes        = timing.secondary_flush_bytes;
            result.primary_memory               = timing.primary_memory;
            result.secondary_memory             = timing.secondary_memory;
            result.primary_input_bytes          = data.primary_input_bytes();
            result.op_host_staging_bytes        = data.op_host_staging_bytes();
            result.primary_setup_device_bytes   = data.primary_input_bytes() +
                                                  data.primary_plane_allocation_bytes() +
                                                  data.op_primary_bytes() + flush.bytes;
            result.secondary_setup_device_bytes = data.remote_plane_allocation_bytes() +
                                                  data.op_secondary_bytes() +
                                                  timing.secondary_flush_bytes;
            report_distributed(result);
            results.push_back(std::move(result));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        const std::vector<Geometry> geometries     = selected_geometries(options.geometry);
        const std::vector<KvCacheStorage> storages = selected_storages(options.kv);

        if (options.profile) {
            const Entry entry            = options.entry;
            const Geometry geometry      = geometries.front();
            const KvCacheStorage storage = storages.front();
            const std::int32_t batch     = options.batches.front();
            const std::int32_t width     = options.tokens.front();
            const std::int32_t context =
                options.row_contexts.empty() ? options.contexts.front() : 0;
            const RowProfile rows = make_row_profile(options, batch, width, context);
            Case data(geometry, storage, width, rows.contexts, rows.valid_columns, rows.table_rows,
                      options.mapping);
            const std::string context_name = profile_name(rows.contexts);
            const std::string valid_name   = profile_name(rows.valid_columns);
            const std::string table_name   = profile_name(rows.table_rows);
            profile(data, entry, geometry, storage, options, batch, width, context_name, valid_name,
                    table_name, flush, stream);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        if (options.entry == Entry::Distributed) {
            const Geometry geometry      = geometries.front();
            const KvCacheStorage storage = storages.front();
            const std::int32_t batch     = options.batches.front();
            const std::int32_t width     = options.tokens.front();
            const std::int32_t context =
                options.row_contexts.empty() ? options.contexts.front() : 0;
            const RowProfile rows = make_row_profile(options, batch, width, context);
            run_distributed(options, geometry, storage, batch, width, rows.contexts, flush, stream,
                            results);
            write_csv(options, results);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }
        const std::vector<std::int32_t> context_profiles =
            options.row_contexts.empty() ? options.contexts : std::vector<std::int32_t>{0};
        for (const Geometry& geometry : geometries) {
            for (const KvCacheStorage storage : storages) {
                for (const std::int32_t batch : options.batches) {
                    for (const std::int32_t context : context_profiles) {
                        for (const std::int32_t tokens : options.tokens) {
                            const RowProfile rows =
                                make_row_profile(options, batch, tokens, context);
                            Case data(geometry, storage, tokens, rows.contexts, rows.valid_columns,
                                      rows.table_rows, options.mapping);
                            for (const Entry entry : {Entry::Append, Entry::Cached}) {
                                if ((options.entry == Entry::Append && entry != Entry::Append) ||
                                    (options.entry == Entry::Cached && entry != Entry::Cached) ||
                                    (entry == Entry::Cached && batch != 1)) {
                                    continue;
                                }
                                bench::TimedGraph graph;
                                if (options.execution != Execution::Eager) {
                                    data.launch(entry, stream);
                                    CUDA_CHECK(cudaStreamSynchronize(stream));
                                    graph.capture(stream, [&](cudaStream_t launch_stream) {
                                        for (int call = 0; call < options.graph_calls; ++call)
                                            data.launch(entry, launch_stream);
                                    });
                                }
                                for (const Execution execution :
                                     {Execution::Eager, Execution::Graph}) {
                                    if ((options.execution == Execution::Eager &&
                                         execution != Execution::Eager) ||
                                        (options.execution == Execution::Graph &&
                                         execution != Execution::Graph)) {
                                        continue;
                                    }
                                    for (const CacheState cache :
                                         {CacheState::Cold, CacheState::Warm}) {
                                        if ((options.cache == CacheMode::Cold &&
                                             cache != CacheState::Cold) ||
                                            (options.cache == CacheMode::Warm &&
                                             cache != CacheState::Warm)) {
                                            continue;
                                        }
                                        const auto storage_layout =
                                            paged_kv_storage_layout(storage, kHeadDim);
                                        Result result{
                                            entry,
                                            geometry,
                                            storage,
                                            execution,
                                            cache,
                                            options.mapping,
                                            batch,
                                            tokens,
                                            profile_name(rows.contexts),
                                            profile_name(rows.valid_columns),
                                            profile_name(rows.table_rows),
                                            data.workspace_bytes(),
                                            modeled_bytes(entry, geometry, storage, rows.contexts,
                                                          rows.valid_columns, false),
                                            modeled_bytes(entry, geometry, storage, rows.contexts,
                                                          rows.valid_columns, true),
                                            cache_footprint_bytes(geometry, storage, rows.contexts,
                                                                  rows.valid_columns, false),
                                            cache_vector_bytes(storage_layout.key),
                                            cache_vector_bytes(storage_layout.value),
                                            cache_footprint_bytes(geometry, storage, rows.contexts,
                                                                  rows.valid_columns, true),
                                            contraction_flops(geometry, rows.contexts,
                                                              rows.valid_columns),
                                            contraction_flops(geometry, rows.contexts,
                                                              rows.valid_columns),
                                            unique_kv_bytes(geometry, storage, rows.contexts,
                                                            rows.valid_columns),
                                            measure(data, entry, execution, cache, &graph, flush,
                                                    stream, options.warmup, options.repeat)};
                                        result.graph_nodes =
                                            execution == Execution::Graph ? graph.nodes() : 0;
                                        result.workspace_peak = data.workspace_peak();
                                        result.graph_calls =
                                            execution == Execution::Graph ? options.graph_calls : 1;
                                        result.timing.median_us /= result.graph_calls;
                                        result.timing.min_us /= result.graph_calls;
                                        result.timing.p95_us /= result.graph_calls;
                                        report(result);
                                        results.push_back(result);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_causal_softmax_attention_bench: %s\n", error.what());
        return 1;
    }
}
