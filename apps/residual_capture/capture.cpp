#include "capture.h"

#include "models/common/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::residual_capture {
namespace {

using Json = nlohmann::json;

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "residuals.f32 stores IEEE-754 binary32 words");

[[noreturn]] void invalid(const std::string& message) { throw std::invalid_argument(message); }

std::uint32_t parse_unsigned(std::string_view text, const std::string& label) {
    std::uint32_t value     = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
        invalid("invalid " + label + ": '" + std::string(text) + "'");
    }
    return value;
}

std::string line_prefix(std::size_t line) { return "line " + std::to_string(line) + ": "; }

std::string string_field(const Json& object, const char* key, const std::string& where,
                         bool required) {
    const auto found = object.find(key);
    if (found == object.end()) {
        if (required) { invalid(where + "missing field '" + key + "'"); }
        return {};
    }
    if (!found->is_string()) { invalid(where + "field '" + key + "' must be a string"); }
    return found->get<std::string>();
}

void reject_unknown_fields(const Json& object, std::initializer_list<std::string_view> allowed,
                           const std::string& where) {
    for (const auto& item : object.items()) {
        if (std::find(allowed.begin(), allowed.end(), std::string_view(item.key())) ==
            allowed.end()) {
            invalid(where + "unknown field '" + item.key() + "'");
        }
    }
}

Sample parse_sample(const Json& value, std::size_t line) {
    const std::string where = line_prefix(line);
    if (!value.is_object()) { invalid(where + "sample must be a JSON object"); }
    reject_unknown_fields(value, {"id", "label", "messages", "assistant_opening"}, where);
    Sample sample;
    sample.id = string_field(value, "id", where, true);
    if (sample.id.empty()) { invalid(where + "field 'id' must not be empty"); }
    sample.label             = string_field(value, "label", where, false);
    sample.assistant_opening = string_field(value, "assistant_opening", where, false);

    const auto messages = value.find("messages");
    if (messages == value.end()) { invalid(where + "missing field 'messages'"); }
    if (!messages->is_array() || messages->empty()) {
        invalid(where + "field 'messages' must be a non-empty array");
    }
    std::size_t index = 0;
    for (const Json& item : *messages) {
        const std::string message_where = where + "messages[" + std::to_string(index++) + "]: ";
        if (!item.is_object()) { invalid(message_where + "message must be a JSON object"); }
        reject_unknown_fields(item, {"role", "content"}, message_where);
        Message message;
        message.role = string_field(item, "role", message_where, true);
        if (message.role != "system" && message.role != "developer" && message.role != "user" &&
            message.role != "assistant" && message.role != "tool") {
            invalid(message_where + "unsupported role '" + message.role + "'");
        }
        const auto content = item.find("content");
        if (content == item.end()) { invalid(message_where + "missing field 'content'"); }
        if (!content->is_string()) {
            invalid(message_where + "field 'content' must be a string (text-only input)");
        }
        message.content = content->get<std::string>();
        sample.messages.push_back(std::move(message));
    }
    return sample;
}

bool is_blank(std::string_view line) {
    return std::all_of(line.begin(), line.end(),
                       [](char c) { return c == ' ' || c == '\t' || c == '\r'; });
}

struct FlagState {
    std::vector<std::string_view> seen;

    void mark(std::string_view flag) {
        if (std::find(seen.begin(), seen.end(), flag) != seen.end()) {
            invalid("option specified more than once: " + std::string(flag));
        }
        seen.push_back(flag);
    }
};

} // namespace

std::string usage_text() {
    return "usage: ninfer-residual-capture <model.ninfer> --dataset <calibration.jsonl> "
           "--output <directory>\n"
           "       [--layers all|LIST] [--chat-template <path>] [--device N]\n"
           "       [--max-context N] [--prefill-chunk N] [--kv-dtype bf16|fp8]\n"
           "\n"
           "  --layers          all (default) or ascending indices and inclusive ranges, e.g. "
           "21-42,50\n"
           "  --chat-template   template used by the production launcher; default is the "
           "artifact template\n"
           "  --device          CUDA ordinal (default 0)\n"
           "  --max-context     longest accepted sequence (default 8192)\n"
           "  --prefill-chunk   prefill window in tokens (default 1024)\n"
           "  --kv-dtype        KV cache storage, bf16 or fp8 (default fp8, as the production "
           "launcher)\n";
}

Options parse_options(std::span<const char* const> arguments) {
    Options out;
    FlagState flags;
    bool have_model = false;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view argument = arguments[i];
        if (argument == "--help") {
            out.help_requested = true;
            return out;
        }
        if (!argument.starts_with('-')) {
            if (have_model) { invalid("unexpected extra argument: " + std::string(argument)); }
            out.model  = std::filesystem::path(std::string(argument));
            have_model = true;
            continue;
        }
        const auto value = [&]() -> std::string_view {
            if (i + 1 >= arguments.size()) { invalid(std::string(argument) + " requires a value"); }
            return arguments[++i];
        };
        if (argument == "--dataset") {
            flags.mark(argument);
            out.dataset = std::filesystem::path(std::string(value()));
        } else if (argument == "--output") {
            flags.mark(argument);
            out.output = std::filesystem::path(std::string(value()));
        } else if (argument == "--layers") {
            flags.mark(argument);
            out.layers = std::string(value());
        } else if (argument == "--chat-template") {
            flags.mark(argument);
            out.chat_template = std::filesystem::path(std::string(value()));
        } else if (argument == "--device") {
            flags.mark(argument);
            const std::uint32_t device = parse_unsigned(value(), "device");
            if (device > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                invalid("invalid device: " + std::to_string(device));
            }
            out.device = static_cast<int>(device);
        } else if (argument == "--max-context") {
            flags.mark(argument);
            out.max_context = parse_unsigned(value(), "max-context");
            if (out.max_context == 0) { invalid("max-context must be positive"); }
        } else if (argument == "--prefill-chunk") {
            flags.mark(argument);
            out.prefill_chunk = parse_unsigned(value(), "prefill-chunk");
            if (out.prefill_chunk == 0) { invalid("prefill-chunk must be positive"); }
        } else if (argument == "--kv-dtype") {
            flags.mark(argument);
            out.kv_dtype = std::string(value());
            if (out.kv_dtype != "fp8" && out.kv_dtype != "bf16") {
                invalid("kv-dtype must be bf16 or fp8, got '" + out.kv_dtype + "'");
            }
        } else {
            invalid("unknown option: " + std::string(argument));
        }
    }
    if (!have_model || out.model.empty()) { invalid("model path is required"); }
    if (out.dataset.empty()) { invalid("--dataset is required"); }
    if (out.output.empty()) { invalid("--output is required"); }
    (void)parse_layer_selection(out.layers);
    return out;
}

LayerSelection parse_layer_selection(std::string_view text) {
    LayerSelection out;
    if (text == "all") {
        out.all = true;
        return out;
    }
    if (text.empty()) { invalid("layer list is empty"); }
    bool have_previous     = false;
    std::uint32_t previous = 0;
    std::size_t begin      = 0;
    for (;;) {
        const std::size_t comma = text.find(',', begin);
        const std::string_view item =
            text.substr(begin, comma == std::string_view::npos ? comma : comma - begin);
        if (item.empty()) { invalid("layer list contains an empty item"); }
        LayerRange range;
        const std::size_t dash = item.find('-');
        if (dash == std::string_view::npos) {
            range.first = range.last = parse_unsigned(item, "layer index");
        } else {
            range.first = parse_unsigned(item.substr(0, dash), "layer range start");
            range.last  = parse_unsigned(item.substr(dash + 1), "layer range end");
            if (range.last < range.first) {
                invalid("layer range is descending: '" + std::string(item) + "'");
            }
        }
        if (have_previous && range.first <= previous) {
            invalid("layer list must be strictly ascending: '" + std::string(item) +
                    "' follows layer " + std::to_string(previous));
        }
        previous      = range.last;
        have_previous = true;
        out.ranges.push_back(range);
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    return out;
}

std::vector<std::uint32_t> resolve_layers(const LayerSelection& selection,
                                          std::uint32_t layer_count) {
    if (layer_count == 0) { invalid("model has no layers"); }
    std::vector<std::uint32_t> layers;
    if (selection.all) {
        layers.resize(layer_count);
        for (std::uint32_t layer = 0; layer < layer_count; ++layer) { layers[layer] = layer; }
        return layers;
    }
    if (selection.ranges.empty()) { invalid("layer selection is empty"); }
    for (const LayerRange& range : selection.ranges) {
        if (range.last >= layer_count) {
            invalid("layer " + std::to_string(range.last) + " is outside the model's " +
                    std::to_string(layer_count) + " layers");
        }
        for (std::uint32_t layer = range.first; layer <= range.last; ++layer) {
            layers.push_back(layer);
        }
    }
    return layers;
}

Dataset parse_dataset(std::string_view bytes) {
    Dataset out;
    std::unordered_map<std::string, std::size_t> first_line;
    std::string_view rest = bytes;
    std::size_t line_no   = 0;
    while (!rest.empty()) {
        const std::size_t newline = rest.find('\n');
        const std::string_view line =
            newline == std::string_view::npos ? rest : rest.substr(0, newline);
        rest = newline == std::string_view::npos ? std::string_view{} : rest.substr(newline + 1);
        ++line_no;
        if (is_blank(line)) { continue; }
        Json value;
        try {
            value = Json::parse(line.begin(), line.end());
        } catch (const Json::exception& error) {
            invalid(line_prefix(line_no) + "invalid JSON: " + error.what());
        }
        Sample sample                   = parse_sample(value, line_no);
        const auto [position, inserted] = first_line.emplace(sample.id, line_no);
        if (!inserted) {
            invalid(line_prefix(line_no) + "duplicate id '" + sample.id + "' (first on line " +
                    std::to_string(position->second) + ")");
        }
        out.samples.push_back(std::move(sample));
    }
    if (out.samples.empty()) { invalid("dataset contains no samples"); }
    out.sha256 = models::frontend::sha256_hex(models::frontend::sha256(bytes));
    return out;
}

Dataset read_dataset(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open dataset: " + path.string()); }
    std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) { throw std::runtime_error("cannot read dataset: " + path.string()); }
    try {
        return parse_dataset(bytes);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(path.string() + ": " + error.what());
    }
}

std::string token_ids_sha256(std::span<const std::int32_t> ids) {
    std::vector<std::uint8_t> bytes(ids.size() * sizeof(std::int32_t));
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const auto word = static_cast<std::uint32_t>(ids[index]);
        for (std::size_t byte = 0; byte < 4; ++byte) {
            bytes[4 * index + byte] = static_cast<std::uint8_t>((word >> (8U * byte)) & 0xffU);
        }
    }
    return models::frontend::sha256_hex(models::frontend::sha256(bytes));
}

std::string manifest_json(const Manifest& manifest) {
    nlohmann::ordered_json out;
    out["schema_version"]    = 1;
    out["artifact_id"]       = manifest.artifact_id;
    out["model_name"]        = manifest.model_name;
    out["hidden_size"]       = manifest.hidden_size;
    out["num_hidden_layers"] = manifest.num_hidden_layers;
    out["layers"]            = manifest.layers;
    out["dtype"]             = "float32";
    out["byteorder"]         = "little";
    out["residual_point"]    = "layer_output_after_ffn_residual_add_before_dflash_capture";
    out["token_position"]    = "last_prompt_token";
    out["reasoning"]         = "enabled";
    out["chat_template"]     = manifest.chat_template;
    out["prefill_chunk"]     = manifest.prefill_chunk;
    out["kv_dtype"]          = manifest.kv_dtype;
    out["dataset_sha256"]    = manifest.dataset_sha256;
    auto& samples = out["samples"] = nlohmann::ordered_json::array();
    for (const SampleRecord& record : manifest.samples) {
        nlohmann::ordered_json item;
        item["id"]                      = record.id;
        item["label"]                   = record.label;
        item["prompt_tokens"]           = record.prompt_tokens;
        item["opening_tokens"]          = record.opening_tokens;
        item["prompt_token_ids_sha256"] = record.prompt_token_ids_sha256;
        samples.push_back(std::move(item));
    }
    return out.dump(2, ' ', false, nlohmann::ordered_json::error_handler_t::replace) + "\n";
}

void CaptureWriter::require_available(const std::filesystem::path& directory) {
    std::error_code error;
    const bool exists = std::filesystem::exists(directory, error);
    if (error) {
        throw std::runtime_error("cannot inspect output directory " + directory.string() + ": " +
                                 error.message());
    }
    if (!exists) { return; }
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("output path exists and is not a directory: " +
                                 directory.string());
    }
    if (std::filesystem::directory_iterator(directory) != std::filesystem::directory_iterator()) {
        throw std::runtime_error("output directory exists and is not empty: " + directory.string());
    }
}

CaptureWriter::CaptureWriter(const std::filesystem::path& directory,
                             std::uint32_t values_per_sample)
    : residuals_temporary_(directory / "residuals.f32.tmp"),
      residuals_final_(directory / "residuals.f32"),
      manifest_temporary_(directory / "manifest.json.tmp"),
      manifest_final_(directory / "manifest.json"), values_per_sample_(values_per_sample) {
    if (values_per_sample_ == 0) { invalid("capture rows must contain values"); }
    require_available(directory);
    std::filesystem::create_directories(directory);
    residuals_.open(residuals_temporary_, std::ios::binary | std::ios::trunc);
    if (!residuals_) { throw std::runtime_error("cannot create " + residuals_temporary_.string()); }
    owns_residuals_ = true;
}

CaptureWriter::~CaptureWriter() { discard(); }

void CaptureWriter::discard() noexcept {
    residuals_.close();
    std::error_code ignored;
    if (owns_residuals_) {
        std::filesystem::remove(residuals_published_ ? residuals_final_ : residuals_temporary_,
                                ignored);
        owns_residuals_ = false;
    }
    if (owns_manifest_) {
        std::filesystem::remove(manifest_temporary_, ignored);
        owns_manifest_ = false;
    }
}

void CaptureWriter::append(std::span<const float> values) {
    if (state_ != State::Open) { throw std::logic_error("capture writer is already finished"); }
    if (values.size() != values_per_sample_) { invalid("residual row has the wrong extent"); }
    if constexpr (std::endian::native == std::endian::little) {
        residuals_.write(reinterpret_cast<const char*>(values.data()),
                         static_cast<std::streamsize>(values.size_bytes()));
    } else {
        // Bounded stack conversion buffer; the row is written in chunks of kWords values.
        constexpr std::size_t kWords = 1024;
        std::array<char, kWords * sizeof(float)> bytes;
        for (std::size_t first = 0; first < values.size(); first += kWords) {
            const std::size_t count = std::min(kWords, values.size() - first);
            for (std::size_t index = 0; index < count; ++index) {
                const auto word = std::bit_cast<std::uint32_t>(values[first + index]);
                for (std::size_t byte = 0; byte < 4; ++byte) {
                    bytes[4 * index + byte] = static_cast<char>((word >> (8U * byte)) & 0xffU);
                }
            }
            residuals_.write(bytes.data(), static_cast<std::streamsize>(count * sizeof(float)));
        }
    }
    if (!residuals_) { throw std::runtime_error("cannot write " + residuals_temporary_.string()); }
    ++samples_;
}

void CaptureWriter::finish(const Manifest& manifest) {
    if (state_ != State::Open) { throw std::logic_error("capture writer is already finished"); }
    if (manifest.samples.size() != samples_) {
        invalid("manifest sample count differs from the written rows");
    }
    if (std::uint64_t(manifest.layers.size()) * manifest.hidden_size != values_per_sample_) {
        invalid("manifest geometry differs from the written rows");
    }

    try {
        residuals_.flush();
        residuals_.close();
        if (!residuals_) {
            throw std::runtime_error("cannot finish " + residuals_temporary_.string());
        }
        std::filesystem::rename(residuals_temporary_, residuals_final_);
        residuals_published_ = true;

        {
            std::ofstream output(manifest_temporary_, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("cannot create " + manifest_temporary_.string());
            }
            owns_manifest_ = true;
            output << manifest_json(manifest);
            output.flush();
            output.close();
            if (!output) {
                throw std::runtime_error("cannot write " + manifest_temporary_.string());
            }
        }
        std::filesystem::rename(manifest_temporary_, manifest_final_);
    } catch (...) {
        discard();
        state_ = State::Abandoned;
        throw;
    }
    // manifest.json is in place: the files now belong to the caller.
    owns_residuals_ = false;
    owns_manifest_  = false;
    state_          = State::Published;
}

} // namespace ninfer::residual_capture
