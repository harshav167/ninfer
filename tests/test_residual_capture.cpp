// Host-only coverage of the ninfer-residual-capture logic: command line, layer selection,
// dataset parsing and the residuals.f32 / manifest.json writer. The Engine route is covered by
// ninfer_qwen3_5_residual_capture_real_test.

#include "capture.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace rc = ninfer::residual_capture;
namespace fs = std::filesystem;

int failures = 0;

void require(bool condition, const std::string& label) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << label << '\n';
}

// Error text is not part of any contract here; the exception type is.
template <class Exception, class Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) { return true; } catch (...) {
        return false;
    }
    return false;
}

template <class Function>
bool rejects(Function&& function) {
    return throws<std::invalid_argument>(function);
}

// The header documents that dataset errors name the offending line.
template <class Function>
bool rejects_naming(Function&& function, std::string_view needle) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        return std::string_view(error.what()).find(needle) != std::string_view::npos;
    } catch (...) { return false; }
    return false;
}

// Scratch space under the repository's ignored .agent-tmp, not the system temporary directory.
struct TemporaryDirectory {
    fs::path path = fs::path(NINFER_SOURCE_DIR) / ".agent-tmp" /
                    ("residual-capture-test-" + std::to_string(::getpid()));

    TemporaryDirectory() {
        fs::remove_all(path);
        fs::create_directories(path);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_file(const fs::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary);
    output << bytes;
}

// Independent little-endian decoder: no memcpy of the host representation.
std::uint32_t word_at(const std::string& bytes, std::size_t index) {
    std::uint32_t word = 0;
    for (std::size_t byte = 0; byte < 4; ++byte) {
        word |= std::uint32_t(static_cast<std::uint8_t>(bytes[4 * index + byte])) << (8U * byte);
    }
    return word;
}

// True when the directory holds exactly these entries, in any order.
bool has_entries(const fs::path& directory, std::vector<std::string> expected) {
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(directory)) {
        names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    std::sort(expected.begin(), expected.end());
    return names == expected;
}

bool anything_published(const fs::path& directory) {
    return fs::exists(directory / "residuals.f32") || fs::exists(directory / "manifest.json");
}

std::vector<const char*> argv_of(std::initializer_list<const char*> values) { return values; }

void test_options() {
    {
        const auto argv = argv_of({"model.ninfer", "--dataset", "d.jsonl", "--output", "out"});
        const rc::Options options = rc::parse_options(argv);
        require(options.model == "model.ninfer" && options.dataset == "d.jsonl" &&
                    options.output == "out",
                "options: required paths");
    }
    {
        const auto argv =
            argv_of({"--dataset", "d.jsonl", "m.ninfer", "--output", "out", "--layers", "21-42,50",
                     "--chat-template", "t.jinja", "--device", "1", "--max-context", "4096",
                     "--prefill-chunk", "2048", "--kv-dtype", "bf16"});
        const rc::Options options = rc::parse_options(argv);
        require(options.model == "m.ninfer", "options: positional model may follow flags");
        require(options.layers == "21-42,50" && options.chat_template == "t.jinja" &&
                    options.device == 1 && options.max_context == 4096 &&
                    options.prefill_chunk == 2048 && options.kv_dtype == "bf16",
                "options: every value flag");
        const auto fp8 = argv_of({"m", "--dataset", "d", "--output", "o", "--kv-dtype", "fp8"});
        require(rc::parse_options(fp8).kv_dtype == "fp8", "options: fp8 is a supported KV dtype");
    }
    {
        const auto argv = argv_of({"--help"});
        require(rc::parse_options(argv).help_requested, "options: --help");
    }
    const auto bad = [](std::initializer_list<const char*> values, const char* label) {
        const auto argv = argv_of(values);
        require(rejects([&] { (void)rc::parse_options(argv); }), label);
    };
    bad({"m", "--dataset", "d", "--output", "o", "--bogus", "1"}, "options: unknown flag");
    bad({"m", "--dataset", "d", "--output", "o", "-x"}, "options: unknown short flag");
    bad({"m", "--dataset"}, "options: missing value");
    bad({"m", "--output", "o"}, "options: dataset is required");
    bad({"m", "--dataset", "d"}, "options: output is required");
    bad({"--dataset", "d", "--output", "o"}, "options: model required");
    bad({"m", "n", "--dataset", "d", "--output", "o"}, "options: second positional");
    bad({"m", "--dataset", "d", "--dataset", "e", "--output", "o"}, "options: repeated flag");
    bad({"m", "--dataset", "d", "--output", "o", "--device", "-1"}, "options: negative device");
    bad({"m", "--dataset", "d", "--output", "o", "--max-context", "0"},
        "options: zero max-context");
    bad({"m", "--dataset", "d", "--output", "o", "--prefill-chunk", "0"},
        "options: zero prefill-chunk");
    bad({"m", "--dataset", "d", "--output", "o", "--prefill-chunk", "x"},
        "options: non-numeric prefill-chunk");
    bad({"m", "--dataset", "d", "--output", "o", "--kv-dtype", "int8"},
        "options: unsupported kv-dtype");
    bad({"m", "--dataset", "d", "--output", "o", "--kv-dtype", "FP8"},
        "options: kv-dtype is lowercase");
    bad({"m", "--dataset", "d", "--output", "o", "--kv-dtype"}, "options: kv-dtype needs a value");
    bad({"m", "--dataset", "d", "--output", "o", "--kv-dtype", "bf16", "--kv-dtype", "fp8"},
        "options: repeated kv-dtype");
    bad({"m", "--dataset", "d", "--output", "o", "--layers", "3,2"},
        "options: layer list is validated at parse time");
}

void test_layers() {
    const auto all = rc::parse_layer_selection("all");
    require(all.all && all.ranges.empty(), "layers: all");
    const auto sixty_four = rc::resolve_layers(all, 64);
    bool identity         = sixty_four.size() == 64;
    for (std::uint32_t layer = 0; identity && layer < 64; ++layer)
        identity = sixty_four[layer] == layer;
    require(identity, "layers: all expands to every layer");

    const auto full_range = rc::resolve_layers(rc::parse_layer_selection("0-63"), 64);
    require(full_range == sixty_four, "layers: 0-63 equals all for 64 layers");

    const auto mixed = rc::resolve_layers(rc::parse_layer_selection("21-23,50"), 64);
    require((mixed == std::vector<std::uint32_t>{21, 22, 23, 50}), "layers: ranges and indices");
    const auto single = rc::resolve_layers(rc::parse_layer_selection("5-5"), 8);
    require((single == std::vector<std::uint32_t>{5}), "layers: degenerate range");
    const auto spaced = rc::resolve_layers(rc::parse_layer_selection("0,2,4"), 8);
    require((spaced == std::vector<std::uint32_t>{0, 2, 4}), "layers: sparse list");

    const auto rejected = [](std::string_view text, const char* label) {
        require(rejects([&] { (void)rc::parse_layer_selection(text); }), label);
    };
    rejected("", "layers: empty text");
    rejected("3,2", "layers: descending list");
    rejected("5-4", "layers: descending range");
    rejected("1,1", "layers: duplicate index");
    rejected("1-3,3", "layers: overlapping range and index");
    rejected("1-3,2-5", "layers: overlapping ranges");
    rejected("1,,2", "layers: empty item");
    rejected("1,", "layers: trailing comma");
    rejected(",1", "layers: leading comma");
    rejected("1-", "layers: open range");
    rejected("-1", "layers: negative index");
    rejected("1-2-3", "layers: chained range");
    rejected("a", "layers: non-numeric");
    rejected(" 1", "layers: whitespace");
    rejected("+1", "layers: explicit sign");
    rejected("4294967296", "layers: uint32 overflow");
    rejected("ALL", "layers: all is lowercase");

    require(rejects([] { (void)rc::resolve_layers(rc::parse_layer_selection("2-5"), 4); }),
            "layers: range beyond the model");
    require(rejects([] { (void)rc::resolve_layers(rc::parse_layer_selection("4"), 4); }),
            "layers: first index beyond the model");
    require(rejects([] { (void)rc::resolve_layers(rc::parse_layer_selection("0"), 0); }),
            "layers: model without layers");
    require(
        rejects([] { (void)rc::resolve_layers(rc::parse_layer_selection("0-4294967295"), 64); }),
        "layers: a huge range is rejected without expansion");
}

void test_dataset() {
    const std::string text =
        "{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n"
        "\n"
        "{\"id\":\"b\",\"label\":\"L\",\"messages\":[{\"role\":\"system\",\"content\":\"s\"},"
        "{\"role\":\"user\",\"content\":\"u\"}],\"assistant_opening\":\"Sure\"}";
    const rc::Dataset dataset = rc::parse_dataset(text);
    require(dataset.samples.size() == 2, "dataset: blank lines are ignored");
    require(dataset.samples[0].id == "a" && dataset.samples[0].label.empty() &&
                dataset.samples[0].assistant_opening.empty() &&
                dataset.samples[0].messages.size() == 1 &&
                dataset.samples[0].messages[0].role == "user" &&
                dataset.samples[0].messages[0].content == "hi",
            "dataset: optional fields may be omitted");
    require(dataset.samples[1].id == "b" && dataset.samples[1].label == "L" &&
                dataset.samples[1].assistant_opening == "Sure" &&
                dataset.samples[1].messages.size() == 2 &&
                dataset.samples[1].messages[0].role == "system",
            "dataset: second sample keeps its label and opening, in input order");
    require(dataset.sha256 == "91740b8ae0905d8c53e69b8ae6bc18a9a7c723599c0b460c01904382f4f2c8a4",
            "dataset: SHA-256 covers the raw input bytes, blank lines included");

    const rc::Dataset crlf = rc::parse_dataset(
        "{\"id\":\"x\",\"messages\":[{\"role\":\"tool\",\"content\":\"\"}]}\r\n   \r\n");
    require(crlf.samples.size() == 1 && crlf.samples[0].messages[0].role == "tool" &&
                crlf.samples[0].messages[0].content.empty(),
            "dataset: CRLF, whitespace-only lines and empty content");
    require(
        rc::parse_dataset("{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n")
                .sha256 == "8e12eaeda1eb01892759c4b5fd5a7eea4dc6ff78ddb3161e10fe2623d6062a88",
        "dataset: SHA-256 test vector");
    for (const char* role : {"system", "developer", "user", "assistant", "tool"}) {
        const std::string line = std::string("{\"id\":\"r\",\"messages\":[{\"role\":\"") + role +
                                 "\",\"content\":\"c\"}]}";
        require(rc::parse_dataset(line).samples[0].messages[0].role == role,
                std::string("dataset: role ") + role);
    }

    const auto bad = [](const std::string& bytes, const char* label) {
        require(rejects([&] { (void)rc::parse_dataset(bytes); }), label);
    };
    const std::string good = "{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    bad("", "dataset: empty input");
    bad("\n  \n\r\n", "dataset: blank-only input");
    bad(good + "\n{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}\n",
        "dataset: duplicate id");
    bad(good + "\n\n{not json}\n", "dataset: malformed JSON");
    bad("[1]\n", "dataset: non-object line");
    bad("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", "dataset: missing id");
    bad("{\"id\":\"\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", "dataset: empty id");
    bad("{\"id\":7,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
        "dataset: non-string id");
    bad("{\"id\":\"a\"}", "dataset: missing messages");
    bad("{\"id\":\"a\",\"messages\":[]}", "dataset: empty messages");
    bad("{\"id\":\"a\",\"messages\":{}}", "dataset: object messages");
    bad("{\"id\":\"a\",\"messages\":[{\"content\":\"x\"}]}", "dataset: missing role");
    bad("{\"id\":\"a\",\"messages\":[{\"role\":\"user\"}]}", "dataset: missing content");
    bad("{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"},"
        "{\"role\":\"narrator\",\"content\":\"x\"}]}",
        "dataset: unknown role");
    bad("{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\"}]}]}",
        "dataset: media-style content");
    bad("{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"x\",\"image\":\"p\"}]}",
        "dataset: unknown message field");
    bad("{\"id\":\"a\",\"images\":[],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
        "dataset: unknown sample field");
    bad("{\"id\":\"a\",\"label\":3,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
        "dataset: non-string label");
    bad("{\"id\":\"a\",\"assistant_opening\":null,\"messages\":"
        "[{\"role\":\"user\",\"content\":\"x\"}]}",
        "dataset: non-string opening");

    // Errors name the offending line; blank lines count toward the numbering.
    require(rejects_naming([&] { (void)rc::parse_dataset(good + "\n\n{not json}\n"); }, "line 3"),
            "dataset: malformed JSON is reported at its physical line");
    require(rejects_naming(
                [&] {
                    (void)rc::parse_dataset(
                        good +
                        "\n{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
                },
                "line 2"),
            "dataset: a duplicate id is reported at its second line");

    TemporaryDirectory directory;
    require(throws<std::runtime_error>(
                [&] { (void)rc::read_dataset(directory.path / "absent.jsonl"); }),
            "dataset: missing file");
    write_file(directory.path / "d.jsonl", good + "\n");
    require(rc::read_dataset(directory.path / "d.jsonl").samples.size() == 1,
            "dataset: file round trip");
    write_file(directory.path / "bad.jsonl", "{oops}\n");
    require(rejects([&] { (void)rc::read_dataset(directory.path / "bad.jsonl"); }),
            "dataset: invalid file contents");
}

void test_token_digest() {
    const std::array<std::int32_t, 3> ids{1, 2, -1};
    require(rc::token_ids_sha256(ids) ==
                "0a1ce634879f6a487527c9a185b1a4a3de7f41238ad4139e3c6d9e0da723628c",
            "digest: little-endian int32 ids, two's complement");
    require(rc::token_ids_sha256({}) ==
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "digest: empty sequence");
}

rc::Manifest sample_manifest() {
    rc::Manifest manifest;
    manifest.artifact_id       = "00112233445566778899aabbccddeeff";
    manifest.model_name        = "demo";
    manifest.hidden_size       = 4;
    manifest.num_hidden_layers = 8;
    manifest.layers            = {3, 7};
    manifest.chat_template     = "artifact";
    manifest.prefill_chunk     = 1024;
    manifest.kv_dtype          = "fp8";
    manifest.dataset_sha256    = "ab";
    manifest.samples           = {{"s0", "first", 10, 0, "d0"}, {"s1", "second", 12, 2, "d1"}};
    return manifest;
}

// Two rows of the eight values that sample_manifest() describes (two layers x hidden 4).
const std::array<std::array<float, 8>, 2> kRows{{
    {1.0F, -2.0F, -0.0F, 0.0F, FLT_MAX, FLT_MIN, 1.0e-40F, 0.15625F},
    {-FLT_MAX, 3.5F, 1.0F / 3.0F, -1.0e-40F, 65504.0F, -0.5F, 123456.789F, 2.0F},
}};

void append_rows(rc::CaptureWriter& writer) {
    writer.append(kRows[0]);
    writer.append(kRows[1]);
}

// Little-endian, bit-exact, row-major [sample][layer slot][hidden], decoded independently.
bool residuals_match_rows(const std::string& residuals) {
    if (residuals.size() != 2 * 8 * 4) { return false; }
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t value = 0; value < 8; ++value) {
            if (word_at(residuals, row * 8 + value) !=
                std::bit_cast<std::uint32_t>(kRows[row][value])) {
                return false;
            }
        }
    }
    return true;
}

void test_manifest_and_residuals() {
    TemporaryDirectory scratch;
    const fs::path output = scratch.path / "nested" / "capture";

    {
        rc::CaptureWriter writer(output, 8);
        require(!anything_published(output), "writer: nothing is published before finish");
        append_rows(writer);
        require(writer.samples() == 2, "writer: sample counter");
        require(rejects([&] { writer.append(std::vector<float>(7)); }),
                "writer: rows have a fixed extent");
        require(writer.samples() == 2, "writer: a rejected row is not counted");
        rc::Manifest wrong = sample_manifest();
        wrong.samples.pop_back();
        require(rejects([&] { writer.finish(wrong); }), "writer: manifest sample count must match");
        wrong        = sample_manifest();
        wrong.layers = {3};
        require(rejects([&] { writer.finish(wrong); }), "writer: manifest geometry must match");
        require(!anything_published(output), "writer: a rejected manifest publishes nothing");
        writer.finish(sample_manifest());
        require(has_entries(output, {"manifest.json", "residuals.f32"}),
                "writer: final files only, no temporaries");
        require(throws<std::logic_error>([&] { writer.finish(sample_manifest()); }),
                "writer: finish is single-shot");
        require(throws<std::logic_error>([&] { writer.append(kRows[0]); }),
                "writer: no rows after finish");
    }
    require(has_entries(output, {"manifest.json", "residuals.f32"}),
            "writer: a published capture survives destruction");

    const std::string residuals = read_file(output / "residuals.f32");
    require(residuals_match_rows(residuals),
            "residuals: little-endian bit-exact, row-major [sample][slot][hidden]");
    require(static_cast<std::uint8_t>(residuals[0]) == 0x00 &&
                static_cast<std::uint8_t>(residuals[1]) == 0x00 &&
                static_cast<std::uint8_t>(residuals[2]) == 0x80 &&
                static_cast<std::uint8_t>(residuals[3]) == 0x3f,
            "residuals: 1.0f serializes as 00 00 80 3f");
    require(static_cast<std::uint8_t>(residuals[4]) == 0x00 &&
                static_cast<std::uint8_t>(residuals[7]) == 0xc0,
            "residuals: -2.0f serializes as 00 00 00 c0");

    const nlohmann::json manifest = nlohmann::json::parse(read_file(output / "manifest.json"));
    std::vector<std::string> keys;
    for (const auto& item : manifest.items()) keys.push_back(item.key());
    std::sort(keys.begin(), keys.end());
    std::vector<std::string> documented{
        "schema_version",    "artifact_id",    "model_name",     "hidden_size",
        "num_hidden_layers", "layers",         "dtype",          "byteorder",
        "residual_point",    "token_position", "reasoning",      "chat_template",
        "prefill_chunk",     "kv_dtype",       "dataset_sha256", "samples"};
    std::sort(documented.begin(), documented.end());
    require(keys == documented, "manifest: exactly the documented fields");
    require(manifest["schema_version"] == 1 &&
                manifest["artifact_id"] == "00112233445566778899aabbccddeeff" &&
                manifest["model_name"] == "demo" && manifest["hidden_size"] == 4 &&
                manifest["num_hidden_layers"] == 8 &&
                manifest["layers"] == nlohmann::json::array({3, 7}) &&
                manifest["dtype"] == "float32" && manifest["byteorder"] == "little" &&
                manifest["residual_point"] ==
                    "layer_output_after_ffn_residual_add_before_dflash_capture" &&
                manifest["token_position"] == "last_prompt_token" &&
                manifest["reasoning"] == "enabled" && manifest["chat_template"] == "artifact" &&
                manifest["prefill_chunk"] == 1024 && manifest["kv_dtype"] == "fp8" &&
                manifest["dataset_sha256"] == "ab",
            "manifest: file-format constants and copied fields");
    require(manifest["samples"].size() == 2 && manifest["samples"][0]["id"] == "s0" &&
                manifest["samples"][0]["label"] == "first" &&
                manifest["samples"][0]["prompt_tokens"] == 10 &&
                manifest["samples"][0]["opening_tokens"] == 0 &&
                manifest["samples"][0]["prompt_token_ids_sha256"] == "d0" &&
                manifest["samples"][1]["id"] == "s1" &&
                manifest["samples"][1]["opening_tokens"] == 2,
            "manifest: per-sample records in input order");
}

void test_output_rules() {
    TemporaryDirectory scratch;

    const fs::path fresh = scratch.path / "a" / "b";
    rc::CaptureWriter::require_available(fresh);
    require(!fs::exists(fresh), "output: availability check creates nothing");

    const fs::path empty = scratch.path / "empty";
    fs::create_directories(empty);
    {
        rc::CaptureWriter writer(empty, 4);
        writer.append(std::vector<float>(4, 1.0F));
        rc::Manifest manifest = sample_manifest();
        manifest.hidden_size  = 2;
        manifest.samples.resize(1);
        writer.finish(manifest);
    }
    require(has_entries(empty, {"manifest.json", "residuals.f32"}),
            "output: an existing empty directory is accepted");

    require(throws<std::runtime_error>([&] { rc::CaptureWriter::require_available(empty); }),
            "output: a populated directory is rejected");
    require(throws<std::runtime_error>([&] { rc::CaptureWriter writer(empty, 4); }),
            "output: the writer rejects a populated directory");
    require(has_entries(empty, {"manifest.json", "residuals.f32"}),
            "output: a rejected directory is left untouched");

    const fs::path file = scratch.path / "file";
    write_file(file, "x");
    require(throws<std::runtime_error>([&] { rc::CaptureWriter::require_available(file); }),
            "output: an existing file is rejected");

    const fs::path abandoned = scratch.path / "abandoned";
    {
        rc::CaptureWriter writer(abandoned, 4);
        writer.append(std::vector<float>(4, 2.0F));
    }
    require(fs::exists(abandoned) && has_entries(abandoned, {}),
            "output: an unfinished capture leaves neither residuals nor a manifest");
    rc::CaptureWriter::require_available(abandoned);

    require(rejects([&] { rc::CaptureWriter writer(scratch.path / "zero", 0); }),
            "output: rows must contain values");
}

// An entry that appears after the writer started and occupies a publication target makes
// finish() fail at a different step each time: payload rename, manifest temporary creation, and
// manifest rename. The writer must remove exactly what it created, leave the conflict and an
// unrelated entry alone, and let a retry in the same directory succeed.
void test_publication_failure(const char* conflict) {
    const std::string where = std::string("publication blocked at ") + conflict + ": ";
    TemporaryDirectory scratch;
    const fs::path output    = scratch.path / "capture";
    const fs::path blocker   = output / conflict;
    const fs::path bystander = output / "notes.txt";

    {
        rc::CaptureWriter writer(output, 8);
        append_rows(writer);
        fs::create_directories(blocker);
        write_file(blocker / "keep", "conflict");
        write_file(bystander, "notes");

        require(throws<std::runtime_error>([&] { writer.finish(sample_manifest()); }),
                where + "finish reports the failure");
        require(has_entries(output, {conflict, "notes.txt"}),
                where + "everything the writer created is gone, the rest is untouched");
        require(throws<std::logic_error>([&] { writer.finish(sample_manifest()); }),
                where + "a failed writer cannot publish again");
        require(throws<std::logic_error>([&] { writer.append(kRows[0]); }),
                where + "a failed writer takes no more rows");
    }
    require(has_entries(output, {conflict, "notes.txt"}),
            where + "destroying the failed writer removes nothing further");
    require(read_file(blocker / "keep") == "conflict" && read_file(bystander) == "notes",
            where + "conflict and bystander contents are preserved");

    fs::remove_all(blocker);
    fs::remove(bystander);
    {
        rc::CaptureWriter retry(output, 8);
        append_rows(retry);
        retry.finish(sample_manifest());
    }
    require(has_entries(output, {"manifest.json", "residuals.f32"}),
            where + "a retry in the same directory publishes");
    require(residuals_match_rows(read_file(output / "residuals.f32")),
            where + "the retried residuals are complete");
    require(nlohmann::json::parse(read_file(output / "manifest.json"))["samples"].size() == 2,
            where + "the retried manifest is complete");
}

} // namespace

int main() {
    test_options();
    test_layers();
    test_dataset();
    test_token_digest();
    test_manifest_and_residuals();
    test_output_rules();
    for (const char* conflict : {"residuals.f32", "manifest.json.tmp", "manifest.json"}) {
        test_publication_failure(conflict);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " residual_capture\n";
    return failures == 0 ? 0 : 1;
}
