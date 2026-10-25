#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "product/logging/pretty_format.h"

#include <spdlog/logger.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

class StderrCapture {
public:
    StderrCapture() {
        if (::pipe(pipe_) != 0) { throw std::runtime_error(std::strerror(errno)); }
        saved_ = ::dup(STDERR_FILENO);
        if (saved_ < 0 || ::dup2(pipe_[1], STDERR_FILENO) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        ::close(pipe_[1]);
        pipe_[1] = -1;
    }

    ~StderrCapture() {
        if (saved_ >= 0) {
            (void)::dup2(saved_, STDERR_FILENO);
            ::close(saved_);
        }
        if (pipe_[0] >= 0) { ::close(pipe_[0]); }
    }

    std::string finish() {
        std::fflush(stderr);
        if (::dup2(saved_, STDERR_FILENO) < 0) { throw std::runtime_error(std::strerror(errno)); }
        ::close(saved_);
        saved_ = -1;

        std::string output;
        std::array<char, 4096> buffer{};
        for (;;) {
            const ssize_t count = ::read(pipe_[0], buffer.data(), buffer.size());
            if (count == 0) { break; }
            if (count < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(std::strerror(errno));
            }
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        ::close(pipe_[0]);
        pipe_[0] = -1;
        return output;
    }

private:
    int pipe_[2]{-1, -1};
    int saved_ = -1;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::size_t line_count(std::string_view value) {
    return static_cast<std::size_t>(std::count(value.begin(), value.end(), '\n'));
}

std::string ready_output(const ninfer::LoadSummary& load) {
    StderrCapture capture;
    {
        ninfer::product::LoggingRuntime logging(
            {.logger_name  = "ninfer-serve",
             .color        = ninfer::product::LogColorMode::Never,
             .presentation = ninfer::product::LogPresentation::Service});
        ninfer::product::StartupLogRenderer startup(logging);
        startup.engine_ready(load);
        logging.flush();
    }
    return capture.finish();
}

} // namespace

int main() {
    int failures = 0;
    std::string service_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Auto,
                 .presentation = ninfer::product::LogPresentation::Service});
            logging.logger()->info("throughput | sample");
            logging.flush();
        }
        service_output = capture.finish();
    }
    failures += check(
        std::regex_match(
            service_output,
            std::regex(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}  INFO  throughput \| sample\n$)")),
        "service pretty prefix mismatch");
    failures += check(service_output.find("ninfer-serve") == std::string::npos,
                      "service pretty output repeated the executable name");
    failures += check(service_output.find("\x1b[") == std::string::npos,
                      "redirected service output contains ANSI escapes");

    std::string startup_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Service});
            ninfer::product::StartupLogRenderer startup(logging);
            ninfer::StartupObserver observer = startup.observer();
            observer.callback({.phase  = ninfer::StartupPhase::EngineStartup,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase  = ninfer::StartupPhase::CudaInitialize,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase      = ninfer::StartupPhase::CudaInitialize,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 1'000'000'000});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Begin,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .total         = 16ULL << 30});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Complete,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .current       = 16ULL << 30,
                               .total         = 16ULL << 30,
                               .elapsed_ns    = 2'000'000'000});
            observer.callback({.phase      = ninfer::StartupPhase::EngineStartup,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 3'000'000'000});
            startup.engine_ready({.model_name           = "qwen3.6-27b",
                                  .artifact_id          = "0123456789abcdef0123456789abcdef",
                                  .cuda_sync_mode       = "blocking",
                                  .weight_formats       = {"q4_g64_fp16", "q8_g32_fp16"},
                                  .host_to_device_bytes = 16ULL << 30});
            logging.flush();
        }
        startup_output = capture.finish();
    }
    failures += check(
        line_count(startup_output) == 5 &&
            startup_output.find("starting engine") != std::string::npos &&
            startup_output.find("loading weights | 16.0 GiB") != std::string::npos &&
            startup_output.find("weights ready | 16.0 GiB | 2.0s | 8.00 GiB/s") !=
                std::string::npos &&
            startup_output.find("engine ready | qwen3.6-27b | total 3.0s | weights 16.0 GiB") !=
                std::string::npos &&
            startup_output.find(
                "artifact 0123456789abcdef0123456789abcdef | residual intervention OFF\n") !=
                std::string::npos &&
            startup_output.find("CUDA sync blocking") != std::string::npos &&
            startup_output.find("CUDA initialized") == std::string::npos,
        "normal startup pretty output is noisy or incomplete");

    ninfer::LoadSummary edited;
    edited.model_name                         = "qwen3.6-27b";
    edited.artifact_id                        = "fedcba9876543210fedcba9876543210";
    edited.residual_intervention.enabled      = true;
    edited.residual_intervention.profile_path = "/profiles/arm-a.json";
    for (std::uint32_t layer = 27; layer <= 42; ++layer)
        edited.residual_intervention.layers.push_back(layer);
    const std::string edited_output = ready_output(edited);
    failures +=
        check(edited_output.find(
                  "artifact fedcba9876543210fedcba9876543210 | residual intervention ON | "
                  "profile /profiles/arm-a.json | layers 27..42 (16)\n") != std::string::npos,
              "enabled residual intervention was not reported at startup");
    edited.residual_intervention.layers = {3, 5, 6, 7, 40};
    failures += check(ready_output(edited).find("layers 3,5..7,40 (5)\n") != std::string::npos,
                      "non-contiguous residual layers were not reported as runs");
    const std::array<std::uint32_t, 0> no_layers{};
    const std::array<std::uint32_t, 1> one_layer{7};
    const std::array<std::uint32_t, 4> two_runs{1, 2, 4, 9};
    failures += check(ninfer::product::format_pretty_index_runs(no_layers).empty() &&
                          ninfer::product::format_pretty_index_runs(one_layer) == "7" &&
                          ninfer::product::format_pretty_index_runs(two_runs) == "1..2,4,9",
                      "index run formatting changed");

    std::string tool_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Tool});
            logging.logger()->info("engine ready");
            logging.logger()->error("failed");
            logging.flush();
        }
        tool_output = capture.finish();
    }
    failures +=
        check(tool_output == "engine ready\nerror: failed\n", "tool pretty prefix mismatch");
    return failures == 0 ? 0 : 1;
}
