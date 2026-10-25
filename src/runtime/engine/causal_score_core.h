#pragma once

#include "ninfer/types.h"
#include "runtime/engine/context_capacity.h"
#include "runtime/observability/snapshot.h"

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// One scoring or residual-capture job owns the execution slot until its result is delivered.
template <class Instance>
class CausalScoreCore {
public:
    static constexpr EnginePurpose purpose = EnginePurpose::CausalScoring;

    using ModelContract  = typename Instance::ModelContract;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;

    // Whether the model's Session observes residuals; architectures without that capability
    // reject capture requests instead of providing a placeholder.
    static constexpr bool supports_residual_capture =
        requires(Instance& instance, PreparedPrompt prompt, std::span<const TokenId> suffix,
                 std::span<const std::uint32_t> layers) {
            instance.session->start_residual_capture(std::move(prompt), suffix, layers);
        };

    explicit CausalScoreCore(Instance& instance) : instance_(instance) {
        published_snapshot_.ranks.assign(instance_.executor->observations().begin(),
                                         instance_.executor->observations().end());
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                instance_.executor->bind_to_current_thread();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    ~CausalScoreCore() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    CausalScoreCore(const CausalScoreCore&)            = delete;
    CausalScoreCore& operator=(const CausalScoreCore&) = delete;

    [[nodiscard]] std::vector<float> score(PreparedPrompt prompt, std::uint32_t first_target) {
        auto job          = std::make_unique<Job>();
        job->prompt       = std::move(prompt);
        job->first_target = first_target;
        return run(std::move(job)).logprobs;
    }

    // Prefills prompt + forced_suffix and returns the residual of the last token at each layer.
    [[nodiscard]] ResidualCapture
    capture_residuals([[maybe_unused]] PreparedPrompt prompt,
                      [[maybe_unused]] std::span<const TokenId> forced_suffix,
                      [[maybe_unused]] std::span<const std::uint32_t> layers) {
        if constexpr (supports_residual_capture) {
            auto job     = std::make_unique<Job>();
            job->prompt  = std::move(prompt);
            job->capture = true;
            job->forced_suffix.assign(forced_suffix.begin(), forced_suffix.end());
            job->layers.assign(layers.begin(), layers.end());
            return run(std::move(job)).capture;
        } else {
            throw std::invalid_argument("residual capture is not supported by this architecture");
        }
    }

    [[nodiscard]] EngineSnapshot snapshot() const {
        std::lock_guard lock(stats_mutex_);
        return published_snapshot_;
    }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    void reset_measurement_peaks() {
        std::scoped_lock execution_lock(execution_mutex_);
        if (instance_.executor->has_pending_work())
            throw std::logic_error("measurement window requires an idle engine");
        instance_.session->reset_measurement_peaks();
        std::lock_guard stats_lock(stats_mutex_);
        const auto ranks = instance_.executor->observations();
        published_snapshot_.ranks.assign(ranks.begin(), ranks.end());
    }

private:
    void publish() {
        std::lock_guard lock(stats_mutex_);
        published_snapshot_.stamp = {published_snapshot_.stamp.sequence + 1, observation_time_ns()};
        published_snapshot_.scoring    = scoring_;
        published_snapshot_.executions = instance_.executor->execution_statistics();
        const auto ranks               = instance_.executor->observations();
        published_snapshot_.ranks.assign(ranks.begin(), ranks.end());
    }

    struct Completion {
        std::vector<float> logprobs;
        ResidualCapture capture;
    };

    struct Job {
        PreparedPrompt prompt;
        // Scoring: first_target. Capture: forced_suffix and layers.
        std::uint32_t first_target = 0;
        std::vector<TokenId> forced_suffix;
        std::vector<std::uint32_t> layers;
        bool capture = false;
        std::promise<Completion> promise;
    };

    [[nodiscard]] Completion run(std::unique_ptr<Job> job) {
        // One synchronous public call owns the sole job slot until its result is delivered.
        std::scoped_lock call_lock(call_mutex_);
        std::future<Completion> result = job->promise.get_future();
        {
            std::lock_guard queue_lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw std::runtime_error("causal scoring engine is unavailable");
            }
            if (job_ != nullptr) {
                throw std::logic_error("causal scoring core already has an in-flight job");
            }
            job_ = std::move(job);
        }
        queue_cv_.notify_one();
        return result.get();
    }

    [[nodiscard]] auto start(Job& job) {
        if (job.capture) {
            if constexpr (supports_residual_capture) {
                return instance_.session->start_residual_capture(std::move(job.prompt),
                                                                 job.forced_suffix, job.layers);
            } else {
                throw std::logic_error("residual capture reached an architecture without support");
            }
        }
        return instance_.session->start_score(std::move(job.prompt), job.first_target);
    }

    void worker_loop() noexcept {
        for (;;) {
            std::unique_ptr<Job> job;
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [&] { return stopping_ || job_ != nullptr; });
                if (job_ == nullptr) {
                    if (stopping_) { return; }
                    continue;
                }
                job = std::move(job_);
            }
            bool execution_started = false;
            const auto started     = observation_time_ns();
            scoring_.active_jobs   = 1;
            publish();
            try {
                Completion result;
                {
                    std::scoped_lock lock(execution_mutex_);
                    const auto ticket = start(*job);
                    if (!ticket) {
                        instance_.executor->fail("scoring execution slot is occupied");
                        throw std::logic_error("scoring execution slot is occupied");
                    }
                    execution_started = true;
                    instance_.executor->flush();
                    for (;;) {
                        auto event = instance_.session->poll();
                        if (event) {
                            if (event->ticket != *ticket ||
                                !std::holds_alternative<typename ModelContract::ScoreComplete>(
                                    event->result)) {
                                throw std::logic_error(
                                    "scoring received an unrelated execution result");
                            }
                            auto& complete =
                                std::get<typename ModelContract::ScoreComplete>(event->result);
                            result.logprobs = std::move(complete.logprobs);
                            if constexpr (supports_residual_capture) {
                                if (job->capture != complete.residuals.has_value()) {
                                    throw std::logic_error(
                                        "execution result does not match its scoring job kind");
                                }
                                if (complete.residuals) {
                                    auto& observed                  = *complete.residuals;
                                    result.capture.prompt_token_ids = std::move(observed.token_ids);
                                    result.capture.layers           = std::move(job->layers);
                                    result.capture.hidden_size      = observed.hidden_size;
                                    result.capture.values           = std::move(observed.values);
                                }
                            }
                            break;
                        }
                        instance_.executor->flush();
                        std::this_thread::yield();
                    }
                }
                ++scoring_.completed_jobs;
                scoring_.scored_tokens += result.logprobs.size();
                scoring_.active_jobs = 0;
                scoring_.wall.observe(observation_time_ns() - started);
                publish();
                job->promise.set_value(std::move(result));
            } catch (...) {
                const auto error = std::current_exception();
                if (execution_started || instance_.executor->failed()) {
                    {
                        std::scoped_lock lock(execution_mutex_);
                        instance_.session->fail_all_cleanup();
                    }
                    {
                        std::lock_guard lock(queue_mutex_);
                        failed_ = true;
                    }
                }
                ++scoring_.failed_jobs;
                scoring_.active_jobs = 0;
                publish();
                try {
                    job->promise.set_exception(error);
                } catch (...) {}
            }
        }
    }

    Instance& instance_;
    mutable std::mutex execution_mutex_;
    mutable std::mutex stats_mutex_;
    EngineSnapshot published_snapshot_;
    ScoringStatistics scoring_;
    std::mutex call_mutex_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::unique_ptr<Job> job_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
