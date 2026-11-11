#include "soniclab/inference/service.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace soniclab::inference {
namespace {

bool terminal(JobState state) {
    return state == JobState::Succeeded || state == JobState::Failed || state == JobState::Cancelled;
}

struct Job : RunGuard {
    std::uint64_t id = 0;
    Model model;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    JobStatus status;
    std::atomic<bool> cancel{false};
    std::function<void()> terminate;
    std::mutex terminateMutex;

    void starting(const std::function<void()>& stop) override {
        std::lock_guard<std::mutex> lock(terminateMutex);
        terminate = stop;
        if (cancel.load(std::memory_order_acquire)) terminate();
    }

    void finished() override {
        std::lock_guard<std::mutex> lock(terminateMutex);
        terminate = nullptr;
    }
};

}  // namespace

struct InferenceService::State {
    mutable std::mutex mutex;
    std::condition_variable waiting;
    std::condition_variable finished;
    std::deque<std::uint64_t> queue;
    std::unordered_map<std::uint64_t, std::shared_ptr<Job>> jobs;
    std::unordered_map<std::uint64_t, Model> models;
    std::uint64_t nextModel = 1;
    std::uint64_t nextJob = 1;
    bool stopping = false;
    std::thread worker;

    State() {
        worker = std::thread([this] { loop(); });
    }

    ~State() {
        std::vector<std::shared_ptr<Job>> running;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            running.reserve(jobs.size());
            for (auto& [id, job] : jobs) {
                if (!terminal(job->status.state)) job->cancel.store(true, std::memory_order_release);
                running.push_back(job);
            }
        }
        for (const auto& job : running) {
            std::lock_guard<std::mutex> lock(job->terminateMutex);
            if (job->terminate) job->terminate();
        }
        waiting.notify_all();
        if (worker.joinable()) worker.join();
    }

    void loop() {
#if defined(__linux__)
        pthread_setname_np(pthread_self(), "soniclab-infer");
#endif
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                waiting.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) return;
                const auto id = queue.front();
                queue.pop_front();
                auto found = jobs.find(id);
                if (found == jobs.end()) continue;
                job = found->second;
                if (job->cancel.load(std::memory_order_acquire)) {
                    job->status = {JobState::Cancelled, 1.0f, "inference cancelled"};
                    finished.notify_all();
                    continue;
                }
                job->status = {JobState::Running, 0.5f, {}};
            }

            auto result = job->model.run(job->inputs, &job->cancel, job.get());
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!result.ok && result.message == "inference cancelled")
                    job->status = {JobState::Cancelled, 1.0f, result.message};
                else if (!result.ok)
                    job->status = {JobState::Failed, 1.0f, result.message};
                else {
                    job->outputs = std::move(result.value);
                    job->status = {JobState::Succeeded, 1.0f, {}};
                }
                job->inputs.clear();
            }
            finished.notify_all();
        }
    }
};

InferenceService::InferenceService() : state_(std::make_unique<State>()) {}
InferenceService::~InferenceService() = default;

Result<std::uint64_t> InferenceService::load(const Model& model) {
    if (!model.loaded()) return Result<std::uint64_t>::failure("model is not loaded");
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->stopping) return Result<std::uint64_t>::failure("inference service is stopping");
    const auto id = state_->nextModel++;
    state_->models.emplace(id, model.share());
    return Result<std::uint64_t>::success(id);
}

void InferenceService::unload(std::uint64_t modelId) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->models.erase(modelId);
}

Result<std::uint64_t> InferenceService::submit(std::uint64_t modelId, std::vector<Tensor> inputs) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->stopping) return Result<std::uint64_t>::failure("inference service is stopping");
    auto model = state_->models.find(modelId);
    if (model == state_->models.end()) return Result<std::uint64_t>::failure("unknown model");
    auto job = std::make_shared<Job>();
    job->id = state_->nextJob++;
    job->model = model->second.share();
    job->inputs = std::move(inputs);
    job->status = {JobState::Queued, 0.0f, {}};
    state_->jobs.emplace(job->id, job);
    state_->queue.push_back(job->id);
    state_->waiting.notify_one();
    return Result<std::uint64_t>::success(job->id);
}

bool InferenceService::cancel(std::uint64_t jobId) {
    std::shared_ptr<Job> job;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        auto found = state_->jobs.find(jobId);
        if (found == state_->jobs.end()) return false;
        job = found->second;
        if (terminal(job->status.state)) return false;
        job->cancel.store(true, std::memory_order_release);
    }
    std::lock_guard<std::mutex> lock(job->terminateMutex);
    if (job->terminate) job->terminate();
    return true;
}

JobStatus InferenceService::status(std::uint64_t jobId) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto found = state_->jobs.find(jobId);
    if (found == state_->jobs.end()) return {JobState::Failed, 1.0f, "unknown job"};
    return found->second->status;
}

Result<std::vector<Tensor>> InferenceService::wait(std::uint64_t jobId) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    auto found = state_->jobs.find(jobId);
    if (found == state_->jobs.end()) return Result<std::vector<Tensor>>::failure("unknown job");
    auto job = found->second;
    state_->finished.wait(lock, [&] { return terminal(job->status.state); });
    if (job->status.state == JobState::Succeeded) return Result<std::vector<Tensor>>::success(job->outputs);
    return Result<std::vector<Tensor>>::failure(job->status.message.empty() ? "inference failed" : job->status.message);
}

}  // namespace soniclab::inference
