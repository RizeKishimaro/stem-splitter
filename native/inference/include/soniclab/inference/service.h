// One background thread that loads nothing itself: callers load a Model on
// a normal thread, then submit tensors here. The audio callback must not
// call submit, wait, load, or run.
//
// Progress is the job state, not a percentage from inside the model.
// ONNX Runtime does not report partial progress for a single Run.
//   queued    0
//   running   0.5
//   finished  1
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "soniclab/inference/model.h"

namespace soniclab::inference {

enum class JobState { Queued, Running, Succeeded, Failed, Cancelled };

struct JobStatus {
    JobState state = JobState::Queued;
    float progress = 0.0f;
    std::string message;
};

class InferenceService {
public:
    InferenceService();
    ~InferenceService();

    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    // The service keeps a shared copy of the model's session. Destroying
    // the caller's Model does not cancel jobs already accepted. unload
    // rejects new submit calls for that id and leaves accepted jobs running.
    Result<std::uint64_t> load(const Model& model);
    void unload(std::uint64_t modelId);

    // Copies ``inputs`` into the job so the caller may reuse or destroy them.
    Result<std::uint64_t> submit(std::uint64_t modelId, std::vector<Tensor> inputs);

    // Queued jobs never start. A job already inside Run is asked to stop;
    // if the runtime cannot interrupt it, the job still finishes.
    bool cancel(std::uint64_t jobId);

    JobStatus status(std::uint64_t jobId) const;

    // Blocks the calling thread until the job reaches a terminal state.
    // Not for the audio thread.
    Result<std::vector<Tensor>> wait(std::uint64_t jobId);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace soniclab::inference
