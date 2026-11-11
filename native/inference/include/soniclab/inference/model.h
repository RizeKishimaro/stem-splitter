// A loaded inference model. The header does not mention a particular
// runtime. Today the only backend is ONNX Runtime; plugins should include
// this header and not the runtime's headers.
//
// Loading reads the model file and creates a session. run() executes one
// forward pass on the calling thread. Neither is safe to call from an
// audio callback. InferenceService runs them on a worker instead.
//
// Models are files supplied by the user. This library does not embed weights.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "soniclab/inference/tensor.h"

namespace soniclab::inference {

struct Status {
    bool ok = false;
    std::string message;

    static Status success() { return {true, {}}; }
    static Status failure(std::string text) { return {false, std::move(text)}; }
};

template <typename T>
struct Result {
    bool ok = false;
    T value{};
    std::string message;

    static Result success(T item) { return {true, std::move(item), {}}; }
    static Result failure(std::string text) { return {false, {}, std::move(text)}; }
};

struct TensorSpec {
    std::string name;
    std::vector<std::int64_t> shape;  // negative: dynamic
    std::string element = "float32";
};

struct ModelInfo {
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
};

// Lets a worker publish the way to stop an in-flight run. starting() is
// called on the worker immediately before execution, finished() immediately
// after. The function passed to starting() may be called from another thread
// and must not be used after finished() returns.
struct RunGuard {
    virtual ~RunGuard() = default;
    virtual void starting(const std::function<void()>& terminate) = 0;
    virtual void finished() = 0;
};

struct LoadOptions {
    // Graph rewrites. Tests that need every node to actually run turn this off.
    bool optimizeGraph = true;
    // "CPUExecutionProvider" is the default and the only provider exercised
    // on the Linux development machine. Any other name is requested from the
    // runtime and fails the load when this build does not contain it.
    std::string executionProvider = "CPUExecutionProvider";
};

// Providers this process's runtime was compiled with. Presence in the list
// is not a test that the provider executes a model correctly.
std::vector<std::string> executionProviders();

class Model {
public:
    Model();
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    static Result<Model> load(const std::string& path, const LoadOptions& options = {});

    bool loaded() const noexcept;
    const ModelInfo& info() const;
    // Another owner of the same session. Used by InferenceService so a job
    // outlives the Model object the caller keeps.
    Model share() const;

    // ``inputs`` follow the model's input order. One float32 tensor each.
    // When ``cancel`` becomes true, a run that the runtime can interrupt
    // stops and the result is a failure whose message says it was cancelled.
    Result<std::vector<Tensor>> run(const std::vector<Tensor>& inputs, const std::atomic<bool>* cancel = nullptr,
                                    RunGuard* guard = nullptr) const;

private:
    struct Session;
    explicit Model(std::shared_ptr<Session> session);
    std::shared_ptr<Session> session_;
};

}  // namespace soniclab::inference
