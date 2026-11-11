#include "soniclab/inference/model.h"

#include <utility>

#include <onnxruntime_cxx_api.h>

namespace soniclab::inference {
namespace {

Ort::Env& environment() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "soniclab");
    return env;
}

std::string cancelledMessage() { return "inference cancelled"; }

}  // namespace

struct Model::Session {
    Ort::Session session;
    ModelInfo info;

    Session(Ort::Session opened, ModelInfo described) : session(std::move(opened)), info(std::move(described)) {}
};

std::vector<std::string> executionProviders() {
    try {
        return Ort::GetAvailableProviders();
    } catch (const Ort::Exception& error) {
        return {std::string("unavailable: ") + error.what()};
    }
}

Model::Model() = default;
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::Model(std::shared_ptr<Session> session) : session_(std::move(session)) {}

bool Model::loaded() const noexcept { return static_cast<bool>(session_); }

const ModelInfo& Model::info() const {
    static const ModelInfo empty;
    return session_ ? session_->info : empty;
}

Model Model::share() const { return Model(session_); }

Result<Model> Model::load(const std::string& path, const LoadOptions& options) {
    try {
        Ort::SessionOptions sessionOptions;
        // One thread. A plugin process already has an audio thread, and the
        // prebuilt runtime's own pool reports false races under ThreadSanitizer
        // because that binary was not built with the sanitizer.
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(options.optimizeGraph ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                                                        : GraphOptimizationLevel::ORT_DISABLE_ALL);
        if (!options.executionProvider.empty() && options.executionProvider != "CPUExecutionProvider") {
            sessionOptions.AppendExecutionProvider(options.executionProvider, {});
        }
        Ort::Session session(environment(), path.c_str(), sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;

        ModelInfo info;
        const auto describe = [&](std::size_t count, bool input, std::vector<TensorSpec>& out) {
            for (std::size_t index = 0; index < count; ++index) {
                auto name = input ? session.GetInputNameAllocated(index, allocator)
                                  : session.GetOutputNameAllocated(index, allocator);
                auto type = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
                auto tensor = type.GetTensorTypeAndShapeInfo();
                if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                    throw std::runtime_error(std::string(name.get()) + " is not float32");
                TensorSpec spec;
                spec.name = name.get();
                spec.shape = tensor.GetShape();
                out.push_back(std::move(spec));
            }
        };
        describe(session.GetInputCount(), true, info.inputs);
        describe(session.GetOutputCount(), false, info.outputs);
        if (info.inputs.empty() || info.outputs.empty())
            return Result<Model>::failure("the model has no float32 input or output");
        return Result<Model>::success(Model(std::make_shared<Session>(std::move(session), std::move(info))));
    } catch (const Ort::Exception& error) {
        return Result<Model>::failure(error.what());
    } catch (const std::exception& error) {
        return Result<Model>::failure(error.what());
    }
}

Result<std::vector<Tensor>> Model::run(const std::vector<Tensor>& inputs, const std::atomic<bool>* cancel,
                                       RunGuard* guard) const {
    if (!session_) return Result<std::vector<Tensor>>::failure("model is not loaded");
    if (cancel != nullptr && cancel->load(std::memory_order_acquire))
        return Result<std::vector<Tensor>>::failure(cancelledMessage());
    const auto& expected = session_->info.inputs;
    if (inputs.size() != expected.size())
        return Result<std::vector<Tensor>>::failure("expected " + std::to_string(expected.size()) + " input tensors, got " +
                                                    std::to_string(inputs.size()));

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].empty()) return Result<std::vector<Tensor>>::failure(expected[i].name + " is empty");
        if (!shapeMatches(expected[i].shape, inputs[i].shape()))
            return Result<std::vector<Tensor>>::failure(expected[i].name + " shape " + shapeText(inputs[i].shape()) +
                                                        " does not match " + shapeText(expected[i].shape));
    }

    bool started = false;
    try {
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> values;
        values.reserve(inputs.size());
        std::vector<const char*> inputNames;
        inputNames.reserve(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            inputNames.push_back(expected[i].name.c_str());
            // The session reads these floats and does not write them.
            // The vector must stay alive until Run returns.
            auto* samples = const_cast<float*>(inputs[i].data());
            values.push_back(Ort::Value::CreateTensor<float>(memory, samples, inputs[i].size(), inputs[i].shape().data(),
                                                             inputs[i].shape().size()));
        }
        std::vector<const char*> outputNames;
        outputNames.reserve(session_->info.outputs.size());
        for (const auto& output : session_->info.outputs) outputNames.push_back(output.name.c_str());

        Ort::RunOptions runOptions;
        const auto terminate = [&runOptions] { runOptions.SetTerminate(); };
        if (guard != nullptr) {
            guard->starting(terminate);
            started = true;
        }
        if (cancel != nullptr && cancel->load(std::memory_order_acquire)) {
            if (guard != nullptr) guard->finished();
            return Result<std::vector<Tensor>>::failure(cancelledMessage());
        }
        auto produced = session_->session.Run(runOptions, inputNames.data(), values.data(), values.size(),
                                              outputNames.data(), outputNames.size());
        if (guard != nullptr) guard->finished();
        started = false;
        if (cancel != nullptr && cancel->load(std::memory_order_acquire))
            return Result<std::vector<Tensor>>::failure(cancelledMessage());

        std::vector<Tensor> tensors;
        tensors.reserve(produced.size());
        for (std::size_t i = 0; i < produced.size(); ++i) {
            auto type = produced[i].GetTensorTypeAndShapeInfo();
            if (type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                return Result<std::vector<Tensor>>::failure(session_->info.outputs[i].name + " is not float32");
            const auto shape = type.GetShape();
            std::size_t count = 0;
            if (!elementCount(shape, count))
                return Result<std::vector<Tensor>>::failure(session_->info.outputs[i].name + " has a dynamic shape");
            const float* data = produced[i].GetTensorData<float>();
            Tensor tensor;
            if (!tensor.assign(shape, std::vector<float>(data, data + count)))
                return Result<std::vector<Tensor>>::failure("could not store " + session_->info.outputs[i].name);
            tensors.push_back(std::move(tensor));
        }
        return Result<std::vector<Tensor>>::success(std::move(tensors));
    } catch (const Ort::Exception& error) {
        if (started && guard != nullptr) guard->finished();
        if (cancel != nullptr && cancel->load(std::memory_order_acquire))
            return Result<std::vector<Tensor>>::failure(cancelledMessage());
        return Result<std::vector<Tensor>>::failure(error.what());
    }
}

}  // namespace soniclab::inference
