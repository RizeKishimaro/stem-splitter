#include "soniclab/inference/model.h"
#include "soniclab/inference/service.h"
#include "test_framework.h"

#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string dataFile(const char* name) { return std::string(SONICLAB_INFERENCE_TESTDATA) + "/" + name; }

soniclab::inference::Tensor vector14(std::initializer_list<float> values) {
    soniclab::inference::Tensor tensor;
    CHECK(tensor.assign({1, static_cast<std::int64_t>(values.size())}, values));
    return tensor;
}

}  // namespace

TEST("inference.load_reports_float32_io") {
    auto loaded = soniclab::inference::Model::load(dataFile("scale_mul.onnx"));
    CHECK(loaded.ok);
    CHECK(loaded.value.loaded());
    CHECK_EQ(loaded.value.info().inputs.size(), 1u);
    CHECK_EQ(loaded.value.info().outputs.size(), 1u);
    CHECK_EQ(loaded.value.info().inputs[0].name, std::string("x"));
    CHECK_EQ(loaded.value.info().outputs[0].name, std::string("y"));
    CHECK_EQ(loaded.value.info().inputs[0].element, std::string("float32"));
    CHECK_EQ(loaded.value.info().inputs[0].shape.size(), 2u);
    CHECK_EQ(loaded.value.info().inputs[0].shape[0], 1);
    CHECK_EQ(loaded.value.info().inputs[0].shape[1], 4);
}

TEST("inference.invalid_model_is_rejected") {
    const auto path = std::string("/tmp/soniclab-not-a-model.onnx");
    {
        std::ofstream out(path, std::ios::binary);
        out << "this is not an onnx model";
    }
    auto loaded = soniclab::inference::Model::load(path);
    CHECK(!loaded.ok);
    CHECK(!loaded.message.empty());
    CHECK(!loaded.value.loaded());
}

TEST("inference.rejects_the_wrong_input_shape") {
    auto loaded = soniclab::inference::Model::load(dataFile("scale_mul.onnx"));
    CHECK(loaded.ok);
    auto shortInput = vector14({1.0f, 2.0f, 3.0f});
    auto result = loaded.value.run({shortInput});
    CHECK(!result.ok);
    CHECK(result.value.empty());
    CHECK(!result.message.empty());

    soniclab::inference::Tensor wrongRank;
    CHECK(wrongRank.assign({4}, {1.0f, 2.0f, 3.0f, 4.0f}));
    result = loaded.value.run({wrongRank});
    CHECK(!result.ok);
    CHECK(result.value.empty());
}

TEST("inference.scale_matches_the_known_tensor") {
    auto loaded = soniclab::inference::Model::load(dataFile("scale_mul.onnx"));
    CHECK(loaded.ok);
    auto result = loaded.value.run({vector14({1.0f, 2.0f, 3.0f, 4.0f})});
    CHECK(result.ok);
    CHECK_EQ(result.value.size(), 1u);
    CHECK_EQ(result.value[0].shape().size(), 2u);
    CHECK_EQ(result.value[0].shape()[0], 1);
    CHECK_EQ(result.value[0].shape()[1], 4);
    const float expected[] = {2.0f, 4.0f, 6.0f, 8.0f};
    CHECK_EQ(result.value[0].size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) CHECK_NEAR(result.value[0].data()[i], expected[i], 1.0e-6);
}

TEST("inference.repeated_runs_match") {
    auto loaded = soniclab::inference::Model::load(dataFile("scale_mul.onnx"));
    CHECK(loaded.ok);
    const auto input = vector14({1.0f, 2.0f, 3.0f, 4.0f});
    auto first = loaded.value.run({input});
    auto second = loaded.value.run({input});
    CHECK(first.ok);
    CHECK(second.ok);
    CHECK_EQ(first.value[0].size(), second.value[0].size());
    for (std::size_t i = 0; i < first.value[0].size(); ++i)
        CHECK_NEAR(first.value[0].data()[i], second.value[0].data()[i], 0.0);
}

TEST("inference.background_job_returns_the_same_numbers") {
    auto loaded = soniclab::inference::Model::load(dataFile("scale_mul.onnx"));
    CHECK(loaded.ok);
    soniclab::inference::InferenceService service;
    auto modelId = service.load(loaded.value);
    CHECK(modelId.ok);
    auto job = service.submit(modelId.value, {vector14({1.0f, 2.0f, 3.0f, 4.0f})});
    CHECK(job.ok);
    CHECK(job.value != 0);
    auto result = service.wait(job.value);
    CHECK(result.ok);
    const auto done = service.status(job.value);
    CHECK(done.state == soniclab::inference::JobState::Succeeded);
    CHECK_NEAR(done.progress, 1.0, 1.0e-6);
    CHECK_EQ(result.value[0].size(), 4u);
    const float expected[] = {2.0f, 4.0f, 6.0f, 8.0f};
    for (std::size_t i = 0; i < 4; ++i) CHECK_NEAR(result.value[0].data()[i], expected[i], 1.0e-6);
}

TEST("inference.queued_job_can_be_cancelled") {
    soniclab::inference::LoadOptions plain;
    plain.optimizeGraph = false;
    auto heavy = soniclab::inference::Model::load(dataFile("chain_mul.onnx"), plain);
    CHECK(heavy.ok);
    auto light = soniclab::inference::Model::load(dataFile("scale_mul.onnx"));
    CHECK(light.ok);

    soniclab::inference::InferenceService service;
    auto heavyId = service.load(heavy.value);
    auto lightId = service.load(light.value);
    CHECK(heavyId.ok);
    CHECK(lightId.ok);

    soniclab::inference::Tensor block;
    CHECK(block.assign({1, 1048576}, std::vector<float>(1048576, 1.0f)));
    auto running = service.submit(heavyId.value, {std::move(block)});
    CHECK(running.ok);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (service.status(running.value).state == soniclab::inference::JobState::Queued &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(service.status(running.value).state == soniclab::inference::JobState::Running);

    auto queued = service.submit(lightId.value, {vector14({1.0f, 2.0f, 3.0f, 4.0f})});
    CHECK(queued.ok);
    CHECK(service.cancel(queued.value));
    auto cancelled = service.wait(queued.value);
    CHECK(!cancelled.ok);
    CHECK(service.status(queued.value).state == soniclab::inference::JobState::Cancelled);

    auto finished = service.wait(running.value);
    CHECK(finished.ok);
    CHECK_EQ(finished.value[0].size(), 1048576u);
    CHECK_NEAR(finished.value[0].data()[0], 1.0, 1.0e-6);
    CHECK_NEAR(finished.value[0].data()[1048575], 1.0, 1.0e-6);
}

TEST("inference.unknown_model_and_unload_are_errors") {
    auto loaded = soniclab::inference::Model::load(dataFile("scale_mul.onnx"));
    CHECK(loaded.ok);
    soniclab::inference::InferenceService service;
    auto missing = service.submit(99, {vector14({1.0f, 2.0f, 3.0f, 4.0f})});
    CHECK(!missing.ok);
    auto modelId = service.load(loaded.value);
    CHECK(modelId.ok);
    service.unload(modelId.value);
    auto after = service.submit(modelId.value, {vector14({1.0f, 2.0f, 3.0f, 4.0f})});
    CHECK(!after.ok);
    CHECK(!service.cancel(0));
}

TEST("inference.service_destructor_joins_the_worker") {
    auto loaded = soniclab::inference::Model::load(dataFile("scale_mul.onnx"));
    CHECK(loaded.ok);
    auto modelId = soniclab::inference::Result<std::uint64_t>::failure("");
    {
        soniclab::inference::InferenceService service;
        modelId = service.load(loaded.value);
        CHECK(modelId.ok);
        auto job = service.submit(modelId.value, {vector14({1.0f, 2.0f, 3.0f, 4.0f})});
        CHECK(job.ok);
    }
}

TEST("inference.cpu_provider_is_available") {
    const auto providers = soniclab::inference::executionProviders();
    bool found = false;
    for (const auto& name : providers) {
        if (name == "CPUExecutionProvider") found = true;
    }
    CHECK(found);

    soniclab::inference::LoadOptions options;
    options.executionProvider = "SonicLabMissingExecutionProvider";
    auto loaded = soniclab::inference::Model::load(dataFile("scale_mul.onnx"), options);
    CHECK(!loaded.ok);
    CHECK(!loaded.message.empty());
}

TEST("inference.tensor_rejects_a_size_mismatch") {
    soniclab::inference::Tensor tensor;
    CHECK(!tensor.assign({1, 4}, {1.0f, 2.0f}));
    CHECK(tensor.empty());
    CHECK(tensor.assign({1, 2}, {1.0f, 2.0f}));
    CHECK_EQ(tensor.size(), 2u);
}
