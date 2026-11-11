#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "soniclab/engine_capi.h"
#include "test_framework.h"

namespace {

struct Engine {
    sl_engine* handle;
    Engine(double rate = 44100.0, int32_t channels = 2, int32_t block = 512)
        : handle(sl_engine_create(rate, channels, block)) {}
    ~Engine() { sl_engine_destroy(handle); }
};

}  // namespace

TEST("engine_capi.version_and_create_limits") {
    CHECK_EQ(sl_engine_abi_version(), SL_ENGINE_ABI_VERSION);
    CHECK(sl_engine_create(44100.0, 0, 512) == nullptr);
    CHECK(sl_engine_create(44100.0, 2, 0) == nullptr);
    CHECK(sl_engine_create(100.0, 2, 512) == nullptr);
    Engine engine;
    CHECK(engine.handle != nullptr);
    CHECK_EQ(sl_engine_channels(engine.handle), 2);
    CHECK_EQ(sl_engine_max_block_frames(engine.handle), 512);
    sl_engine_destroy(nullptr);  // allowed, does nothing
}

TEST("engine_capi.null_and_range_errors") {
    Engine engine;
    float buffer[8] = {};
    CHECK_EQ(sl_engine_process(nullptr, buffer, 4), SL_ERR_NULL);
    CHECK_EQ(sl_engine_process(engine.handle, nullptr, 4), SL_ERR_NULL);
    CHECK_EQ(sl_engine_process(engine.handle, buffer, 513), SL_ERR_ARG);
    CHECK_EQ(sl_engine_set_output_gain(engine.handle, NAN), SL_ERR_ARG);
    CHECK_EQ(sl_engine_stop_device(engine.handle), SL_ERR_STATE);
    CHECK_EQ(sl_engine_start_device(engine.handle, 30000.0f), SL_ERR_ARG);
    sl_engine_stats stats;
    CHECK_EQ(sl_engine_get_stats(nullptr, &stats), SL_ERR_NULL);
}

TEST("engine_capi.process_applies_master_stage_in_place") {
    Engine engine;
    std::vector<float> buffer = {0.1f, -0.2f, 0.9f, -0.9f};
    CHECK_EQ(sl_engine_process(engine.handle, buffer.data(), 2), SL_OK);  // default gain 1
    CHECK_EQ(buffer[2], 0.9f);
    CHECK_EQ(sl_engine_set_output_gain(engine.handle, 1.5f), SL_OK);
    CHECK_EQ(sl_engine_process(engine.handle, buffer.data(), 2), SL_OK);
    CHECK_NEAR(buffer[0], 0.15, 1e-6);
    CHECK_EQ(buffer[2], 1.0f);  // clipped
    CHECK_EQ(buffer[3], -1.0f);
    CHECK_EQ(sl_engine_take_peak(engine.handle), 1.0f);
    CHECK_EQ(sl_engine_take_peak(engine.handle), 0.0f);
    CHECK_EQ(sl_engine_set_master_muted(engine.handle, 1), SL_OK);
    CHECK_EQ(sl_engine_process(engine.handle, buffer.data(), 2), SL_OK);
    for (float sample : buffer) CHECK_EQ(sample, 0.0f);
    sl_engine_stats stats{};
    CHECK_EQ(sl_engine_get_stats(engine.handle, &stats), SL_OK);
    CHECK_EQ(stats.blocks_processed, int64_t{3});
    CHECK_EQ(stats.frames_processed, int64_t{6});
    CHECK_EQ(stats.commands_applied, int64_t{2});
}

TEST("engine_capi.gain_is_clamped_like_the_python_engine") {
    Engine engine(44100.0, 1, 4);
    float buffer[1] = {0.5f};
    sl_engine_set_output_gain(engine.handle, 9.0f);
    sl_engine_process(engine.handle, buffer, 1);
    CHECK_EQ(buffer[0], 0.75f);  // 0.5 * 1.5
    buffer[0] = 0.5f;
    sl_engine_set_output_gain(engine.handle, -3.0f);
    sl_engine_process(engine.handle, buffer, 1);
    CHECK_EQ(buffer[0], 0.0f);
}

TEST("engine_capi.device_thread_owns_audio_side") {
    Engine engine(48000.0, 2, 256);
    CHECK_EQ(sl_engine_start_device(engine.handle, 440.0f), SL_OK);
    CHECK_EQ(sl_engine_device_running(engine.handle), 1);
    CHECK_EQ(sl_engine_start_device(engine.handle, 440.0f), SL_ERR_BUSY);
    float buffer[4] = {};
    CHECK_EQ(sl_engine_process(engine.handle, buffer, 2), SL_ERR_BUSY);

    std::vector<float> out(48000 * 2);
    int32_t got = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (got < 4800 && std::chrono::steady_clock::now() < deadline) {
        const auto n = sl_engine_read_output(engine.handle, out.data() + got * 2, 48000 - got);
        CHECK(n >= 0);
        got += n;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(got >= 4800);
    float peak = 0.0f;
    for (int32_t i = 0; i < got * 2; ++i) peak = std::max(peak, std::fabs(out[static_cast<std::size_t>(i)]));
    CHECK_NEAR(peak, 0.5, 0.01);
    CHECK_EQ(out[0], out[1]);  // whole interleaved frames, both channels equal

    // A queued gain change reaches the running device.
    CHECK_EQ(sl_engine_set_master_muted(engine.handle, 1), SL_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sl_engine_take_peak(engine.handle);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK_EQ(sl_engine_take_peak(engine.handle), 0.0f);

    CHECK_EQ(sl_engine_stop_device(engine.handle), SL_OK);
    CHECK_EQ(sl_engine_device_running(engine.handle), 0);
    CHECK_EQ(sl_engine_process(engine.handle, buffer, 2), SL_OK);  // offline use is available again
}

TEST("engine_capi.destroy_stops_a_running_device") {
    for (int i = 0; i < 10; ++i) {
        Engine engine(44100.0, 2, 64);
        CHECK_EQ(sl_engine_start_device(engine.handle, 220.0f), SL_OK);
    }
}

TEST("engine_capi.many_control_threads_share_one_queue") {
    Engine engine(44100.0, 2, 128);
    CHECK_EQ(sl_engine_start_device(engine.handle, 100.0f), SL_OK);
    std::vector<std::thread> controls;
    for (int t = 0; t < 4; ++t)
        controls.emplace_back([&engine] {
            float out[256];
            for (int i = 0; i < 200; ++i) {
                while (sl_engine_set_output_gain(engine.handle, 0.5f) == SL_ERR_FULL) std::this_thread::yield();
                sl_engine_read_output(engine.handle, out, 128);
            }
        });
    for (auto& thread : controls) thread.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    sl_engine_stats stats{};
    sl_engine_get_stats(engine.handle, &stats);
    CHECK_EQ(stats.commands_applied, int64_t{800});
}
