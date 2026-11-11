// Prototype implementation of soniclab/engine_capi.h.
//
// Control threads talk to the audio side only through lock-free structures
// from soniclab::common: an SpscRing of EngineCommands (control → audio), an
// SpscRing of output samples (audio → control) and an AtomicPeak. The two
// mutexes serialise *control* threads among themselves, because each ring
// has exactly one producer and one consumer. The audio side never takes them.
#include "soniclab/engine_capi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <new>
#include <thread>

#include "soniclab/common/atomic_peak.h"
#include "soniclab/common/engine_command.h"
#include "soniclab/common/spsc_ring.h"

using soniclab::AtomicPeak;
using soniclab::CommandType;
using soniclab::EngineCommand;
using soniclab::SpscRing;

namespace {
constexpr float kMaxOutputGain = 1.5f;  // same clamp as AudioEngine.set_output_gain
constexpr double kTwoPi = 6.283185307179586;
}  // namespace

struct sl_engine {
    sl_engine(double rate, int32_t channelCount, int32_t blockFrames)
        : sampleRate(rate),
          channels(channelCount),
          maxBlock(blockFrames),
          commands(256),
          output(static_cast<std::size_t>(rate) * static_cast<std::size_t>(channelCount)),
          deviceBuffer(std::make_unique<float[]>(static_cast<std::size_t>(blockFrames * channelCount))) {}

    const double sampleRate;
    const int32_t channels;
    const int32_t maxBlock;

    // Control → audio.
    SpscRing<EngineCommand> commands;
    std::mutex commandWriters;

    // Audio → control.
    SpscRing<float> output;
    std::mutex outputReaders;
    AtomicPeak peak;

    // Whoever holds this owns the audio side: the device thread or one
    // sl_engine_process call. Taken with exchange, never waited on.
    std::atomic<bool> audioOwned{false};

    // Audio-side state; touched only by the audio-side owner.
    float gain = 1.0f;
    bool muted = false;
    double phase = 0.0;

    std::unique_ptr<float[]> deviceBuffer;
    std::thread device;
    std::atomic<bool> deviceRunning{false};
    std::atomic<bool> quit{false};

    std::atomic<int64_t> blocks{0};
    std::atomic<int64_t> frames{0};
    std::atomic<int64_t> applied{0};
    std::atomic<int64_t> dropped{0};
    std::atomic<int64_t> maxBlockNs{0};

    int32_t queue(const EngineCommand& command) {
        std::lock_guard<std::mutex> lock(commandWriters);
        return commands.push(command) ? SL_OK : SL_ERR_FULL;
    }

    // Audio side. Mirrors the master stage of AudioEngine._callback:
    // gain, then mute, then clip to ±1, then peak.
    void processBlock(float* data, int32_t frameCount) noexcept {
        const auto started = std::chrono::steady_clock::now();
        EngineCommand command;
        while (commands.pop(command)) {
            if (command.type == CommandType::SetOutputGain) gain = command.value;
            else if (command.type == CommandType::SetMasterMuted) muted = command.value != 0.0f;
            applied.fetch_add(1, std::memory_order_relaxed);
        }
        const auto count = static_cast<std::size_t>(frameCount) * static_cast<std::size_t>(channels);
        float blockPeak = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
            float sample = muted ? 0.0f : data[i] * gain;
            sample = std::clamp(sample, -1.0f, 1.0f);
            data[i] = sample;
            blockPeak = std::max(blockPeak, std::fabs(sample));
        }
        peak.raise(blockPeak);
        blocks.fetch_add(1, std::memory_order_relaxed);
        frames.fetch_add(frameCount, std::memory_order_relaxed);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                            .count();
        auto worst = maxBlockNs.load(std::memory_order_relaxed);
        while (ns > worst && !maxBlockNs.compare_exchange_weak(worst, ns, std::memory_order_relaxed)) {
        }
    }

    // The stand-in device: paced like a real callback, owns its buffer.
    void runDevice(float toneHz) noexcept {
        const auto period = std::chrono::duration<double>(static_cast<double>(maxBlock) / sampleRate);
        const auto step = kTwoPi * static_cast<double>(toneHz) / sampleRate;
        auto next = std::chrono::steady_clock::now();
        float* buffer = deviceBuffer.get();
        while (!quit.load(std::memory_order_acquire)) {
            for (int32_t f = 0; f < maxBlock; ++f) {
                const auto value = static_cast<float>(0.5 * std::sin(phase));
                phase = std::fmod(phase + step, kTwoPi);
                for (int32_t ch = 0; ch < channels; ++ch) buffer[f * channels + ch] = value;
            }
            processBlock(buffer, maxBlock);
            // Push whole frames only so the reader never sees a split frame.
            const auto total = static_cast<std::size_t>(maxBlock) * static_cast<std::size_t>(channels);
            const auto room = output.freeSpace() / static_cast<std::size_t>(channels) * static_cast<std::size_t>(channels);
            const auto pushed = output.pushBulk(buffer, std::min(total, room));
            if (pushed < total)
                dropped.fetch_add(static_cast<int64_t>((total - pushed) / static_cast<std::size_t>(channels)),
                                  std::memory_order_relaxed);
            next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            std::this_thread::sleep_until(next);
        }
    }

    int32_t stopDevice() {
        if (!device.joinable()) return SL_ERR_STATE;
        quit.store(true, std::memory_order_release);
        device.join();
        deviceRunning.store(false);
        audioOwned.store(false, std::memory_order_release);
        return SL_OK;
    }
};

extern "C" {

uint32_t sl_engine_abi_version(void) { return SL_ENGINE_ABI_VERSION; }

sl_engine* sl_engine_create(double sample_rate, int32_t channels, int32_t max_block_frames) {
    if (!(sample_rate >= 8000.0 && sample_rate <= 384000.0)) return nullptr;
    if (channels < 1 || channels > 8 || max_block_frames < 1 || max_block_frames > 8192) return nullptr;
    return new (std::nothrow) sl_engine(sample_rate, channels, max_block_frames);
}

void sl_engine_destroy(sl_engine* engine) {
    if (engine == nullptr) return;
    engine->stopDevice();
    delete engine;
}

int32_t sl_engine_channels(const sl_engine* engine) { return engine ? engine->channels : SL_ERR_NULL; }

int32_t sl_engine_max_block_frames(const sl_engine* engine) { return engine ? engine->maxBlock : SL_ERR_NULL; }

int32_t sl_engine_set_output_gain(sl_engine* engine, float gain) {
    if (engine == nullptr) return SL_ERR_NULL;
    if (!std::isfinite(gain)) return SL_ERR_ARG;
    EngineCommand command;
    command.type = CommandType::SetOutputGain;
    command.value = std::clamp(gain, 0.0f, kMaxOutputGain);
    return engine->queue(command);
}

int32_t sl_engine_set_master_muted(sl_engine* engine, int32_t muted) {
    if (engine == nullptr) return SL_ERR_NULL;
    EngineCommand command;
    command.type = CommandType::SetMasterMuted;
    command.value = muted ? 1.0f : 0.0f;
    return engine->queue(command);
}

int32_t sl_engine_process(sl_engine* engine, float* interleaved, int32_t frames) {
    if (engine == nullptr || interleaved == nullptr) return SL_ERR_NULL;
    if (frames < 0 || frames > engine->maxBlock) return SL_ERR_ARG;
    if (engine->audioOwned.exchange(true, std::memory_order_acquire)) return SL_ERR_BUSY;
    engine->processBlock(interleaved, frames);
    engine->audioOwned.store(false, std::memory_order_release);
    return SL_OK;
}

int32_t sl_engine_start_device(sl_engine* engine, float tone_hz) {
    if (engine == nullptr) return SL_ERR_NULL;
    if (!(tone_hz > 0.0f) || static_cast<double>(tone_hz) >= engine->sampleRate / 2.0) return SL_ERR_ARG;
    if (engine->audioOwned.exchange(true, std::memory_order_acquire)) return SL_ERR_BUSY;
    engine->quit.store(false);
    engine->phase = 0.0;
    try {
        engine->device = std::thread([engine, tone_hz] { engine->runDevice(tone_hz); });
    } catch (...) {
        engine->audioOwned.store(false, std::memory_order_release);
        return SL_ERR_STATE;
    }
    engine->deviceRunning.store(true);
    return SL_OK;
}

int32_t sl_engine_stop_device(sl_engine* engine) {
    if (engine == nullptr) return SL_ERR_NULL;
    return engine->stopDevice();
}

int32_t sl_engine_device_running(const sl_engine* engine) {
    if (engine == nullptr) return SL_ERR_NULL;
    return engine->deviceRunning.load() ? 1 : 0;
}

int32_t sl_engine_read_output(sl_engine* engine, float* out, int32_t max_frames) {
    if (engine == nullptr || out == nullptr) return SL_ERR_NULL;
    if (max_frames < 0) return SL_ERR_ARG;
    std::lock_guard<std::mutex> lock(engine->outputReaders);
    const auto channels = static_cast<std::size_t>(engine->channels);
    const auto ready = engine->output.available() / channels;
    const auto n = std::min(ready, static_cast<std::size_t>(max_frames));
    engine->output.popBulk(out, n * channels);
    return static_cast<int32_t>(n);
}

float sl_engine_take_peak(sl_engine* engine) { return engine ? engine->peak.take() : 0.0f; }

int32_t sl_engine_get_stats(const sl_engine* engine, sl_engine_stats* out) {
    if (engine == nullptr || out == nullptr) return SL_ERR_NULL;
    out->blocks_processed = engine->blocks.load(std::memory_order_relaxed);
    out->frames_processed = engine->frames.load(std::memory_order_relaxed);
    out->commands_applied = engine->applied.load(std::memory_order_relaxed);
    out->output_frames_dropped = engine->dropped.load(std::memory_order_relaxed);
    out->max_block_nanoseconds = engine->maxBlockNs.load(std::memory_order_relaxed);
    return SL_OK;
}

}  // extern "C"
