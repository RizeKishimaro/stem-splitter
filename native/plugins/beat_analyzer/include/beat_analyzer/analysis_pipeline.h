// Moves audio from the realtime callback to a worker thread for analysis,
// and markers from the worker back to the UI. JUCE-free so it is unit tested
// on every platform; the plugin in src/PluginProcessor.cpp only forwards.
//
//   audio thread   pushAudio()   downmix to mono, stamp, push chunks (never blocks)
//   worker thread  analyzer      pops chunks, emits AnalysisMarkers
//   UI thread      popMarkers()  takes the markers
//
// When the worker falls behind the audio thread drops chunks instead of
// waiting, and counts them in droppedFrames(). Chunks carry their stream
// position, so markers after a drop still land on the right sample.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include "soniclab/common/analysis_marker.h"
#include "soniclab/common/spsc_ring.h"

namespace soniclab::beat_analyzer {

inline constexpr int kChunkFrames = 256;

struct AudioChunk {
    std::int64_t firstSample = 0;
    std::int32_t frames = 0;
    float mono[kChunkFrames] = {};
};

class MarkerSink {
public:
    virtual ~MarkerSink() = default;
    virtual void emit(const AnalysisMarker& marker) = 0;
};

// Runs on the worker thread only. May allocate and take its time.
class OnlineAnalyzer {
public:
    virtual ~OnlineAnalyzer() = default;
    virtual void reset(double sampleRate) = 0;
    virtual void analyze(const AudioChunk& chunk, MarkerSink& sink) = 0;
};

// Phase 1 stand-in: consumes audio and reports nothing. The real detector
// (onset envelope + tempo tracking, or a port of the Madmom pipeline) slots
// in behind the same interface.
class PlaceholderAnalyzer final : public OnlineAnalyzer {
public:
    void reset(double) override { framesSeen_ = 0; }
    void analyze(const AudioChunk& chunk, MarkerSink&) override { framesSeen_ += chunk.frames; }
    std::int64_t framesSeen() const { return framesSeen_; }

private:
    std::int64_t framesSeen_ = 0;
};

class AnalysisPipeline {
public:
    // ``fifoChunks`` of kChunkFrames each; 1024 is about 6 s at 44.1 kHz.
    explicit AnalysisPipeline(std::unique_ptr<OnlineAnalyzer> analyzer, std::size_t fifoChunks = 1024,
                              std::size_t markerCapacity = 4096);
    ~AnalysisPipeline();

    AnalysisPipeline(const AnalysisPipeline&) = delete;
    AnalysisPipeline& operator=(const AnalysisPipeline&) = delete;

    // Message thread, with the audio callback not running.
    void start(double sampleRate);
    void stop();
    bool running() const { return worker_.joinable(); }

    // Audio thread. Never locks, allocates or blocks.
    void pushAudio(const float* const* channels, int numChannels, int numFrames) noexcept;

    // UI thread (single consumer).
    std::size_t popMarkers(AnalysisMarker* out, std::size_t maximum) noexcept;

    std::uint64_t droppedFrames() const noexcept { return dropped_.load(std::memory_order_relaxed); }
    std::uint64_t droppedMarkers() const noexcept { return droppedMarkers_.load(std::memory_order_relaxed); }
    std::int64_t framesAnalyzed() const noexcept { return analyzed_.load(std::memory_order_relaxed); }

    OnlineAnalyzer& analyzer() { return *analyzer_; }

private:
    class RingSink;
    void run();

    std::unique_ptr<OnlineAnalyzer> analyzer_;
    SpscRing<AudioChunk> chunks_;
    SpscRing<AnalysisMarker> markers_;
    std::thread worker_;
    std::atomic<bool> quit_{false};
    std::int64_t position_ = 0;  // audio thread only
    double sampleRate_ = 0.0;
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<std::uint64_t> droppedMarkers_{0};
    std::atomic<std::int64_t> analyzed_{0};
};

}  // namespace soniclab::beat_analyzer
