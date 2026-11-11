#include <chrono>
#include <thread>
#include <vector>

#include "beat_analyzer/analysis_pipeline.h"
#include "test_framework.h"

using namespace soniclab;
using namespace soniclab::beat_analyzer;

namespace {

// Emits an onset whenever the signal rises above 0.5 after being below it.
// Test-only: checks the plumbing, not a detector.
class ThresholdAnalyzer final : public OnlineAnalyzer {
public:
    void reset(double) override { above_ = false; }
    void analyze(const AudioChunk& chunk, MarkerSink& sink) override {
        for (int i = 0; i < chunk.frames; ++i) {
            const bool above = chunk.mono[i] > 0.5f;
            if (above && !above_) {
                AnalysisMarker marker;
                marker.samplePosition = chunk.firstSample + i;
                marker.strength = chunk.mono[i];
                sink.emit(marker);
            }
            above_ = above;
        }
    }

private:
    bool above_ = false;
};

std::vector<AnalysisMarker> waitForMarkers(AnalysisPipeline& pipeline, std::size_t count) {
    std::vector<AnalysisMarker> found;
    AnalysisMarker batch[64];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (found.size() < count && std::chrono::steady_clock::now() < deadline) {
        const auto n = pipeline.popMarkers(batch, 64);
        found.insert(found.end(), batch, batch + n);
        if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return found;
}

}  // namespace

TEST("beat_analyzer.pipeline.markers_land_on_exact_samples") {
    AnalysisPipeline pipeline(std::make_unique<ThresholdAnalyzer>());
    pipeline.start(44100.0);
    // Stereo blocks of 512 with clicks at known stream positions, including
    // one that straddles an internal 256-frame chunk boundary.
    const std::vector<std::int64_t> clicks = {100, 255, 256, 1000, 5000};
    std::vector<float> left(512), right(512);
    for (int block = 0; block < 12; ++block) {
        std::fill(left.begin(), left.end(), 0.0f);
        std::fill(right.begin(), right.end(), 0.0f);
        for (auto click : clicks) {
            const auto local = click - block * 512;
            if (local >= 0 && local < 512) left[local] = right[local] = 1.0f;
        }
        const float* channels[2] = {left.data(), right.data()};
        pipeline.pushAudio(channels, 2, 512);
    }
    // 255 and 256 are adjacent samples above the threshold, so they form one onset.
    const auto markers = waitForMarkers(pipeline, 4);
    pipeline.stop();
    CHECK_EQ(markers.size(), std::size_t{4});
    CHECK_EQ(markers[0].samplePosition, std::int64_t{100});
    CHECK_EQ(markers[1].samplePosition, std::int64_t{255});
    CHECK_EQ(markers[2].samplePosition, std::int64_t{1000});
    CHECK_EQ(markers[3].samplePosition, std::int64_t{5000});
    CHECK_EQ(pipeline.framesAnalyzed(), std::int64_t{12 * 512});
    CHECK_EQ(pipeline.droppedFrames(), std::uint64_t{0});
}

TEST("beat_analyzer.pipeline.audio_is_not_modified") {
    AnalysisPipeline pipeline(std::make_unique<PlaceholderAnalyzer>());
    pipeline.start(48000.0);
    std::vector<float> mono(300);
    for (std::size_t i = 0; i < mono.size(); ++i) mono[i] = static_cast<float>(i) * 0.001f;
    const auto copy = mono;
    const float* channels[1] = {mono.data()};
    pipeline.pushAudio(channels, 1, static_cast<int>(mono.size()));
    pipeline.stop();
    CHECK(mono == copy);
    CHECK_EQ(pipeline.framesAnalyzed(), std::int64_t{300});
}

TEST("beat_analyzer.pipeline.overrun_drops_and_keeps_positions") {
    // A tiny FIFO and no worker: the audio thread must drop, not block.
    AnalysisPipeline pipeline(std::make_unique<ThresholdAnalyzer>(), 2);
    std::vector<float> block(256 * 4, 0.0f);
    block[256 * 3 + 10] = 1.0f;  // lands in the dropped chunk
    const float* channels[1] = {block.data()};
    pipeline.pushAudio(channels, 1, static_cast<int>(block.size()));
    CHECK_EQ(pipeline.droppedFrames(), std::uint64_t{512});

    // A later click still gets its true stream position once the worker runs.
    pipeline.start(44100.0);  // start() resets positions, so push again from 0
    std::fill(block.begin(), block.end(), 0.0f);
    block[700] = 1.0f;
    // One chunk at a time, waiting for the worker, so this half never drops.
    for (int chunk = 0; chunk < 4; ++chunk) {
        const float* piece[1] = {block.data() + chunk * 256};
        pipeline.pushAudio(piece, 1, 256);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (pipeline.framesAnalyzed() < (chunk + 1) * 256 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK_EQ(pipeline.droppedFrames(), std::uint64_t{0});
    const auto markers = waitForMarkers(pipeline, 1);
    pipeline.stop();
    CHECK_EQ(markers.size(), std::size_t{1});
    CHECK_EQ(markers[0].samplePosition, std::int64_t{700});
}

TEST("beat_analyzer.pipeline.restart_is_clean") {
    AnalysisPipeline pipeline(std::make_unique<PlaceholderAnalyzer>());
    for (int i = 0; i < 20; ++i) {
        pipeline.start(44100.0);
        CHECK(pipeline.running());
        pipeline.stop();
        CHECK(!pipeline.running());
    }
}
