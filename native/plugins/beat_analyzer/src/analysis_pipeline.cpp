#include "beat_analyzer/analysis_pipeline.h"

#include <algorithm>
#include <chrono>

namespace soniclab::beat_analyzer {

class AnalysisPipeline::RingSink final : public MarkerSink {
public:
    explicit RingSink(AnalysisPipeline& owner) : owner_(owner) {}
    void emit(const AnalysisMarker& marker) override {
        if (!owner_.markers_.push(marker)) owner_.droppedMarkers_.fetch_add(1, std::memory_order_relaxed);
    }

private:
    AnalysisPipeline& owner_;
};

AnalysisPipeline::AnalysisPipeline(std::unique_ptr<OnlineAnalyzer> analyzer, std::size_t fifoChunks,
                                   std::size_t markerCapacity)
    : analyzer_(std::move(analyzer)), chunks_(fifoChunks), markers_(markerCapacity) {}

AnalysisPipeline::~AnalysisPipeline() { stop(); }

void AnalysisPipeline::start(double sampleRate) {
    stop();
    sampleRate_ = sampleRate;
    position_ = 0;
    chunks_.discardAll();
    dropped_.store(0);
    droppedMarkers_.store(0);
    analyzed_.store(0);
    analyzer_->reset(sampleRate);
    quit_.store(false);
    worker_ = std::thread([this] { run(); });
}

void AnalysisPipeline::stop() {
    if (!worker_.joinable()) return;
    quit_.store(true);
    worker_.join();
}

void AnalysisPipeline::pushAudio(const float* const* channels, int numChannels, int numFrames) noexcept {
    if (numChannels <= 0 || numFrames <= 0) return;
    const float scale = 1.0f / static_cast<float>(numChannels);
    int done = 0;
    while (done < numFrames) {
        const int frames = std::min(kChunkFrames, numFrames - done);
        if (chunks_.freeSpace() == 0) {
            dropped_.fetch_add(static_cast<std::uint64_t>(frames), std::memory_order_relaxed);
        } else {
            AudioChunk chunk;
            chunk.firstSample = position_;
            chunk.frames = frames;
            for (int i = 0; i < frames; ++i) {
                float sum = 0.0f;
                for (int ch = 0; ch < numChannels; ++ch) sum += channels[ch][done + i];
                chunk.mono[i] = sum * scale;
            }
            chunks_.push(chunk);
        }
        position_ += frames;
        done += frames;
    }
}

std::size_t AnalysisPipeline::popMarkers(AnalysisMarker* out, std::size_t maximum) noexcept {
    return markers_.popBulk(out, maximum);
}

void AnalysisPipeline::run() {
    RingSink sink(*this);
    AudioChunk chunk;
    while (true) {
        bool any = false;
        while (chunks_.pop(chunk)) {
            any = true;
            analyzer_->analyze(chunk, sink);
            analyzed_.fetch_add(chunk.frames, std::memory_order_relaxed);
        }
        // Drain what is left before quitting so stop() loses nothing pushed before it.
        if (quit_.load()) {
            if (chunks_.available() == 0) break;
            continue;
        }
        // Polling, not a condition variable: notifying one would put a lock
        // on the audio thread's path. 5 ms is well under a UI frame.
        if (!any) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

}  // namespace soniclab::beat_analyzer
