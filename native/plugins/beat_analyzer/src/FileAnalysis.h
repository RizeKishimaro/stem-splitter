// Offline analysis of an audio file dropped onto the editor.
//
// Message thread: start(), clear(), snapshot(). A worker thread decodes the
// file with JUCE's audio formats, builds a waveform overview, measures the
// peak and runs the same OnlineAnalyzer as the live input over it. Nothing
// here touches the audio thread or the live pipeline, and the file is not
// part of the plugin state: it is a look at a file, not a setting.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <juce_audio_formats/juce_audio_formats.h>

#include "beat_analyzer/analysis_pipeline.h"

namespace soniclab::beat_analyzer {

struct FileAnalysisResult {
    enum class Status { Empty, Analysing, Done, Failed };

    Status status = Status::Empty;
    juce::File file;
    juce::String formatName;
    juce::String error;  // set when status is Failed
    double sampleRate = 0.0;
    int channels = 0;
    std::int64_t lengthFrames = 0;
    float peak = 0.0f;  // largest absolute sample over all channels
    // Min and max of the mono downmix per bucket, kOverviewBuckets of them.
    std::vector<std::pair<float, float>> overview;
    std::vector<AnalysisMarker> markers;

    double lengthSeconds() const { return sampleRate > 0.0 ? static_cast<double>(lengthFrames) / sampleRate : 0.0; }
};

class FileAnalysis {
public:
    using AnalyzerFactory = std::function<std::unique_ptr<OnlineAnalyzer>()>;

    static constexpr int kOverviewBuckets = 1024;
    static constexpr std::size_t kMaxMarkers = 100000;

    explicit FileAnalysis(AnalyzerFactory makeAnalyzer);
    ~FileAnalysis();

    FileAnalysis(const FileAnalysis&) = delete;
    FileAnalysis& operator=(const FileAnalysis&) = delete;

    // Whether a file's extension is one JUCE can decode on this platform
    // (WAV, AIFF, FLAC, Ogg Vorbis; plus MP3, M4A and others on macOS).
    static bool canAnalyse(const juce::File& file);
    // Upper-case extensions for display, for example "WAV, AIFF, FLAC, OGG".
    static juce::String supportedFormatsText();

    // Cancels any running analysis and starts on ``file``.
    void start(const juce::File& file);
    // Cancels any running analysis and forgets the result.
    void clear();

    // Changes whenever snapshot() would return something new, except progress.
    std::uint32_t version() const noexcept { return version_.load(std::memory_order_acquire); }
    // 0..1 while analysing.
    float progress() const noexcept { return progress_.load(std::memory_order_relaxed); }
    FileAnalysisResult snapshot() const;

private:
    void cancel();
    void run(juce::File file);
    void publish(FileAnalysisResult result);

    AnalyzerFactory makeAnalyzer_;
    std::thread worker_;
    std::atomic<bool> cancel_{false};
    std::atomic<float> progress_{0.0f};
    std::atomic<std::uint32_t> version_{0};
    mutable std::mutex mutex_;  // guards result_; worker and message thread only
    FileAnalysisResult result_;
};

}  // namespace soniclab::beat_analyzer
