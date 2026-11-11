// SonicLab's native beat tracker: tempo, beats, downbeats and beats per bar.
//
// JUCE-free and dependency-free, so it is unit tested on every build. It
// cannot reuse the DAW's analysers: Madmom is Python with non-commercial
// models and aubio is GPL. It follows the classic, well documented pipeline
// instead:
//
//   onset features   24 log-spaced band levels at ~100 frames/s; the onset
//                    envelope is their summed rise (spectral flux per band,
//                    so a kick counts about as much as a snare), plus the
//                    rise of the bands under 160 Hz (kick, bass)
//   tempo            autocorrelation of the onset envelope with harmonics,
//                    weighted by a log-normal prior around 120 BPM and limited
//                    to the user's range
//   beats            dynamic-programming beat tracking (Ellis 2007)
//   downbeats, meter for each candidate bar length and phase, how much
//                    stronger bass onsets and harmonic changes are on that
//                    beat than on the others
//
// Confidences are 0..1 heuristics from peak prominence and score margins,
// not probabilities. Live mode re-estimates every half second on the last
// 12 s and emits beats about 0.25 s after they happen; file mode analyses
// the whole stream in finish().
#pragma once

#include <atomic>
#include <complex>
#include <cstdint>
#include <vector>

#include "soniclab/analysis/analysis_pipeline.h"

namespace soniclab::beat_analyzer {

using analysis::AnalysisSummary;
using analysis::AudioChunk;
using analysis::MarkerSink;

// Spectral features at a hop of about 10 ms. Frame i is centred on sample
// i * hop() of the stream.
class OnsetFeatures {
public:
    static constexpr int kBands = 24;

    void reset(double sampleRate);
    void push(const float* samples, int count);

    double sampleRate() const { return rate_; }
    int hop() const { return hop_; }
    double framesPerSecond() const { return hop_ > 0 ? rate_ / hop_ : 0.0; }
    // Absolute index of the first frame still held.
    std::int64_t firstFrame() const { return first_; }
    std::int64_t endFrame() const { return first_ + static_cast<std::int64_t>(flux_.size()); }

    const std::vector<float>& flux() const { return flux_; }
    const std::vector<float>& lowFlux() const { return lowFlux_; }
    // kBands values per frame, frame-major.
    const std::vector<float>& bands() const { return bands_; }

    // Live mode: forget frames before ``frame`` to bound memory.
    void discardBefore(std::int64_t frame);

private:
    void computeFrame();

    double rate_ = 0.0;
    int hop_ = 0;
    int size_ = 0;
    std::vector<float> window_;
    std::vector<float> pending_;
    std::vector<std::complex<float>> spectrum_;
    std::vector<float> previous_;  // band levels of the previous frame
    std::vector<int> bandEdges_;
    int lowBands_ = 0;  // bands entirely under 160 Hz
    std::int64_t first_ = 0;
    std::vector<float> flux_;
    std::vector<float> lowFlux_;
    std::vector<float> bands_;
};

struct TempoEstimate {
    double bpm = 0.0;
    double periodFrames = 0.0;
    float confidence = 0.0f;
};

struct MeterEstimate {
    int beatsPerBar = 0;  // 0: unknown
    int phase = 0;        // index of the first downbeat in the beat list
    float meterConfidence = 0.0f;
    float downbeatConfidence = 0.0f;
};

// The onset envelope scaled to unit standard deviation (still positive).
std::vector<float> normaliseEnvelope(const float* flux, std::size_t count);

TempoEstimate estimateTempo(const std::vector<float>& envelope, double framesPerSecond, double minBpm,
                            double maxBpm);

// Beat positions in frames of ``envelope``. ``tightness`` is how strongly
// beats are held to the period: 1000 keeps a steady grid through off-beat
// accents (100, Ellis's value, lets the beat slip to the off-beat on dance
// music); freely played music with a wandering tempo follows less well.
std::vector<double> trackBeats(const std::vector<float>& envelope, double periodFrames, double tightness = 1000.0);

// ``fixedBeatsPerBar`` 0 chooses between 3 and 4; 2..7 only finds the phase.
// ``lowFlux``, ``flux`` and ``bands`` cover the same frames as the beats.
MeterEstimate estimateMeter(const std::vector<double>& beats, const float* lowFlux, const float* flux,
                            const float* bands, std::size_t frames, int fixedBeatsPerBar);

// Settings shared with the plugin's parameters. Null pointers mean defaults.
struct BeatTrackerSettings {
    const std::atomic<float>* minBpm = nullptr;       // BPM, default 60
    const std::atomic<float>* maxBpm = nullptr;       // BPM, default 200
    // A choice index: 0 is auto (3 or 4 beats per bar), k is k + 1 beats.
    const std::atomic<float>* meterChoice = nullptr;
};

class BeatTracker final : public analysis::OnlineAnalyzer {
public:
    enum class Mode { Live, File };

    explicit BeatTracker(Mode mode, BeatTrackerSettings settings = {});

    void reset(double sampleRate) override;
    void analyze(const AudioChunk& chunk, MarkerSink& sink) override;
    void finish(MarkerSink& sink) override;
    AnalysisSummary summary() const override;

private:
    void update(MarkerSink& sink, bool final);
    void publish(const TempoEstimate& tempo, const MeterEstimate& meter);
    double setting(const std::atomic<float>* value, double fallback) const;

    const Mode mode_;
    const BeatTrackerSettings settings_;
    OnsetFeatures features_;
    bool haveBase_ = false;
    std::int64_t sampleBase_ = 0;  // stream position of feature sample 0
    std::int64_t lastUpdate_ = 0;  // absolute frame
    double lastEmitted_ = -1e9;    // absolute frame of the last emitted beat
    int lastPosition_ = 0;         // beat in bar of the last emitted beat
    int lastBeatsPerBar_ = 0;
    double smoothedBpm_ = 0.0;

    std::atomic<float> bpm_{0.0f};
    std::atomic<float> tempoConfidence_{0.0f};
    std::atomic<int> beatsPerBar_{0};
    std::atomic<float> meterConfidence_{0.0f};
    std::atomic<float> downbeatConfidence_{0.0f};
};

}  // namespace soniclab::beat_analyzer
