// Beat tracker tests on synthesised grooves with known tempo, beats and
// downbeats. Accuracy uses the usual beat-tracking tolerance of +-70 ms.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "beat_analyzer/beat_tracker.h"
#include "test_framework.h"

using namespace soniclab;
using namespace soniclab::beat_analyzer;

namespace {

constexpr double kRate = 44100.0;
constexpr double kTolerance = 0.07;  // seconds

struct Groove {
    std::vector<float> audio;
    std::vector<double> beats;      // seconds
    std::vector<double> downbeats;  // seconds
};

void addKick(std::vector<float>& out, std::size_t at, float gain) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(0.18 * kRate) && at + i < out.size(); ++i) {
        const auto t = static_cast<double>(i) / kRate;
        const auto frequency = 55.0 + 90.0 * std::exp(-t * 40.0);  // pitch drop
        out[at + i] += gain * static_cast<float>(std::sin(2.0 * M_PI * frequency * t) * std::exp(-t * 18.0));
    }
}

void addNoise(std::vector<float>& out, std::size_t at, float gain, double decay, bool bright, std::mt19937& rng) {
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    float previous = 0.0f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(0.2 * kRate) && at + i < out.size(); ++i) {
        const auto t = static_cast<double>(i) / kRate;
        auto sample = noise(rng);
        if (bright) {  // first difference: a crude high-pass for hi-hats
            const auto raw = sample;
            sample = raw - previous;
            previous = raw;
        }
        out[at + i] += gain * sample * static_cast<float>(std::exp(-t * decay));
    }
}

// Drums plus a bass line that changes note on every downbeat. In 4/4 the
// kick plays 1 and 3, the snare 2 and 4, hi-hats eighths. In 3/4 the kick
// plays 1, the snare 2 and 3, hi-hats on beats.
Groove makeGroove(double bpm, int beatsPerBar, double seconds, unsigned seed = 7) {
    Groove g;
    g.audio.assign(static_cast<std::size_t>(seconds * kRate), 0.0f);
    std::mt19937 rng(seed);
    const auto beat = 60.0 / bpm;
    const double start = 0.5;
    const double notes[] = {55.0, 73.4, 61.7, 82.4, 49.0, 65.4};
    int index = 0;
    for (double t = start; t < seconds - 0.3; t += beat, ++index) {
        const auto at = static_cast<std::size_t>(t * kRate);
        const auto inBar = index % beatsPerBar;
        g.beats.push_back(t);
        if (inBar == 0) g.downbeats.push_back(t);
        if (beatsPerBar == 4) {
            if (inBar == 0 || inBar == 2) addKick(g.audio, at, 0.7f);
            else addNoise(g.audio, at, 0.35f, 30.0, false, rng);
            addNoise(g.audio, at, 0.08f, 90.0, true, rng);
            addNoise(g.audio, at + static_cast<std::size_t>(beat / 2 * kRate), 0.06f, 90.0, true, rng);
        } else {
            if (inBar == 0) addKick(g.audio, at, 0.7f);
            else addNoise(g.audio, at, 0.25f, 30.0, false, rng);
            addNoise(g.audio, at, 0.08f, 90.0, true, rng);
        }
        if (inBar == 0) {
            const auto note = notes[(index / beatsPerBar) % 6];
            const auto length = static_cast<std::size_t>(beat * beatsPerBar * kRate);
            for (std::size_t i = 0; i < length && at + i < g.audio.size(); ++i) {
                const auto u = static_cast<double>(i) / kRate;
                g.audio[at + i] += 0.25f * static_cast<float>(std::sin(2.0 * M_PI * note * u) * std::exp(-u * 1.5));
            }
        }
    }
    return g;
}

class CollectingSink final : public MarkerSink {
public:
    void emit(const AnalysisMarker& marker) override { markers.push_back(marker); }
    std::vector<AnalysisMarker> markers;
};

void feed(BeatTracker& tracker, const std::vector<float>& audio, MarkerSink& sink, std::int64_t firstSample = 0) {
    analysis::AudioChunk chunk;
    for (std::size_t done = 0; done < audio.size(); done += analysis::kChunkFrames) {
        chunk.firstSample = firstSample + static_cast<std::int64_t>(done);
        chunk.frames = static_cast<std::int32_t>(std::min<std::size_t>(analysis::kChunkFrames, audio.size() - done));
        std::copy_n(audio.begin() + static_cast<std::ptrdiff_t>(done), chunk.frames, chunk.mono);
        tracker.analyze(chunk, sink);
    }
}

std::vector<double> times(const std::vector<AnalysisMarker>& markers, bool downbeatsOnly, double from = 0.0,
                          std::int64_t firstSample = 0) {
    std::vector<double> out;
    for (const auto& m : markers) {
        const auto t = static_cast<double>(m.samplePosition - firstSample) / kRate;
        if (t >= from && (!downbeatsOnly || m.kind == MarkerKind::Downbeat)) out.push_back(t);
    }
    return out;
}

// Beat F-measure: each reference matched at most once within the tolerance.
double fMeasure(const std::vector<double>& detected, std::vector<double> reference, double from = 0.0) {
    reference.erase(std::remove_if(reference.begin(), reference.end(), [&](double t) { return t < from; }),
                    reference.end());
    if (detected.empty() || reference.empty()) return 0.0;
    std::vector<bool> used(reference.size(), false);
    int hits = 0;
    for (const auto d : detected) {
        for (std::size_t i = 0; i < reference.size(); ++i) {
            if (!used[i] && std::abs(reference[i] - d) <= kTolerance) {
                used[i] = true;
                ++hits;
                break;
            }
        }
    }
    const auto precision = static_cast<double>(hits) / static_cast<double>(detected.size());
    const auto recall = static_cast<double>(hits) / static_cast<double>(reference.size());
    return precision + recall > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
}

AnalysisSummary analyseFile(const Groove& g, std::vector<AnalysisMarker>& markers, BeatTrackerSettings settings = {}) {
    BeatTracker tracker(BeatTracker::Mode::File, settings);
    tracker.reset(kRate);
    CollectingSink sink;
    feed(tracker, g.audio, sink);
    tracker.finish(sink);
    markers = sink.markers;
    return tracker.summary();
}

}  // namespace

TEST("beat_analyzer.features.one_frame_per_hop") {
    OnsetFeatures features;
    features.reset(kRate);
    CHECK_EQ(features.hop(), 441);
    std::vector<float> second(static_cast<std::size_t>(kRate), 0.0f);
    features.push(second.data(), static_cast<int>(second.size()));
    // Centred frames: the last full window ends half a window after its centre.
    CHECK(features.endFrame() >= 97 && features.endFrame() <= 100);
    CHECK_EQ(features.bands().size(), features.flux().size() * OnsetFeatures::kBands);
    features.discardBefore(50);
    CHECK_EQ(features.firstFrame(), std::int64_t{50});
}

TEST("beat_analyzer.tempo.impulse_train") {
    std::vector<float> envelope(3000, 0.0f);
    for (std::size_t i = 25; i < envelope.size(); i += 50) envelope[i] = 1.0f;  // 50 frames = 120 BPM at 100 fps
    const auto tempo = estimateTempo(normaliseEnvelope(envelope.data(), envelope.size()), 100.0, 60.0, 200.0);
    CHECK(std::abs(tempo.bpm - 120.0) < 0.5);
    CHECK(tempo.confidence > 0.8f);
}

TEST("beat_analyzer.file.four_four_groove") {
    const auto g = makeGroove(120.0, 4, 30.0);
    std::vector<AnalysisMarker> markers;
    const auto summary = analyseFile(g, markers);
    CHECK(std::abs(summary.tempoBpm - 120.0f) < 1.5f);
    CHECK(summary.tempoConfidence > 0.5f);
    CHECK_EQ(summary.beatsPerBar, 4);
    CHECK(fMeasure(times(markers, false), g.beats) > 0.95);
    CHECK(fMeasure(times(markers, true), g.downbeats) > 0.9);
    for (const auto& m : markers) CHECK(m.beatInBar >= 1 && m.beatInBar <= 4);
}

TEST("beat_analyzer.file.waltz_is_three") {
    const auto g = makeGroove(96.0, 3, 30.0, 11);
    std::vector<AnalysisMarker> markers;
    const auto summary = analyseFile(g, markers);
    CHECK(std::abs(summary.tempoBpm - 96.0f) < 1.5f);
    CHECK_EQ(summary.beatsPerBar, 3);
    CHECK(fMeasure(times(markers, false), g.beats) > 0.95);
    CHECK(fMeasure(times(markers, true), g.downbeats) > 0.9);
}

TEST("beat_analyzer.file.tempo_range_is_respected") {
    const auto g = makeGroove(120.0, 4, 30.0);
    std::atomic<float> low{50.0f}, high{90.0f};
    BeatTrackerSettings settings;
    settings.minBpm = &low;
    settings.maxBpm = &high;
    std::vector<AnalysisMarker> markers;
    const auto summary = analyseFile(g, markers, settings);
    CHECK(std::abs(summary.tempoBpm - 60.0f) < 1.0f);  // the same pulse at half speed
}

TEST("beat_analyzer.file.fixed_meter_finds_phase") {
    const auto g = makeGroove(110.0, 4, 30.0, 3);
    std::atomic<float> fourBeats{3.0f};  // choice index 3 = 4 beats per bar
    BeatTrackerSettings settings;
    settings.meterChoice = &fourBeats;
    std::vector<AnalysisMarker> markers;
    const auto summary = analyseFile(g, markers, settings);
    CHECK_EQ(summary.beatsPerBar, 4);
    CHECK(summary.meterConfidence >= 1.0f);
    CHECK(fMeasure(times(markers, true), g.downbeats) > 0.9);
}

TEST("beat_analyzer.file.silence_and_noise") {
    std::vector<AnalysisMarker> markers;
    Groove silence;
    silence.audio.assign(static_cast<std::size_t>(10 * kRate), 0.0f);
    auto summary = analyseFile(silence, markers);
    CHECK_EQ(summary.tempoBpm, 0.0f);
    CHECK(markers.empty());

    Groove noise;
    noise.audio.resize(static_cast<std::size_t>(20 * kRate));
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> uniform(-0.3f, 0.3f);
    for (auto& s : noise.audio) s = uniform(rng);
    summary = analyseFile(noise, markers);
    CHECK(summary.tempoConfidence < 0.3f);
}

TEST("beat_analyzer.live.locks_on_and_counts_bars") {
    const auto g = makeGroove(124.0, 4, 30.0, 21);
    BeatTracker tracker(BeatTracker::Mode::Live);
    tracker.reset(kRate);
    CollectingSink sink;
    const std::int64_t streamStart = 1'000'000;  // the plugin's stream position when analysis started
    feed(tracker, g.audio, sink, streamStart);
    const auto summary = tracker.summary();
    CHECK(std::abs(summary.tempoBpm - 124.0f) < 2.0f);
    CHECK_EQ(summary.beatsPerBar, 4);
    // After a few seconds of warm-up the live beats are on the grid...
    const auto beats = times(sink.markers, false, 8.0, streamStart);
    CHECK(beats.size() > 30);
    CHECK(fMeasure(beats, g.beats, 8.0) > 0.9);
    // ...in order, never repeated...
    for (std::size_t i = 1; i < sink.markers.size(); ++i)
        CHECK(sink.markers[i].samplePosition > sink.markers[i - 1].samplePosition);
    // ...and the bar count lands on the real downbeats.
    CHECK(fMeasure(times(sink.markers, true, 8.0, streamStart), g.downbeats, 8.0) > 0.8);
}

TEST("beat_analyzer.live.reset_forgets") {
    const auto g = makeGroove(120.0, 4, 10.0);
    BeatTracker tracker(BeatTracker::Mode::Live);
    tracker.reset(kRate);
    CollectingSink sink;
    feed(tracker, g.audio, sink);
    CHECK(tracker.summary().tempoBpm > 0.0f);
    tracker.reset(kRate);
    CHECK_EQ(tracker.summary().tempoBpm, 0.0f);
}
