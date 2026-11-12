// Runs SonicLab's native beat tracker on raw audio, for comparing it with
// other analysers (see scripts/compare_beats.py). Development tool only.
//
//   soniclab_beat_cli <sample rate> [min bpm] [max bpm] [meter choice] [live] < mono_float32.raw
//
// "live" runs the tracker the way the plugin runs it on incoming audio
// (rolling window, beats reported as they are found) instead of on the whole file.
//
// Reads little-endian float32 mono samples from stdin and prints one JSON
// object: tempo, confidences, beats per bar and [seconds, beat in bar] pairs.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "beat_analyzer/beat_tracker.h"

using namespace soniclab;
using namespace soniclab::beat_analyzer;

namespace {
class Collect final : public MarkerSink {
public:
    void emit(const AnalysisMarker& marker) override { markers.push_back(marker); }
    std::vector<AnalysisMarker> markers;
};
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <sample rate> [min bpm] [max bpm] [meter choice] < mono_float32.raw\n", argv[0]);
        return 2;
    }
    const double rate = std::atof(argv[1]);
    std::atomic<float> minBpm{argc > 2 ? static_cast<float>(std::atof(argv[2])) : 60.0f};
    std::atomic<float> maxBpm{argc > 3 ? static_cast<float>(std::atof(argv[3])) : 200.0f};
    std::atomic<float> meter{argc > 4 ? static_cast<float>(std::atof(argv[4])) : 0.0f};
    if (rate <= 0.0) return 2;

    BeatTrackerSettings settings;
    settings.minBpm = &minBpm;
    settings.maxBpm = &maxBpm;
    settings.meterChoice = &meter;
    const bool live = argc > 5 && std::string(argv[5]) == "live";
    BeatTracker tracker(live ? BeatTracker::Mode::Live : BeatTracker::Mode::File, settings);
    tracker.reset(rate);
    Collect sink;
    analysis::AudioChunk chunk;
    std::int64_t position = 0;
    for (;;) {
        const auto read = std::fread(chunk.mono, sizeof(float), analysis::kChunkFrames, stdin);
        if (read == 0) break;
        chunk.firstSample = position;
        chunk.frames = static_cast<std::int32_t>(read);
        tracker.analyze(chunk, sink);
        position += static_cast<std::int64_t>(read);
    }
    tracker.finish(sink);

    auto summary = tracker.summary();
    std::printf("{\"bpm\": %.3f, \"tempo_confidence\": %.3f, \"beats_per_bar\": %d, \"meter_confidence\": %.3f, "
                "\"downbeat_confidence\": %.3f, \"beats\": [",
                static_cast<double>(summary.tempoBpm), static_cast<double>(summary.tempoConfidence), summary.beatsPerBar,
                static_cast<double>(summary.meterConfidence), static_cast<double>(summary.downbeatConfidence));
    for (std::size_t i = 0; i < sink.markers.size(); ++i)
        std::printf("%s[%.4f, %d]", i == 0 ? "" : ", ", sink.markers[i].seconds(rate), sink.markers[i].beatInBar);
    std::printf("]}\n");
    return 0;
}
