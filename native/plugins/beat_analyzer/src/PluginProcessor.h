// SonicLab Beat Analyzer: a pass-through effect that reports tempo, beats,
// downbeats and beats per bar of what it hears, and of audio files dropped
// onto its editor.
//
// Everything a SonicLab analysis plugin shares (pass-through, metering, the
// realtime-to-worker pipeline, state, dropped files) is in
// plugins/shared/juce/AnalysisProcessorBase. This class adds the beat
// tracker and its parameters.
#pragma once

#include <memory>

#include "AnalysisProcessorBase.h"
#include "beat_analyzer/beat_tracker.h"

namespace soniclab::beat_analyzer {

class BeatAnalyzerProcessor final : public plugin::AnalysisProcessorBase {
public:
    // Parameter IDs. Never rename: hosts store automation against them.
    static constexpr const char* kTempoMinId = "tempoMin";
    static constexpr const char* kTempoMaxId = "tempoMax";
    static constexpr const char* kMeterId = "meter";

    BeatAnalyzerProcessor();

    juce::AudioProcessorEditor* createEditor() override;

    // "Auto", "2", ... "7": the choices of the meter parameter.
    static juce::StringArray meterChoices();

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    static std::unique_ptr<analysis::OnlineAnalyzer> makeAnalyzer(plugin::AnalysisProcessorBase& processor,
                                                                  AnalyzerUse use);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BeatAnalyzerProcessor)
};

}  // namespace soniclab::beat_analyzer
