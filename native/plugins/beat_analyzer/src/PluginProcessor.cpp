#include "PluginProcessor.h"

#include "PluginEditor.h"

namespace soniclab::beat_analyzer {
namespace {
// 1: version only (Phase 1); 2: + enabled; 3: + tempo range and meter.
// Older states load with the new parameters at their defaults.
constexpr int kStateVersion = 3;
}  // namespace

juce::StringArray BeatAnalyzerProcessor::meterChoices() { return {"Auto", "2", "3", "4", "5", "6", "7"}; }

juce::AudioProcessorValueTreeState::ParameterLayout BeatAnalyzerProcessor::createLayout() {
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    const auto bpm = juce::AudioParameterFloatAttributes().withLabel("BPM");
    // The tracker looks for the beat between these two tempos. Narrow the
    // range to settle half-time / double-time readings.
    layout.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{kTempoMinId, 1}, "Slowest tempo",
                                                           juce::NormalisableRange<float>(40.0f, 200.0f, 1.0f), 60.0f,
                                                           bpm));
    layout.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{kTempoMaxId, 1}, "Fastest tempo",
                                                           juce::NormalisableRange<float>(60.0f, 300.0f, 1.0f), 200.0f,
                                                           bpm));
    // Auto chooses between 3 and 4 beats per bar; a number fixes the bar
    // length and only the downbeat is searched for.
    layout.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{kMeterId, 1}, "Beats per bar",
                                                            meterChoices(), 0));
    return layout;
}

std::unique_ptr<analysis::OnlineAnalyzer> BeatAnalyzerProcessor::makeAnalyzer(plugin::AnalysisProcessorBase& processor,
                                                                              AnalyzerUse use) {
    BeatTrackerSettings settings;
    settings.minBpm = processor.rawParameter(kTempoMinId);
    settings.maxBpm = processor.rawParameter(kTempoMaxId);
    settings.meterChoice = processor.rawParameter(kMeterId);
    return std::make_unique<BeatTracker>(use == AnalyzerUse::Live ? BeatTracker::Mode::Live : BeatTracker::Mode::File,
                                         settings);
}

BeatAnalyzerProcessor::BeatAnalyzerProcessor()
    : AnalysisProcessorBase({juce::Identifier("SonicLabBeatAnalyzer"), kStateVersion, createLayout(),
                             &BeatAnalyzerProcessor::makeAnalyzer}) {}

juce::AudioProcessorEditor* BeatAnalyzerProcessor::createEditor() { return new BeatAnalyzerEditor(*this); }

}  // namespace soniclab::beat_analyzer

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new soniclab::beat_analyzer::BeatAnalyzerProcessor();
}
