// SonicLab Beat Analyzer: a pass-through effect that analyses what it hears.
//
// processBlock leaves the audio untouched. When the "enabled" parameter is
// on it also measures the input level (lock-free AtomicPeak per channel)
// and copies a mono downmix into the AnalysisPipeline's lock-free FIFO.
// Analysis runs on the pipeline's worker thread; the editor reads levels
// and markers on a timer. processBlock never locks, allocates or waits.
#pragma once

#include <array>
#include <atomic>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>

#include "FileAnalysis.h"
#include "beat_analyzer/analysis_pipeline.h"
#include "soniclab/common/atomic_peak.h"

namespace soniclab::beat_analyzer {

class BeatAnalyzerProcessor final : public juce::AudioProcessor {
public:
    static constexpr int kMaxMeteredChannels = 2;
    static constexpr const char* kEnabledId = "enabled";

    BeatAnalyzerProcessor();
    ~BeatAnalyzerProcessor() override;

    void prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& parameters() { return parameters_; }
    bool analysisEnabled() const noexcept { return enabled_->load(std::memory_order_relaxed) >= 0.5f; }

    // Message thread. Input peak per channel since the last call (0..1+),
    // then reset. Channels beyond the current layout read 0.
    std::array<float, kMaxMeteredChannels> takeInputPeaks() noexcept;
    int meteredChannels() const noexcept { return meteredChannels_.load(std::memory_order_relaxed); }

    // Message thread. Moves new markers out of the pipeline and returns all
    // markers seen since the last prepareToPlay (most recent last, capped).
    const std::vector<AnalysisMarker>& collectMarkers();
    const std::vector<AnalysisMarker>& markers() const { return markers_; }
    const AnalysisPipeline& pipeline() const { return pipeline_; }
    double sampleRate() const { return getSampleRate(); }

    // Message thread. Analysis of a file dropped onto the editor; independent
    // of the live input and not saved with the plugin state.
    FileAnalysis& fileAnalysis() { return fileAnalysis_; }

    // The analyzer used for both the live input and dropped files.
    static std::unique_ptr<OnlineAnalyzer> makeAnalyzer();

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    juce::AudioProcessorValueTreeState parameters_;
    std::atomic<float>* enabled_ = nullptr;  // owned by parameters_, read on the audio thread

    std::array<AtomicPeak, kMaxMeteredChannels> inputPeaks_;
    std::atomic<int> meteredChannels_{0};

    AnalysisPipeline pipeline_;
    // prepareToPlay may run off the message thread (Logic does this), so it
    // only bumps this counter; collectMarkers clears markers_ when it changes.
    std::atomic<std::uint32_t> generation_{0};
    std::uint32_t seenGeneration_ = 0;      // message thread only
    std::vector<AnalysisMarker> markers_;  // message thread only

    FileAnalysis fileAnalysis_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BeatAnalyzerProcessor)
};

}  // namespace soniclab::beat_analyzer
