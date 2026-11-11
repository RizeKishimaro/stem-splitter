// Base class for SonicLab analysis plugins: pass-through effects that
// measure and analyse what they hear.
//
// processBlock leaves the audio untouched. When the "enabled" parameter is
// on it also measures the input level (lock-free AtomicPeak per channel)
// and copies a mono downmix into the AnalysisPipeline's lock-free FIFO.
// Analysis runs on the pipeline's worker thread; editors read levels and
// markers on a timer. processBlock never locks, allocates or waits.
//
// A plugin supplies its extra parameters, its state type and version, and
// a factory for its analyzer (one instance for the live input, one per
// dropped file). Everything else, including bus layouts, state save and
// restore, and dropped-file analysis, is shared.
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>

#include "FileAnalysis.h"
#include "Widgets.h"
#include "soniclab/analysis/analysis_pipeline.h"
#include "soniclab/common/atomic_peak.h"

namespace soniclab::plugin {

class AnalysisProcessorBase : public juce::AudioProcessor {
public:
    static constexpr int kMaxMeteredChannels = plugin::kMaxMeteredChannels;
    static constexpr const char* kEnabledId = "enabled";

    enum class AnalyzerUse { Live, File };
    // Called on the message thread while the plugin is being constructed
    // (parameters exist by then) and whenever a file is dropped.
    using AnalyzerFactory =
        std::function<std::unique_ptr<OnlineAnalyzer>(AnalysisProcessorBase& processor, AnalyzerUse use)>;

    struct Options {
        juce::Identifier stateType;  // root tag of the saved state; never change it
        int stateVersion = 1;
        // The plugin's own parameters; "enabled" is added by the base.
        juce::AudioProcessorValueTreeState::ParameterLayout parameters;
        AnalyzerFactory makeAnalyzer;
    };

    explicit AnalysisProcessorBase(Options options);
    ~AnalysisProcessorBase() override;

    void prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

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
    const std::atomic<float>* rawParameter(const juce::String& id) const;
    bool analysisEnabled() const noexcept { return enabled_->load(std::memory_order_relaxed) >= 0.5f; }
    int stateVersion() const noexcept { return stateVersion_; }

    // Message thread. Input peak per channel since the last call (0..1+),
    // then reset. Channels beyond the current layout read 0.
    std::array<float, kMaxMeteredChannels> takeInputPeaks() noexcept;
    int meteredChannels() const noexcept { return meteredChannels_.load(std::memory_order_relaxed); }

    // Message thread. Moves new markers out of the pipeline and returns all
    // markers seen since the last prepareToPlay or reset (most recent last,
    // capped).
    const std::vector<AnalysisMarker>& collectMarkers();
    const std::vector<AnalysisMarker>& markers() const { return markers_; }
    const analysis::AnalysisPipeline& pipeline() const { return pipeline_; }
    analysis::AnalysisSummary liveSummary() const { return pipeline_.analyzer().summary(); }
    // Any thread. Forgets the live analysis; markers clear on the next collect.
    void resetLiveAnalysis() noexcept;
    double sampleRate() const { return getSampleRate(); }

    // Message thread. Analysis of a file dropped onto the editor; independent
    // of the live input and not saved with the plugin state.
    FileAnalysis& fileAnalysis() { return fileAnalysis_; }

protected:
    // Called before a saved state older than stateVersion() is applied.
    // Parameters missing from the old state keep their defaults.
    virtual void upgradeState(juce::ValueTree& state, int savedVersion);

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout withEnabled(
        juce::AudioProcessorValueTreeState::ParameterLayout layout);

    const juce::Identifier stateType_;
    const int stateVersion_;
    juce::AudioProcessorValueTreeState parameters_;
    std::atomic<float>* enabled_ = nullptr;  // owned by parameters_, read on the audio thread
    AnalyzerFactory makeAnalyzer_;

    std::array<AtomicPeak, kMaxMeteredChannels> inputPeaks_;
    std::atomic<int> meteredChannels_{0};

    analysis::AnalysisPipeline pipeline_;
    // prepareToPlay may run off the message thread (Logic does this), so it
    // only bumps this counter; collectMarkers clears markers_ when it changes.
    std::atomic<std::uint32_t> generation_{0};
    std::uint32_t seenGeneration_ = 0;     // message thread only
    std::vector<AnalysisMarker> markers_;  // message thread only

    FileAnalysis fileAnalysis_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AnalysisProcessorBase)
};

}  // namespace soniclab::plugin
