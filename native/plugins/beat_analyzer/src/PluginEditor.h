// Beat Analyzer editor. The frame, input meter, dropped-file view and live
// marker lane come from plugins/shared/juce/AnalysisEditorBase; this adds
// the tempo readout and the tracker's settings.
#pragma once

#include "AnalysisEditorBase.h"
#include "PluginProcessor.h"

namespace soniclab::beat_analyzer {

class BeatAnalyzerEditor final : public plugin::AnalysisEditorBase {
public:
    explicit BeatAnalyzerEditor(BeatAnalyzerProcessor& owner);

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    void layoutContent(juce::Rectangle<int> area) override;
    void paintContent(juce::Graphics& g) override;
    void tick() override;
    void fileResultChanged() override;

    BeatAnalyzerProcessor& processor_;
    juce::Slider tempoMin_;
    juce::Slider tempoMax_;
    juce::ComboBox meter_;
    juce::TextButton resetLive_{"Reset"};
    juce::TextButton openFile_{"Open"};
    juce::TextButton analyseAgain_{"Analyse again"};
    SliderAttachment tempoMinAttachment_;
    SliderAttachment tempoMaxAttachment_;
    std::unique_ptr<ComboAttachment> meterAttachment_;  // after the combo has its items

    juce::Rectangle<int> railArea_;
    juce::Rectangle<int> inputLabelArea_;
    juce::Rectangle<int> lcdArea_;
    juce::Rectangle<int> settingsArea_;
    juce::Rectangle<int> tempoLabelArea_;
    juce::Rectangle<int> fastLabelArea_;
    juce::Rectangle<int> meterLabelArea_;
    juce::Rectangle<int> fileLabelArea_;
    juce::Rectangle<int> fileInfoArea_;
    juce::Rectangle<int> liveLabelArea_;
    bool fileTimeline_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BeatAnalyzerEditor)
};

}  // namespace soniclab::beat_analyzer
