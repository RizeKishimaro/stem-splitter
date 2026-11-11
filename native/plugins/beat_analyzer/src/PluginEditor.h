// Basic editor: Enabled switch, input level meter, a dropped-file panel and
// a placeholder marker strip. Everything here runs on the message thread
// from a 30 Hz timer; it only reads atomics and lock-free FIFOs that the
// audio thread fills, and FileAnalysis snapshots for a dropped file.
#pragma once

#include <array>
#include <cstdint>

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "PluginProcessor.h"
#include "Theme.h"

namespace soniclab::beat_analyzer {

// Peak meter with UI-side ballistics: instant attack, ~20 dB/s release and
// a 1.5 s peak hold. Displays -60..+6 dBFS with a dB scale underneath and
// channel labels on the left, in the Python mixer's colours (grey bars,
// amber when hot); ``setHint`` text is drawn over the bars while the input
// is silent.
class LevelMeter final : public juce::Component {
public:
    static constexpr float kFloorDb = -60.0f;
    static constexpr float kCeilingDb = 6.0f;

    void update(const std::array<float, BeatAnalyzerProcessor::kMaxMeteredChannels>& peaks, int channels,
                double elapsedSeconds);
    void reset();
    void setHint(const juce::String& hint);
    void paint(juce::Graphics& g) override;

    float displayedDecibels(int channel) const { return levels_[static_cast<std::size_t>(channel)]; }

private:
    std::array<float, BeatAnalyzerProcessor::kMaxMeteredChannels> levels_{kFloorDb, kFloorDb};
    std::array<float, BeatAnalyzerProcessor::kMaxMeteredChannels> holds_{kFloorDb, kFloorDb};
    std::array<double, BeatAnalyzerProcessor::kMaxMeteredChannels> holdAge_{};
    int channels_ = 0;
    juce::String hint_;
};

class BeatAnalyzerEditor final : public juce::AudioProcessorEditor,
                                 public juce::FileDragAndDropTarget,
                                 private juce::Timer {
public:
    explicit BeatAnalyzerEditor(BeatAnalyzerProcessor& owner);
    ~BeatAnalyzerEditor() override;

    void paint(juce::Graphics& g) override;
    void paintOverChildren(juce::Graphics& g) override;
    void resized() override;

    // Dropping an audio file anywhere on the editor analyses it (see
    // FileAnalysis); the live audio is not affected.
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

private:
    void timerCallback() override;
    juce::String inputHint() const;
    void syncFileResult();
    void paintFilePanel(juce::Graphics& g);
    void paintMarkerStrip(juce::Graphics& g);

    BeatAnalyzerProcessor& processor_;
    theme::LookAndFeel lookAndFeel_;  // before the children, so it outlives them
    juce::ToggleButton enabled_{"Enabled"};
    juce::AudioProcessorValueTreeState::ButtonAttachment enabledAttachment_;
    LevelMeter meter_;
    juce::TextButton clearFile_{"Clear"};

    // Painted areas, set in resized().
    juce::Rectangle<int> headerArea_;
    juce::Rectangle<int> inputLabelArea_;
    juce::Rectangle<int> lcdArea_;
    juce::Rectangle<int> fileLabelArea_;
    juce::Rectangle<int> fileArea_;
    juce::Rectangle<int> fileInfoArea_;
    juce::Rectangle<int> markersLabelArea_;
    juce::Rectangle<int> rulerArea_;
    juce::Rectangle<int> stripArea_;
    juce::Rectangle<int> footerArea_;

    FileAnalysisResult file_;  // last snapshot of the processor's FileAnalysis
    std::uint32_t fileVersion_ = 0;
    bool dragHover_ = false;
    double lastTick_ = 0.0;
    // Silence and stalled-audio detection for the input hint.
    std::int64_t lastFramesAnalyzed_ = 0;
    double stalledSeconds_ = 0.0;
    double silentSeconds_ = 0.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BeatAnalyzerEditor)
};

}  // namespace soniclab::beat_analyzer
