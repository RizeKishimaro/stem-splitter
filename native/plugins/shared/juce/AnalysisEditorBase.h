// Base editor for SonicLab analysis plugins: the shared frame and the
// parts every analysis plugin shows.
//
//   header   "SonicLab <plugin name>" wordmark, Enabled switch on the right
//   footer   status line, sample rate and channel layout
//   shared   input meter with silence hints, dropped-file view (drag and
//            drop anywhere on the editor), live marker lane
//
// A plugin lays out and paints the content area between header and footer
// and places the shared components where it wants them. Everything runs on
// the message thread from a 30 Hz timer and only reads atomics, lock-free
// FIFOs and FileAnalysis snapshots.
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "AnalysisProcessorBase.h"
#include "AnalysisViews.h"
#include "Theme.h"
#include "Widgets.h"

namespace soniclab::plugin {

class AnalysisEditorBase : public juce::AudioProcessorEditor,
                           public juce::FileDragAndDropTarget,
                           private juce::Timer {
public:
    static constexpr int kHeaderHeight = 34;
    static constexpr int kFooterHeight = 24;

    AnalysisEditorBase(AnalysisProcessorBase& owner, const juce::String& pluginTitle);
    ~AnalysisEditorBase() override;

    void paint(juce::Graphics& g) final;
    void paintOverChildren(juce::Graphics& g) override;
    void resized() final;

    // Clicking the file panel, or dropping an audio file anywhere on the
    // editor, analyses it (see FileAnalysis). The live audio is not affected.
    void browseForAudioFile();
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

    // What the 30 Hz timer does: pull levels, markers and file results from
    // the processor and repaint. Public so a host-less test can step it.
    void refresh();

protected:
    // ``area`` is the space between header and footer, already inset.
    virtual void layoutContent(juce::Rectangle<int> area) = 0;
    virtual void paintContent(juce::Graphics& g) = 0;
    // After the shared refresh, 30 times a second.
    virtual void tick() {}
    virtual void fileResultChanged() {}
    virtual juce::String footerText() const { return "Audio passes through unchanged"; }

    AnalysisProcessorBase& processorBase() { return processor_; }
    const AnalysisProcessorBase& processorBase() const { return processor_; }
    LevelMeter& inputMeter() { return meter_; }
    WaveformView& fileView() { return fileView_; }
    MarkerLane& liveLane() { return liveLane_; }
    const FileAnalysisResult& fileResult() const { return fileView_.result(); }
    // Seconds of live audio analysed since the last reset or prepareToPlay.
    double analysedSeconds() const;

private:
    void timerCallback() override { refresh(); }
    juce::String inputHint() const;
    void syncFileResult();
    void analyseFile(const juce::File& file);

    AnalysisProcessorBase& processor_;
    const juce::String title_;
    theme::LookAndFeel lookAndFeel_;  // before the children, so it outlives them
    juce::ToggleButton enabled_{"Enabled"};
    juce::AudioProcessorValueTreeState::ButtonAttachment enabledAttachment_;
    LevelMeter meter_;
    WaveformView fileView_;
    MarkerLane liveLane_;
    std::unique_ptr<juce::FileChooser> fileChooser_;

    juce::Rectangle<int> headerArea_;
    juce::Rectangle<int> footerArea_;
    std::uint32_t fileVersion_ = 0;
    bool dragHover_ = false;
    double lastTick_ = 0.0;
    // Silence and stalled-audio detection for the input hint.
    std::int64_t lastFramesAnalyzed_ = 0;
    double stalledSeconds_ = 0.0;
    double silentSeconds_ = 0.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AnalysisEditorBase)
};

}  // namespace soniclab::plugin
