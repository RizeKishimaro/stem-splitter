// Views of analysis results shared by SonicLab plugins. Message thread only.
//
// WaveformView  a dropped file, drawn like a clip on the Python timeline,
//               with the analyzer's markers on top
// MarkerLane    the last few seconds of live markers on a ruler
#pragma once

#include <functional>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "FileAnalysis.h"
#include "Theme.h"

namespace soniclab::plugin {

// Colour of a marker line: downbeats and bar lines amber, other beats pale,
// onsets blue (the timeline's "early" onset colour).
juce::Colour markerColour(const AnalysisMarker& marker);

class WaveformView final : public juce::Component, public juce::FileDragAndDropTarget {
public:
    WaveformView();

    std::function<void()> onClear;  // the Clear button; hidden while empty
    std::function<void()> onBrowse;  // a click on the panel
    std::function<void(const juce::File&)> onFile;  // a file dropped on the panel

    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    void mouseDown(const juce::MouseEvent& event) override;

    void setResult(const FileAnalysisResult& result);
    void setProgress(float progress);  // 0..1 while analysing
    const FileAnalysisResult& result() const { return result_; }

    void paint(juce::Graphics& g) override;
    void resized() override;

    // "0:02.0   48.0 kHz   Stereo   peak -6.0 dBFS" for the line under the view.
    static juce::String describe(const FileAnalysisResult& result);

private:
    FileAnalysisResult result_;
    float progress_ = 0.0f;
    juce::TextButton clear_{"Clear"};
};

class MarkerLane final : public juce::Component {
public:
    static constexpr int kRulerHeight = 20;

    void setWindowSeconds(double seconds) { liveWindow_ = seconds; }
    void setEmptyText(const juce::String& text) { emptyText_ = text; }
    // Keeps the markers that fall into the last ``windowSeconds`` before
    // ``endSeconds``. This is the rolling live display.
    void update(const std::vector<AnalysisMarker>& markers, double sampleRate, double endSeconds);
    // Shows every marker from ``startSeconds`` to ``endSeconds``. Used for a
    // finished file analysis, which is a whole timeline rather than a live window.
    void showSpan(const std::vector<AnalysisMarker>& markers, double sampleRate, double startSeconds,
                  double endSeconds);

    void paint(juce::Graphics& g) override;

private:
    struct Visible {
        double seconds;
        AnalysisMarker marker;
    };
    std::vector<Visible> visible_;
    double liveWindow_ = 8.0;
    double window_ = 8.0;
    double origin_ = 0.0;
    double end_ = 0.0;
    bool absolute_ = false;  // file timeline: times from the start, not "seconds ago"
    juce::String emptyText_ = "No markers yet.";
};

}  // namespace soniclab::plugin
