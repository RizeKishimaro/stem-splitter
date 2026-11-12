#include "PluginEditor.h"

#include <algorithm>

namespace soniclab::beat_analyzer {

using namespace plugin::theme;
using plugin::FileAnalysisResult;

namespace {
void styleTempoSlider(juce::Slider& slider, const juce::String& tooltip) {
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 38, 20);
    slider.setNumDecimalPlacesToDisplay(0);
    slider.setTooltip(tooltip);
}

// The tracker counts quarter-note beats, so a bar of 4 is shown as 4/4.
juce::String meterText(int beatsPerBar) { return beatsPerBar > 0 ? juce::String(beatsPerBar) + "/4" : juce::String("-"); }

// Where the live input is: the last live beat, if it is recent. Bars are
// counted from the first downbeat heard since the analysis started.
struct LivePosition {
    bool active = false;  // a beat within the last two beat periods
    int bar = 0;
    int beat = 0;
};

LivePosition livePosition(const std::vector<soniclab::AnalysisMarker>& markers, double rate, double now, float bpm) {
    LivePosition out;
    if (markers.empty() || rate <= 0.0 || bpm <= 0.0f) return out;
    const auto& last = markers.back();
    const auto period = 60.0 / static_cast<double>(bpm);
    out.active = now - last.seconds(rate) <= 2.0 * period + 0.3;  // 0.3 s: the tracker's reporting delay
    out.beat = last.beatInBar;
    out.bar = static_cast<int>(std::count_if(markers.begin(), markers.end(), [](const soniclab::AnalysisMarker& m) {
        return m.kind == soniclab::MarkerKind::Downbeat;
    }));
    return out;
}

int countKind(const std::vector<soniclab::AnalysisMarker>& markers, soniclab::MarkerKind kind) {
    return static_cast<int>(std::count_if(markers.begin(), markers.end(),
                                          [&](const AnalysisMarker& marker) { return marker.kind == kind; }));
}
}  // namespace

BeatAnalyzerEditor::BeatAnalyzerEditor(BeatAnalyzerProcessor& owner)
    : AnalysisEditorBase(owner, "Beat Analyzer"),
      processor_(owner),
      tempoMinAttachment_(owner.parameters(), BeatAnalyzerProcessor::kTempoMinId, tempoMin_),
      tempoMaxAttachment_(owner.parameters(), BeatAnalyzerProcessor::kTempoMaxId, tempoMax_) {
    styleTempoSlider(tempoMin_, "Slowest tempo to look for, in BPM");
    styleTempoSlider(tempoMax_, "Fastest tempo to look for, in BPM. Narrow the range if the reading is half or double.");
    addAndMakeVisible(tempoMin_);
    addAndMakeVisible(tempoMax_);

    meter_.addItemList(BeatAnalyzerProcessor::meterChoices(), 1);
    meter_.setTooltip("Beats per bar. Auto chooses between 3 and 4.");
    addAndMakeVisible(meter_);
    meterAttachment_ = std::make_unique<ComboAttachment>(owner.parameters(), BeatAnalyzerProcessor::kMeterId, meter_);

    resetLive_.setTooltip("Forget the live analysis and start listening again");
    resetLive_.onClick = [this] { processor_.resetLiveAnalysis(); };
    addAndMakeVisible(resetLive_);

    openFile_.setTooltip("Choose an audio file to analyse");
    openFile_.onClick = [this] { browseForAudioFile(); };
    addAndMakeVisible(openFile_);

    analyseAgain_.setTooltip("Analyse the dropped file again with the current settings");
    analyseAgain_.onClick = [this] {
        const auto file = fileResult().file;
        if (file != juce::File()) processor_.fileAnalysis().start(file);
    };
    addChildComponent(analyseAgain_);

    liveLane().setEmptyText("Play audio through the plugin to detect beats.");
    fileResultChanged();
    setResizable(true, true);
    setResizeLimits(780, 520, 2400, 1600);
    setSize(980, 640);
}

void BeatAnalyzerEditor::layoutContent(juce::Rectangle<int> area) {
    // Inspector on the left, analysis workspace on the right. Both grow
    // with the window; the beat timeline takes whatever height is left.
    railArea_ = area.removeFromLeft(juce::jlimit(220, 280, area.getWidth() / 4));
    area.removeFromLeft(1);
    auto left = railArea_.reduced(12, 10);
    inputLabelArea_ = left.removeFromTop(16);
    inputMeter().setBounds(left.removeFromTop(juce::jmax(96, left.getHeight() / 3)));
    left.removeFromTop(16);
    settingsArea_ = left.removeFromTop(16);
    tempoLabelArea_ = left.removeFromTop(14);
    tempoMin_.setBounds(left.removeFromTop(22));
    left.removeFromTop(8);
    fastLabelArea_ = left.removeFromTop(14);
    tempoMax_.setBounds(left.removeFromTop(22));
    left.removeFromTop(12);
    meterLabelArea_ = left.removeFromTop(14);
    auto controls = left.removeFromTop(24);
    meter_.setBounds(controls.removeFromLeft(96));
    controls.removeFromLeft(8);
    resetLive_.setBounds(controls.removeFromLeft(72));

    auto workspace = area.reduced(12, 10);
    lcdArea_ = workspace.removeFromTop(92);
    workspace.removeFromTop(10);
    fileLabelArea_ = workspace.removeFromTop(26);
    analyseAgain_.setBounds(fileLabelArea_.withLeft(fileLabelArea_.getRight() - 108).reduced(0, 2));
    openFile_.setBounds(fileLabelArea_.withLeft(fileLabelArea_.getRight() - 176).withWidth(60).reduced(0, 2));
    const auto waveHeight = juce::jmax(110, workspace.getHeight() * 2 / 5);
    fileView().setBounds(workspace.removeFromTop(waveHeight));
    fileInfoArea_ = workspace.removeFromTop(22);
    workspace.removeFromTop(8);
    liveLabelArea_ = workspace.removeFromTop(18);
    liveLane().setBounds(workspace);
}

void BeatAnalyzerEditor::tick() {
    const auto& file = fileResult();
    // The lane follows the live input while it has a beat; otherwise it
    // shows the whole timeline of a finished file.
    const auto live = processor_.analysisEnabled() ? processor_.liveSummary() : analysis::AnalysisSummary{};
    const bool hearing =
        livePosition(processor_.markers(), processor_.sampleRate(), analysedSeconds(), live.tempoBpm).active;
    fileTimeline_ = file.status == FileAnalysisResult::Status::Done && !hearing;
    if (fileTimeline_)
        liveLane().showSpan(file.markers, file.sampleRate, 0.0, std::max(file.lengthSeconds(), 0.001));
    repaint(lcdArea_);
    repaint(liveLabelArea_);
    repaint(fileInfoArea_);
}

void BeatAnalyzerEditor::fileResultChanged() {
    const auto& file = fileResult();
    analyseAgain_.setVisible(file.status == FileAnalysisResult::Status::Done ||
                             file.status == FileAnalysisResult::Status::Failed);
    switch (file.status) {
    case FileAnalysisResult::Status::Analysing:
        liveLane().setEmptyText("Analysing the audio file.");
        break;
    case FileAnalysisResult::Status::Failed:
        liveLane().setEmptyText(file.error.isNotEmpty() ? file.error : juce::String("The file could not be analysed."));
        break;
    case FileAnalysisResult::Status::Done:
        liveLane().setEmptyText(file.summary.tempoBpm > 0.0f ? "No beat markers in this file."
                                                             : "No steady beat found in this file.");
        break;
    case FileAnalysisResult::Status::Empty:
        liveLane().setEmptyText("Play audio through the plugin to detect beats.");
        break;
    }
}

void BeatAnalyzerEditor::paintContent(juce::Graphics& g) {
    g.setColour(kPanel);
    g.fillRect(railArea_);
    g.setColour(kLine);
    g.fillRect(railArea_.withLeft(railArea_.getRight()));

    drawSectionLabel(g, "Input", inputLabelArea_);
    drawSectionLabel(g, "Analysis", settingsArea_);
    drawSectionLabel(g, "Slowest", tempoLabelArea_);
    drawSectionLabel(g, "Fastest", fastLabelArea_);
    drawSectionLabel(g, "Beats / bar", meterLabelArea_);

    // Tempo readout: BPM and meter of the live input, or of a finished
    // file when the input is silent.
    plugin::drawLcdFrame(g, lcdArea_);
    const bool enabled = processor_.analysisEnabled();
    const auto live = enabled ? processor_.liveSummary() : analysis::AnalysisSummary{};
    const auto& file = fileResult();
    const bool fileReady = file.status == FileAnalysisResult::Status::Done && file.summary.tempoBpm > 0.0f;
    // Live tempo wins while audio is actually being tracked. A finished file
    // fills the same readout when the input is silent.
    const bool showingFile = live.tempoBpm <= 0.0f && fileReady;
    const auto summary = showingFile ? file.summary : live;
    const bool haveTempo = summary.tempoBpm > 0.0f;
    auto lcd = lcdArea_.reduced(16, 8);

    auto tempo = lcd.removeFromLeft(250);
    auto tempoMeta = tempo.removeFromBottom(18);
    g.setFont(mono(28.0f, true));
    g.setColour(haveTempo ? kLcd : kTextDisabled);
    g.drawText(haveTempo ? juce::String(summary.tempoBpm, 1) : juce::String("No tempo"), tempo,
               juce::Justification::centredLeft, false);
    g.setFont(sans(10.0f, true));
    g.setColour(kTextDim);
    g.drawText(showingFile ? "FILE" : "LIVE", tempoMeta.removeFromLeft(40), juce::Justification::centredLeft, false);
    plugin::drawConfidence(g, tempoMeta.withSizeKeepingCentre(72, 6), haveTempo ? summary.tempoConfidence : -1.0f);

    g.setColour(kLine);
    g.fillRect(lcd.removeFromLeft(1).reduced(0, 8));
    lcd.removeFromLeft(16);

    auto bar = lcd.removeFromLeft(160);
    auto meterMeta = bar.removeFromBottom(18);
    g.setFont(mono(28.0f, true));
    g.setColour(summary.beatsPerBar > 0 ? kAmber : kTextDisabled);
    g.drawText(summary.beatsPerBar > 0 ? meterText(summary.beatsPerBar) : juce::String("--"), bar,
               juce::Justification::centredLeft, false);
    g.setFont(sans(10.0f, true));
    g.setColour(kTextDim);
    g.drawText("METER", meterMeta.removeFromLeft(48), juce::Justification::centredLeft, false);
    plugin::drawConfidence(g, meterMeta.withSizeKeepingCentre(72, 6),
                           summary.beatsPerBar > 0 ? std::min(summary.meterConfidence, summary.downbeatConfidence + 0.4f)
                                                   : -1.0f);

    auto timeRow = lcd.removeFromTop(lcd.getHeight() / 2);
    g.setFont(mono(16.0f, true));
    g.setColour(enabled ? kAmber : kTextDisabled);
    g.drawText(plugin::formatDuration(showingFile ? file.lengthSeconds() : analysedSeconds()), timeRow,
               juce::Justification::centredRight, false);
    g.setFont(sans(11.0f));
    g.setColour(kTextDim);
    const auto dropped = processor_.pipeline().droppedFrames();
    juce::String heard = "no audio yet";
    const auto position = livePosition(processor_.markers(), processor_.sampleRate(), analysedSeconds(), live.tempoBpm);
    if (!enabled) heard = "analysis off";
    else if (dropped > 0) heard = "dropped " + juce::String(static_cast<juce::int64>(dropped));
    else if (showingFile) heard = "from file";
    else if (position.active && position.beat > 0) heard = "bar " + juce::String(position.bar) + "  beat " + juce::String(position.beat);
    else if (position.active) heard = "beat found, finding the bar";
    else if (haveTempo) heard = "listening";
    else if (analysedSeconds() > 0.5) heard = "finding the beat";
    g.drawText(heard, lcd, juce::Justification::centredRight, true);

    drawSectionLabel(g, "Audio file", fileLabelArea_);
    g.setFont(mono(10.0f));
    g.setColour(kTextSoft);
    switch (file.status) {
    case FileAnalysisResult::Status::Empty:
        g.setFont(sans(11.0f));
        g.setColour(kTextDim);
        g.drawText("Live audio is not affected.", fileInfoArea_, juce::Justification::centredLeft);
        break;
    case FileAnalysisResult::Status::Analysing:
        break;
    case FileAnalysisResult::Status::Failed:
        g.setFont(sans(11.0f));
        g.setColour(kTextDim);
        g.drawText("Drop another file to try again.", fileInfoArea_, juce::Justification::centredLeft);
        break;
    case FileAnalysisResult::Status::Done: {
        juce::String beats = "no steady beat found";
        if (file.summary.tempoBpm > 0.0f) {
            beats = juce::String(file.summary.tempoBpm, 1) + " BPM";
            if (file.summary.beatsPerBar > 0) beats += "   " + meterText(file.summary.beatsPerBar);
            const auto downbeats = countKind(file.markers, soniclab::MarkerKind::Downbeat);
            beats += "   " + juce::String(static_cast<int>(file.markers.size())) + " beats";
            if (downbeats > 0) beats += "   " + juce::String(downbeats) + " downbeats";
            beats += juce::String::formatted("   tempo %d%%", juce::roundToInt(file.summary.tempoConfidence * 100.0f));
        }
        g.drawText(plugin::WaveformView::describe(file) + "   " + beats, fileInfoArea_,
                   juce::Justification::centredLeft);
        break;
    }
    }

    drawSectionLabel(g, fileTimeline_ ? "Detected beats" : "Live beats", liveLabelArea_);
    g.setFont(sans(11.0f));
    g.setColour(kTextDim);
    juce::String laneNote = "last 8 s";
    if (file.status == FileAnalysisResult::Status::Analysing)
        laneNote = "analysing";
    else if (fileTimeline_)
        laneNote = plugin::formatDuration(file.lengthSeconds());
    g.drawText(laneNote, liveLabelArea_, juce::Justification::centredRight);
}

}  // namespace soniclab::beat_analyzer
