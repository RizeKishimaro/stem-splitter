#include "AnalysisEditorBase.h"

#include <algorithm>

namespace soniclab::plugin {
namespace {
// How long the input must stay at the floor, or the analysed-frame count
// stand still, before the meter says so.
constexpr double kHintAfterSeconds = 1.0;
}  // namespace

using namespace theme;

AnalysisEditorBase::AnalysisEditorBase(AnalysisProcessorBase& owner, const juce::String& pluginTitle)
    : AudioProcessorEditor(owner),
      processor_(owner),
      title_(pluginTitle),
      enabledAttachment_(owner.parameters(), AnalysisProcessorBase::kEnabledId, enabled_) {
    setLookAndFeel(&lookAndFeel_);
    enabled_.setTooltip("Audio always passes through unchanged. Off stops metering and analysis.");
    addAndMakeVisible(enabled_);
    addAndMakeVisible(meter_);
    meter_.update({}, owner.meteredChannels(), 0.0);  // channel labels before the first tick
    fileView_.onClear = [this] { processor_.fileAnalysis().clear(); };
    fileView_.onBrowse = [this] { browseForAudioFile(); };
    fileView_.onFile = [this](const juce::File& file) { analyseFile(file); };
    addAndMakeVisible(fileView_);
    addAndMakeVisible(liveLane_);
    syncFileResult();
    lastTick_ = juce::Time::getMillisecondCounterHiRes() / 1000.0;
    startTimerHz(30);
}

AnalysisEditorBase::~AnalysisEditorBase() {
    fileChooser_.reset();
    stopTimer();
    setLookAndFeel(nullptr);
}

double AnalysisEditorBase::analysedSeconds() const {
    const auto rate = processor_.sampleRate();
    return rate > 0.0 ? static_cast<double>(processor_.pipeline().framesAnalyzed()) / rate : 0.0;
}

void AnalysisEditorBase::resized() {
    auto area = getLocalBounds();
    headerArea_ = area.removeFromTop(kHeaderHeight);
    enabled_.setBounds(headerArea_.withLeft(headerArea_.getRight() - 96).withTrimmedRight(10).reduced(0, 7));
    footerArea_ = area.removeFromBottom(kFooterHeight);
    layoutContent(area.reduced(12, 10));
}

juce::String AnalysisEditorBase::inputHint() const {
    const bool standalone = processor_.wrapperType == juce::AudioProcessor::wrapperType_Standalone;
    if (stalledSeconds_ >= kHintAfterSeconds)
        return standalone ? "No audio is running. Open Options > Audio/MIDI Settings and choose an audio device."
                          : "No audio is running. Start playback or recording in the host.";
    if (silentSeconds_ >= kHintAfterSeconds)
        return standalone ? "No input signal. In Options > Audio/MIDI Settings choose an input and untick "
                            "\"Mute audio input\" (use headphones to avoid feedback)."
                          : "No input signal on this track.";
    return {};
}

void AnalysisEditorBase::refresh() {
    const auto now = juce::Time::getMillisecondCounterHiRes() / 1000.0;
    const auto elapsed = std::clamp(now - lastTick_, 0.0, 0.5);
    lastTick_ = now;
    const auto& markers = processor_.collectMarkers();
    const auto peaks = processor_.takeInputPeaks();

    if (processor_.analysisEnabled()) {
        meter_.update(peaks, processor_.meteredChannels(), elapsed);
        const auto frames = processor_.pipeline().framesAnalyzed();
        stalledSeconds_ = frames == lastFramesAnalyzed_ ? stalledSeconds_ + elapsed : 0.0;
        lastFramesAnalyzed_ = frames;
        const auto floorGain = juce::Decibels::decibelsToGain(LevelMeter::kFloorDb);
        const bool silent = std::all_of(peaks.begin(), peaks.end(), [&](float p) { return p <= floorGain; });
        silentSeconds_ = silent ? silentSeconds_ + elapsed : 0.0;
        meter_.setHint(inputHint());
    } else {
        meter_.reset();
        stalledSeconds_ = silentSeconds_ = 0.0;
        meter_.setHint("Disabled: audio passes through, nothing is measured.");
    }
    liveLane_.update(markers, processor_.sampleRate(), analysedSeconds());

    if (processor_.fileAnalysis().version() != fileVersion_) syncFileResult();
    else if (fileResult().status == FileAnalysisResult::Status::Analysing)
        fileView_.setProgress(processor_.fileAnalysis().progress());

    tick();
    repaint(footerArea_);
}

void AnalysisEditorBase::syncFileResult() {
    auto& analysis = processor_.fileAnalysis();
    fileVersion_ = analysis.version();
    fileView_.setResult(analysis.snapshot());
    fileView_.setProgress(analysis.progress());
    fileResultChanged();
    repaint();
}

bool AnalysisEditorBase::isInterestedInFileDrag(const juce::StringArray& files) {
    for (const auto& path : files)
        if (FileAnalysis::canAnalyse(juce::File(path))) return true;
    return false;
}

void AnalysisEditorBase::fileDragEnter(const juce::StringArray&, int, int) {
    dragHover_ = true;
    repaint();
}

void AnalysisEditorBase::fileDragExit(const juce::StringArray&) {
    dragHover_ = false;
    repaint();
}

void AnalysisEditorBase::analyseFile(const juce::File& file) {
    if (!FileAnalysis::canAnalyse(file)) return;
    processor_.fileAnalysis().start(file);
    syncFileResult();
}

void AnalysisEditorBase::browseForAudioFile() {
    fileChooser_ = std::make_unique<juce::FileChooser>("Choose an audio file to analyse", juce::File(),
                                                       FileAnalysis::chooserPattern());
    const juce::Component::SafePointer<AnalysisEditorBase> safe(this);
    fileChooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                              [safe](const juce::FileChooser& chooser) {
                                  if (safe != nullptr) safe->analyseFile(chooser.getResult());
                              });
}

void AnalysisEditorBase::filesDropped(const juce::StringArray& files, int, int) {
    dragHover_ = false;
    for (const auto& path : files) {
        const juce::File file(path);
        if (FileAnalysis::canAnalyse(file)) {
            analyseFile(file);  // only the first usable file
            return;
        }
    }
}

void AnalysisEditorBase::paintOverChildren(juce::Graphics& g) {
    if (!dragHover_) return;
    const auto bounds = getLocalBounds().toFloat().reduced(3.0f);
    g.setColour(kWindow.withAlpha(0.85f));
    g.fillRect(bounds);
    g.setColour(kAmber);
    g.drawRect(bounds, 1.5f);
    g.setFont(sans(14.0f, true));
    g.drawText("Drop to analyse this file", bounds, juce::Justification::centred);
}

void AnalysisEditorBase::paint(juce::Graphics& g) {
    g.fillAll(kWindow);

    // Header bar, like the transport toolbar: wordmark left, Enabled right.
    g.setColour(kRuler);
    g.fillRect(headerArea_);
    g.setColour(kLine);
    g.fillRect(headerArea_.withTop(headerArea_.getBottom() - 1));
    auto title = headerArea_.withTrimmedLeft(12);
    g.setFont(sans(12.0f, true));
    g.setColour(juce::Colour(0xffb7bcc4));
    const juce::String wordmark = "SonicLab";
    g.drawText(wordmark, title, juce::Justification::centredLeft);
    const auto wordmarkWidth = juce::roundToInt(juce::GlyphArrangement::getStringWidth(sans(12.0f, true), wordmark));
    g.setFont(sans(12.0f));
    g.setColour(kTextDim);
    g.drawText(title_, title.withTrimmedLeft(wordmarkWidth + 8), juce::Justification::centredLeft);

    paintContent(g);

    // Footer, like the status bar.
    g.setColour(kRuler);
    g.fillRect(footerArea_);
    g.setColour(kLine);
    g.fillRect(footerArea_.withHeight(1));
    auto footer = footerArea_.reduced(10, 0);
    g.setFont(sans(11.0f));
    g.setColour(kTextDim);
    g.drawText(footerText(), footer, juce::Justification::centredLeft);
    const auto rate = processor_.sampleRate();
    if (rate > 0.0)
        g.drawText(juce::String::formatted("%.1f kHz", rate / 1000.0) + "     " +
                       describeChannels(processor_.getTotalNumInputChannels()),
                   footer, juce::Justification::centredRight);
}

}  // namespace soniclab::plugin
