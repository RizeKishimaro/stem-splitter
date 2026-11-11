#include "PluginEditor.h"

#include <algorithm>

namespace soniclab::beat_analyzer {
namespace {
constexpr double kVisibleSeconds = 8.0;
constexpr float kReleaseDbPerSecond = 20.0f;
constexpr double kHoldSeconds = 1.5;
// How long the input must stay at the floor, or the analysed-frame count
// stand still, before the meter says so.
constexpr double kHintAfterSeconds = 1.0;
constexpr float kLabelWidth = 18.0f;
constexpr float kScaleHeight = 14.0f;
// meter.py turns a bar amber from 0.92 of full scale.
const float kHotDb = juce::Decibels::gainToDecibels(0.92f);

using namespace theme;

juce::String formatDuration(double seconds) {
    const auto minutes = static_cast<int>(seconds / 60.0);
    return juce::String::formatted("%d:%04.1f", minutes, seconds - 60.0 * minutes);
}

juce::String describeChannels(int channels) {
    return channels == 1 ? juce::String("Mono") : channels == 2 ? juce::String("Stereo") : juce::String(channels) + " ch";
}

void drawHorizontalRule(juce::Graphics& g, juce::Rectangle<int> area, bool atBottom) {
    g.setColour(kLine);
    g.fillRect(atBottom ? area.removeFromBottom(1) : area.removeFromTop(1));
}
}  // namespace

// ------------------------------------------------------------------ meter

void LevelMeter::update(const std::array<float, BeatAnalyzerProcessor::kMaxMeteredChannels>& peaks, int channels,
                        double elapsedSeconds) {
    channels_ = std::clamp(channels, 0, BeatAnalyzerProcessor::kMaxMeteredChannels);
    const auto release = kReleaseDbPerSecond * static_cast<float>(elapsedSeconds);
    for (std::size_t ch = 0; ch < levels_.size(); ++ch) {
        const auto db = std::clamp(juce::Decibels::gainToDecibels(peaks[ch], kFloorDb), kFloorDb, kCeilingDb);
        levels_[ch] = db >= levels_[ch] ? db : std::max(db, levels_[ch] - release);
        holdAge_[ch] += elapsedSeconds;
        if (db >= holds_[ch] || holdAge_[ch] > kHoldSeconds) {
            holds_[ch] = db;
            holdAge_[ch] = 0.0;
        }
    }
    repaint();
}

void LevelMeter::reset() {
    levels_.fill(kFloorDb);
    holds_.fill(kFloorDb);
    holdAge_.fill(0.0);
    repaint();
}

void LevelMeter::setHint(const juce::String& hint) {
    if (hint == hint_) return;
    hint_ = hint;
    repaint();
}

void LevelMeter::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    auto scale = bounds.removeFromBottom(kScaleHeight);
    auto labels = bounds.removeFromLeft(kLabelWidth);
    scale.removeFromLeft(kLabelWidth);

    g.setColour(kWell);
    g.fillRect(bounds);

    const auto toX = [&](float db) {
        return bounds.getX() + bounds.getWidth() * (db - kFloorDb) / (kCeilingDb - kFloorDb);
    };

    // dB scale: grid lines inside the well, numbers underneath.
    g.setFont(mono(9.0f));
    for (const int tick : {-60, -48, -36, -24, -18, -12, -6, -3, 0, 6}) {
        const auto x = toX(static_cast<float>(tick));
        g.setColour(tick == 0 ? kHandle : kLineSoft);
        g.drawVerticalLine(juce::roundToInt(x), bounds.getY(), bounds.getBottom());
        const auto text = tick > 0 ? "+" + juce::String(tick) : juce::String(tick);
        const bool first = tick == static_cast<int>(kFloorDb);
        const bool last = tick == static_cast<int>(kCeilingDb);
        const auto justification = first  ? juce::Justification::centredLeft
                                   : last ? juce::Justification::centredRight
                                          : juce::Justification::centred;
        auto box = juce::Rectangle<float>(x - 15.0f, scale.getY(), 30.0f, scale.getHeight());
        if (first) box.setX(x);
        if (last) box.setRight(x);
        g.setColour(tick == 0 ? kTextSoft : kTextDim);
        g.drawText(text, box, justification, false);
    }

    const auto shown = std::max(channels_, 1);
    const auto gap = 3.0f;
    const auto barHeight = (bounds.getHeight() - gap * static_cast<float>(shown + 1)) / static_cast<float>(shown);
    for (int ch = 0; ch < shown; ++ch) {
        const auto i = static_cast<std::size_t>(ch);
        const auto y = bounds.getY() + gap + static_cast<float>(ch) * (barHeight + gap);
        g.setColour(kTextDim);
        g.setFont(sans(10.0f, true));
        const auto name = shown == 1 ? "M" : (ch == 0 ? "L" : "R");
        g.drawText(name, juce::Rectangle<float>(labels.getX(), y, kLabelWidth - 5.0f, barHeight),
                   juce::Justification::centredRight, false);
        if (channels_ == 0) continue;

        const auto level = levels_[i];
        g.setColour(level >= kHotDb ? kMeterHot : kMeter);
        g.fillRect(juce::Rectangle<float>(bounds.getX(), y, toX(level) - bounds.getX(), barHeight));
        if (holds_[i] > kFloorDb) {
            g.setColour(holds_[i] >= kHotDb ? kAmber : kText);
            g.fillRect(juce::Rectangle<float>(toX(holds_[i]) - 1.0f, y, 2.0f, barHeight));
        }
    }

    if (hint_.isNotEmpty()) {
        g.setColour(kWell.withAlpha(0.88f));
        g.fillRect(bounds);
        g.setColour(kTextSoft);
        g.setFont(sans(11.0f));
        g.drawFittedText(hint_, bounds.reduced(8.0f, 2.0f).toNearestInt(), juce::Justification::centred, 2);
    }
}

// ----------------------------------------------------------------- editor

BeatAnalyzerEditor::BeatAnalyzerEditor(BeatAnalyzerProcessor& owner)
    : AudioProcessorEditor(owner),
      processor_(owner),
      enabledAttachment_(owner.parameters(), BeatAnalyzerProcessor::kEnabledId, enabled_) {
    setLookAndFeel(&lookAndFeel_);
    enabled_.setTooltip("Audio always passes through unchanged. Off stops metering and analysis.");
    addAndMakeVisible(enabled_);
    addAndMakeVisible(meter_);
    meter_.update({}, owner.meteredChannels(), 0.0);  // channel labels before the first tick
    clearFile_.setTooltip("Forget the dropped file");
    clearFile_.onClick = [this] { processor_.fileAnalysis().clear(); };
    addChildComponent(clearFile_);
    syncFileResult();
    setSize(560, 440);
    lastTick_ = juce::Time::getMillisecondCounterHiRes() / 1000.0;
    startTimerHz(30);
}

BeatAnalyzerEditor::~BeatAnalyzerEditor() {
    stopTimer();
    setLookAndFeel(nullptr);
}

void BeatAnalyzerEditor::resized() {
    auto area = getLocalBounds();
    headerArea_ = area.removeFromTop(34);
    enabled_.setBounds(headerArea_.withLeft(headerArea_.getRight() - 96).withTrimmedRight(10).reduced(0, 7));
    footerArea_ = area.removeFromBottom(24);

    area.reduce(12, 10);
    inputLabelArea_ = area.removeFromTop(18);
    meter_.setBounds(area.removeFromTop(52));
    area.removeFromTop(6);
    lcdArea_ = area.removeFromTop(28);

    area.removeFromTop(12);
    fileLabelArea_ = area.removeFromTop(22);
    clearFile_.setBounds(fileLabelArea_.withLeft(fileLabelArea_.getRight() - 52).reduced(0, 2));
    fileArea_ = area.removeFromTop(80);
    fileInfoArea_ = area.removeFromTop(18);

    area.removeFromTop(10);
    markersLabelArea_ = area.removeFromTop(18);
    rulerArea_ = area.removeFromTop(20);
    stripArea_ = area;
}

juce::String BeatAnalyzerEditor::inputHint() const {
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

void BeatAnalyzerEditor::timerCallback() {
    const auto now = juce::Time::getMillisecondCounterHiRes() / 1000.0;
    const auto elapsed = std::clamp(now - lastTick_, 0.0, 0.5);
    lastTick_ = now;
    processor_.collectMarkers();
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
    const bool fileChanged = processor_.fileAnalysis().version() != fileVersion_;
    if (fileChanged) syncFileResult();
    if (fileChanged || file_.status == FileAnalysisResult::Status::Analysing)
        repaint(fileLabelArea_.getUnion(fileInfoArea_));
    repaint(lcdArea_);
    repaint(markersLabelArea_.getUnion(stripArea_));
    repaint(footerArea_);
}

void BeatAnalyzerEditor::syncFileResult() {
    auto& analysis = processor_.fileAnalysis();
    fileVersion_ = analysis.version();
    file_ = analysis.snapshot();
    clearFile_.setVisible(file_.status != FileAnalysisResult::Status::Empty);
}

bool BeatAnalyzerEditor::isInterestedInFileDrag(const juce::StringArray& files) {
    for (const auto& path : files)
        if (FileAnalysis::canAnalyse(juce::File(path))) return true;
    return false;
}

void BeatAnalyzerEditor::fileDragEnter(const juce::StringArray&, int, int) {
    dragHover_ = true;
    repaint();
}

void BeatAnalyzerEditor::fileDragExit(const juce::StringArray&) {
    dragHover_ = false;
    repaint();
}

void BeatAnalyzerEditor::filesDropped(const juce::StringArray& files, int, int) {
    dragHover_ = false;
    for (const auto& path : files) {
        const juce::File file(path);
        if (FileAnalysis::canAnalyse(file)) {
            processor_.fileAnalysis().start(file);  // only the first usable file
            break;
        }
    }
    syncFileResult();
    repaint();
}

void BeatAnalyzerEditor::paintOverChildren(juce::Graphics& g) {
    if (!dragHover_) return;
    const auto bounds = getLocalBounds().toFloat().reduced(3.0f);
    g.setColour(kWindow.withAlpha(0.85f));
    g.fillRect(bounds);
    g.setColour(kAmber);
    g.drawRect(bounds, 1.5f);
    g.setFont(sans(14.0f, true));
    g.drawText("Drop to analyse this file", bounds, juce::Justification::centred);
}

void BeatAnalyzerEditor::paint(juce::Graphics& g) {
    g.fillAll(kWindow);

    // Header bar, like the transport toolbar: wordmark left, Enabled right.
    g.setColour(kRuler);
    g.fillRect(headerArea_);
    drawHorizontalRule(g, headerArea_, true);
    auto title = headerArea_.withTrimmedLeft(12);
    g.setFont(sans(12.0f, true));
    g.setColour(juce::Colour(0xffb7bcc4));
    const juce::String wordmark = "SonicLab";
    g.drawText(wordmark, title, juce::Justification::centredLeft);
    const auto wordmarkWidth = juce::roundToInt(juce::GlyphArrangement::getStringWidth(sans(12.0f, true), wordmark));
    g.setFont(sans(12.0f));
    g.setColour(kTextDim);
    g.drawText("Beat Analyzer", title.withTrimmedLeft(wordmarkWidth + 8), juce::Justification::centredLeft);

    // Input: section label, meter (a child), then an LCD readout.
    drawSectionLabel(g, "Input", inputLabelArea_);
    g.setColour(kLcdBackground);
    g.fillRect(lcdArea_);
    g.setColour(kLine);
    g.drawRect(lcdArea_, 1);
    auto lcd = lcdArea_.reduced(10, 0);
    const bool enabled = processor_.analysisEnabled();
    const auto& pipeline = processor_.pipeline();
    const auto rate = processor_.sampleRate();
    const auto analysed = rate > 0.0 ? static_cast<double>(pipeline.framesAnalyzed()) / rate : 0.0;

    juce::String levels;
    if (!enabled) levels = "--.-";
    else if (processor_.meteredChannels() > 1)
        levels = juce::String::formatted("%5.1f %5.1f", meter_.displayedDecibels(0), meter_.displayedDecibels(1));
    else levels = juce::String::formatted("%5.1f", meter_.displayedDecibels(0));
    g.setFont(mono(15.0f, true));
    g.setColour(enabled ? kLcd : kTextDisabled);
    const auto levelsWidth = juce::roundToInt(juce::GlyphArrangement::getStringWidth(mono(15.0f, true), levels));
    g.drawText(levels, lcd.removeFromLeft(levelsWidth + 2), juce::Justification::centredLeft, false);
    g.setFont(sans(10.0f));
    g.setColour(kTextDim);
    g.drawText("dBFS", lcd.removeFromLeft(40).withTrimmedLeft(6), juce::Justification::centredLeft, false);
    g.setFont(mono(13.0f, true));
    g.setColour(enabled ? kAmber : kTextDisabled);
    g.drawText(formatDuration(analysed), lcd.removeFromLeft(80), juce::Justification::centredLeft, false);
    g.setFont(sans(11.0f));
    g.setColour(kTextDim);
    g.drawText(juce::String::formatted("markers %d   dropped %llu", static_cast<int>(processor_.markers().size()),
                                       static_cast<unsigned long long>(pipeline.droppedFrames())),
               lcd, juce::Justification::centredRight, false);

    paintFilePanel(g);
    paintMarkerStrip(g);

    // Footer, like the status bar.
    g.setColour(kRuler);
    g.fillRect(footerArea_);
    drawHorizontalRule(g, footerArea_, false);
    auto footer = footerArea_.reduced(10, 0);
    g.setFont(sans(11.0f));
    g.setColour(kTextDim);
    g.drawText("Audio passes through unchanged", footer, juce::Justification::centredLeft);
    const auto channels = processor_.getTotalNumInputChannels();
    if (rate > 0.0)
        g.drawText(juce::String::formatted("%.1f kHz", rate / 1000.0) + "     " + describeChannels(channels), footer,
                   juce::Justification::centredRight);
}

void BeatAnalyzerEditor::paintFilePanel(juce::Graphics& g) {
    using Status = FileAnalysisResult::Status;
    if (processor_.fileAnalysis().version() != fileVersion_) syncFileResult();

    drawSectionLabel(g, "Audio file", fileLabelArea_);
    const auto panel = fileArea_.toFloat();

    switch (file_.status) {
    case Status::Empty: {
        g.setColour(kPanel);
        g.fillRect(panel);
        const float dashes[] = {4.0f, 4.0f};
        juce::Path outline;
        outline.addRectangle(panel.reduced(0.5f));
        juce::PathStrokeType(1.0f).createDashedStroke(outline, outline, dashes, 2);
        g.setColour(kLine);
        g.fillPath(outline);
        g.setColour(kTextDim);
        g.setFont(sans(12.0f));
        g.drawText("Drop an audio file here to analyse it", panel.withTrimmedBottom(panel.getHeight() / 2.0f - 2.0f),
                   juce::Justification::centredBottom, false);
        g.setFont(mono(10.0f));
        g.setColour(kTextDisabled);
        g.drawText(FileAnalysis::supportedFormatsText(), panel.withTrimmedTop(panel.getHeight() / 2.0f + 4.0f),
                   juce::Justification::centredTop, false);
        g.setFont(sans(11.0f));
        g.setColour(kTextDim);
        g.drawText("Live audio is not affected.", fileInfoArea_, juce::Justification::centredLeft);
        return;
    }
    case Status::Analysing: {
        g.setColour(kPanel);
        g.fillRect(panel);
        const auto progress = std::clamp(processor_.fileAnalysis().progress(), 0.0f, 1.0f);
        // QProgressBar#stemProgress
        const auto bar = panel.withSizeKeepingCentre(panel.getWidth() - 80.0f, 8.0f).translated(0.0f, 8.0f);
        g.setColour(juce::Colour(0xff0e1116));
        g.fillRoundedRectangle(bar, 4.0f);
        g.setColour(kLine);
        g.drawRoundedRectangle(bar, 4.0f, 1.0f);
        g.setColour(kAmber);
        g.fillRoundedRectangle(bar.reduced(1.0f).withWidth((bar.getWidth() - 2.0f) * progress), 3.0f);
        const auto caption = bar.withY(bar.getY() - 26.0f).withHeight(18.0f);
        g.setColour(kText);
        g.setFont(sans(12.0f, true));
        g.drawText("Analysing " + file_.file.getFileName(), caption.withTrimmedRight(48.0f),
                   juce::Justification::centredLeft, true);
        g.setFont(mono(11.0f));
        g.setColour(kAmber);
        g.drawText(juce::String(juce::roundToInt(progress * 100.0f)) + "%", caption, juce::Justification::centredRight,
                   false);
        return;
    }
    case Status::Failed:
        g.setColour(kPanel);
        g.fillRect(panel);
        g.setColour(kError);
        g.setFont(sans(12.0f));
        g.drawFittedText(file_.file.getFileName() + ": " + file_.error, fileArea_.reduced(10),
                         juce::Justification::centred, 2);
        g.setFont(sans(11.0f));
        g.setColour(kTextDim);
        g.drawText("Drop another file to try again.", fileInfoArea_, juce::Justification::centredLeft);
        return;
    case Status::Done:
        break;
    }

    // Drawn like a clip on the Python timeline: green body and edge, title
    // top left, waveform as one min/max line per pixel column.
    g.setColour(kClipBody);
    g.fillRect(panel);
    g.setColour(kClipEdge);
    g.drawRect(panel, 1.0f);
    const auto wave = panel.reduced(1.0f).withTrimmedTop(14.0f);
    const auto mid = wave.getCentreY();
    const auto half = wave.getHeight() / 2.0f - 2.0f;
    g.setColour(kWaveCentre);
    g.drawHorizontalLine(juce::roundToInt(mid), wave.getX(), wave.getRight());

    const auto buckets = static_cast<int>(file_.overview.size());
    const auto width = juce::roundToInt(wave.getWidth());
    g.setColour(kWaveFill);
    for (int x = 0; x < width && buckets > 0; ++x) {
        const auto first = x * buckets / width;
        const auto last = std::max(first + 1, (x + 1) * buckets / width);
        float low = 0.0f;
        float high = 0.0f;
        for (int b = first; b < std::min(last, buckets); ++b) {
            low = std::min(low, file_.overview[static_cast<std::size_t>(b)].first);
            high = std::max(high, file_.overview[static_cast<std::size_t>(b)].second);
        }
        const auto top = mid - std::clamp(high, -1.0f, 1.0f) * half;
        const auto bottom = mid - std::clamp(low, -1.0f, 1.0f) * half;
        g.drawVerticalLine(juce::roundToInt(wave.getX()) + x, top, std::max(bottom, top + 1.0f));
    }

    const auto length = file_.lengthSeconds();
    if (length > 0.0) {
        for (const auto& marker : file_.markers) {
            const auto x =
                wave.getX() + static_cast<float>(marker.seconds(file_.sampleRate) / length) * wave.getWidth();
            const bool strong = marker.kind == MarkerKind::Downbeat || marker.kind == MarkerKind::BarLine;
            g.setColour(strong ? kDownbeatMark : kBeatMark);
            g.drawVerticalLine(juce::roundToInt(x), panel.getY() + 1.0f, panel.getBottom() - 1.0f);
        }
    }

    g.setColour(kLcd);
    g.setFont(sans(8.0f * 1.33f));  // QFont("DejaVu Sans", 8) is in points
    g.drawText(file_.file.getFileName(), panel.reduced(6.0f, 2.0f).withHeight(13.0f),
               juce::Justification::centredLeft, true);

    g.setFont(mono(10.0f));
    g.setColour(kTextSoft);
    const auto markers = file_.markers.empty() ? juce::String("no beat markers yet")
                                               : juce::String(static_cast<int>(file_.markers.size())) + " markers";
    g.drawText(formatDuration(length) + juce::String::formatted("   %.1f kHz   ", file_.sampleRate / 1000.0) +
                   describeChannels(file_.channels) +
                   juce::String::formatted("   peak %.1f dBFS   ",
                                           juce::Decibels::gainToDecibels(file_.peak, -100.0f)) +
                   markers,
               fileInfoArea_, juce::Justification::centredLeft);
}

void BeatAnalyzerEditor::paintMarkerStrip(juce::Graphics& g) {
    drawSectionLabel(g, "Live beat markers", markersLabelArea_);
    g.setFont(sans(11.0f));
    g.setColour(kTextDim);
    g.drawText(juce::String::formatted("last %.0f s", kVisibleSeconds), markersLabelArea_,
               juce::Justification::centredRight);

    // Ruler and lane, like the timeline: seconds ago instead of bar numbers.
    const auto ruler = rulerArea_.toFloat();
    const auto strip = stripArea_.toFloat();
    g.setColour(kRuler);
    g.fillRect(ruler);
    g.setColour(kLaneEven);
    g.fillRect(strip);
    g.setColour(kLine);
    g.drawHorizontalLine(rulerArea_.getBottom() - 1, ruler.getX(), ruler.getRight());

    g.setFont(mono(9.0f));
    const auto steps = static_cast<int>(kVisibleSeconds);
    for (int s = 0; s <= steps; ++s) {
        const auto x = strip.getRight() - strip.getWidth() * static_cast<float>(s / kVisibleSeconds);
        const auto px = juce::roundToInt(std::clamp(x, strip.getX(), strip.getRight() - 1.0f));
        g.setColour(kHandle);
        g.drawVerticalLine(px, ruler.getY() + ruler.getHeight() * 0.55f, ruler.getBottom());
        if (s > 0 && s < steps) {
            g.setColour(kGridBeat);
            g.drawVerticalLine(px, strip.getY(), strip.getBottom());
        }
        // Labels sit just right of their tick, like the timeline's bar
        // numbers; "now" sits left of the right edge.
        g.setColour(kAmberLabel);
        const auto top = ruler.getY() + 1.0f;
        const auto height = ruler.getHeight() * 0.6f;
        if (s == 0)
            g.drawText("now", juce::Rectangle<float>(x - 39.0f, top, 36.0f, height),
                       juce::Justification::centredRight, false);
        else
            g.drawText("-" + juce::String(s), juce::Rectangle<float>(x + 3.0f, top, 36.0f, height),
                       juce::Justification::centredLeft, false);
    }

    const auto rate = processor_.sampleRate();
    const auto end = rate > 0.0 ? static_cast<double>(processor_.pipeline().framesAnalyzed()) / rate : 0.0;
    const auto start = end - kVisibleSeconds;
    int visible = 0;
    if (rate > 0.0) {
        for (const auto& marker : processor_.markers()) {
            const auto t = marker.seconds(rate);
            if (t < start) continue;
            ++visible;
            const auto x = strip.getX() + static_cast<float>((t - start) / kVisibleSeconds) * strip.getWidth();
            const bool strong = marker.kind == MarkerKind::Downbeat || marker.kind == MarkerKind::BarLine;
            g.setColour(strong ? kDownbeatMark : kBeatMark);
            g.drawVerticalLine(juce::roundToInt(x), strip.getY(), strip.getBottom());
        }
    }
    if (visible == 0) {
        g.setColour(kTextDim);
        g.setFont(sans(11.0f));
        g.drawText("No beat markers yet. Beat detection is not implemented in this version.", stripArea_,
                   juce::Justification::centred);
    }
}

}  // namespace soniclab::beat_analyzer
