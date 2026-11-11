#include "AnalysisViews.h"

#include <algorithm>

#include "Widgets.h"

namespace soniclab::plugin {

using namespace theme;

juce::Colour markerColour(const AnalysisMarker& marker) {
    switch (marker.kind) {
    case MarkerKind::Downbeat:
    case MarkerKind::BarLine:
        return kDownbeatMark;
    case MarkerKind::Beat:
        return kBeatMark;
    case MarkerKind::Onset:
        break;
    }
    return juce::Colour::fromRGBA(126, 168, 214, 230);
}

// ------------------------------------------------------------- waveform

WaveformView::WaveformView() {
    setMouseCursor(juce::MouseCursor::PointingHandCursor);
    clear_.setTooltip("Forget the dropped file");
    clear_.onClick = [this] {
        if (onClear) onClear();
    };
    addChildComponent(clear_);
}

bool WaveformView::isInterestedInFileDrag(const juce::StringArray& files) {
    for (const auto& path : files)
        if (FileAnalysis::canAnalyse(juce::File(path))) return true;
    return false;
}

void WaveformView::filesDropped(const juce::StringArray& files, int, int) {
    for (const auto& path : files) {
        const juce::File file(path);
        if (FileAnalysis::canAnalyse(file)) {
            if (onFile) onFile(file);
            return;
        }
    }
}

void WaveformView::mouseDown(const juce::MouseEvent&) {
    if (onBrowse) onBrowse();
}

void WaveformView::setResult(const FileAnalysisResult& result) {
    result_ = result;
    clear_.setVisible(result_.status != FileAnalysisResult::Status::Empty);
    repaint();
}

void WaveformView::setProgress(float progress) {
    progress = std::clamp(progress, 0.0f, 1.0f);
    if (juce::approximatelyEqual(progress, progress_)) return;
    progress_ = progress;
    repaint();
}

void WaveformView::resized() { clear_.setBounds(getWidth() - 52, 2, 48, 16); }

juce::String WaveformView::describe(const FileAnalysisResult& result) {
    return formatDuration(result.lengthSeconds()) + juce::String::formatted("   %.1f kHz   ", result.sampleRate / 1000.0) +
           describeChannels(result.channels) +
           juce::String::formatted("   peak %.1f dBFS", juce::Decibels::gainToDecibels(result.peak, -100.0f));
}

void WaveformView::paint(juce::Graphics& g) {
    using Status = FileAnalysisResult::Status;
    const auto panel = getLocalBounds().toFloat();

    switch (result_.status) {
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
        g.drawText("Click or drop an audio file here", panel.withTrimmedBottom(panel.getHeight() / 2.0f - 2.0f),
                   juce::Justification::centredBottom, false);
        g.setFont(mono(10.0f));
        g.setColour(kTextDisabled);
        g.drawText(FileAnalysis::supportedFormatsText(), panel.withTrimmedTop(panel.getHeight() / 2.0f + 4.0f),
                   juce::Justification::centredTop, false);
        return;
    }
    case Status::Analysing: {
        g.setColour(kPanel);
        g.fillRect(panel);
        // QProgressBar#stemProgress
        const auto bar = panel.withSizeKeepingCentre(panel.getWidth() - 80.0f, 8.0f).translated(0.0f, 8.0f);
        g.setColour(juce::Colour(0xff0e1116));
        g.fillRoundedRectangle(bar, 4.0f);
        g.setColour(kLine);
        g.drawRoundedRectangle(bar, 4.0f, 1.0f);
        g.setColour(kAmber);
        g.fillRoundedRectangle(bar.reduced(1.0f).withWidth((bar.getWidth() - 2.0f) * progress_), 3.0f);
        const auto caption = bar.withY(bar.getY() - 26.0f).withHeight(18.0f);
        g.setColour(kText);
        g.setFont(sans(12.0f, true));
        g.drawText("Analysing " + result_.file.getFileName(), caption.withTrimmedRight(48.0f),
                   juce::Justification::centredLeft, true);
        g.setFont(mono(11.0f));
        g.setColour(kAmber);
        g.drawText(juce::String(juce::roundToInt(progress_ * 100.0f)) + "%", caption, juce::Justification::centredRight,
                   false);
        return;
    }
    case Status::Failed:
        g.setColour(kPanel);
        g.fillRect(panel);
        g.setColour(kError);
        g.setFont(sans(12.0f));
        g.drawFittedText(result_.file.getFileName() + ": " + result_.error, getLocalBounds().reduced(10),
                         juce::Justification::centred, 2);
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

    const auto buckets = static_cast<int>(result_.overview.size());
    const auto width = juce::roundToInt(wave.getWidth());
    g.setColour(kWaveFill);
    for (int x = 0; x < width && buckets > 0; ++x) {
        const auto first = x * buckets / width;
        const auto last = std::max(first + 1, (x + 1) * buckets / width);
        float low = 0.0f;
        float high = 0.0f;
        for (int b = first; b < std::min(last, buckets); ++b) {
            low = std::min(low, result_.overview[static_cast<std::size_t>(b)].first);
            high = std::max(high, result_.overview[static_cast<std::size_t>(b)].second);
        }
        const auto top = mid - std::clamp(high, -1.0f, 1.0f) * half;
        const auto bottom = mid - std::clamp(low, -1.0f, 1.0f) * half;
        g.drawVerticalLine(juce::roundToInt(wave.getX()) + x, top, std::max(bottom, top + 1.0f));
    }

    const auto length = result_.lengthSeconds();
    if (length > 0.0) {
        for (const auto& marker : result_.markers) {
            const auto x =
                wave.getX() + static_cast<float>(marker.seconds(result_.sampleRate) / length) * wave.getWidth();
            const bool strong = marker.kind == MarkerKind::Downbeat || marker.kind == MarkerKind::BarLine;
            g.setColour(markerColour(marker));
            g.drawVerticalLine(juce::roundToInt(x), strong ? panel.getY() + 1.0f : wave.getY(),
                               panel.getBottom() - 1.0f);
        }
    }

    g.setColour(kLcd);
    g.setFont(sans(8.0f * 1.33f));  // QFont("DejaVu Sans", 8) is in points
    g.drawText(result_.file.getFileName(), panel.reduced(6.0f, 2.0f).withHeight(13.0f).withTrimmedRight(56.0f),
               juce::Justification::centredLeft, true);
}

// ----------------------------------------------------------- marker lane

void MarkerLane::update(const std::vector<AnalysisMarker>& markers, double sampleRate, double endSeconds) {
    absolute_ = false;
    window_ = liveWindow_;
    visible_.clear();
    end_ = endSeconds;
    origin_ = endSeconds - window_;
    if (sampleRate > 0.0) {
        for (const auto& marker : markers) {
            const auto t = marker.seconds(sampleRate);
            if (t >= origin_ && t <= endSeconds) visible_.push_back({t, marker});
        }
    }
    repaint();
}

void MarkerLane::showSpan(const std::vector<AnalysisMarker>& markers, double sampleRate, double startSeconds,
                          double endSeconds) {
    absolute_ = true;
    origin_ = startSeconds;
    end_ = std::max(endSeconds, startSeconds + 0.001);
    window_ = end_ - origin_;
    visible_.clear();
    if (sampleRate > 0.0) {
        for (const auto& marker : markers) {
            const auto t = marker.seconds(sampleRate);
            if (t >= origin_ && t <= end_) visible_.push_back({t, marker});
        }
    }
    repaint();
}

void MarkerLane::paint(juce::Graphics& g) {
    auto area = getLocalBounds();
    const auto ruler = area.removeFromTop(kRulerHeight).toFloat();
    const auto strip = area.toFloat();

    // Ruler and lane, like the timeline: seconds ago instead of bar numbers.
    g.setColour(kRuler);
    g.fillRect(ruler);
    g.setColour(kLaneEven);
    g.fillRect(strip);
    g.setColour(kLine);
    g.drawHorizontalLine(juce::roundToInt(ruler.getBottom()) - 1, ruler.getX(), ruler.getRight());

    g.setFont(mono(9.0f));
    if (absolute_) {
        // Whole-file timeline: a few absolute times across the file.
        constexpr int divisions = 4;
        for (int s = 0; s <= divisions; ++s) {
            const auto fraction = static_cast<float>(s) / static_cast<float>(divisions);
            const auto x = strip.getX() + strip.getWidth() * fraction;
            const auto px = juce::roundToInt(std::clamp(x, strip.getX(), strip.getRight() - 1.0f));
            g.setColour(kHandle);
            g.drawVerticalLine(px, ruler.getY() + ruler.getHeight() * 0.55f, ruler.getBottom());
            if (s > 0 && s < divisions) {
                g.setColour(kGridBeat);
                g.drawVerticalLine(px, strip.getY(), strip.getBottom());
            }
            g.setColour(kAmberLabel);
            const auto label = formatDuration(origin_ + window_ * fraction);
            const auto top = ruler.getY() + 1.0f;
            const auto height = ruler.getHeight() * 0.6f;
            if (s == divisions)
                g.drawText(label, juce::Rectangle<float>(x - 52.0f, top, 48.0f, height),
                           juce::Justification::centredRight, false);
            else
                g.drawText(label, juce::Rectangle<float>(x + 3.0f, top, 48.0f, height),
                           juce::Justification::centredLeft, false);
        }
    } else {
        const auto steps = static_cast<int>(window_);
        for (int s = 0; s <= steps; ++s) {
            const auto x = strip.getRight() - strip.getWidth() * static_cast<float>(s / window_);
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
    }

    const auto start = absolute_ ? origin_ : end_ - window_;
    float lastNumberX = -100.0f;
    for (const auto& item : visible_) {
        const auto x = strip.getX() + static_cast<float>((item.seconds - start) / window_) * strip.getWidth();
        const bool strong = item.marker.kind == MarkerKind::Downbeat || item.marker.kind == MarkerKind::BarLine;
        g.setColour(markerColour(item.marker));
        const auto top = strong ? strip.getY() : strip.getY() + strip.getHeight() * 0.3f;
        g.fillRect(juce::Rectangle<float>(x - (strong ? 1.0f : 0.5f), top, strong ? 2.0f : 1.0f, strip.getBottom() - top));
        // A whole file has a marker every half second, so only number the
        // downbeats, and only when the last number has room.
        const bool numbered = item.marker.beatInBar > 0 && (!absolute_ || (strong && x - lastNumberX >= 16.0f));
        if (numbered) {
            g.setFont(mono(9.0f));
            g.setColour(strong ? kAmber : kTextDim);
            g.drawText(juce::String(item.marker.beatInBar), juce::Rectangle<float>(x + 3.0f, strip.getY() + 2.0f, 14.0f, 12.0f),
                       juce::Justification::centredLeft, false);
            lastNumberX = x;
        }
    }
    if (visible_.empty()) {
        g.setColour(kTextDim);
        g.setFont(sans(11.0f));
        g.drawText(emptyText_, area, juce::Justification::centred);
    }
}

}  // namespace soniclab::plugin
