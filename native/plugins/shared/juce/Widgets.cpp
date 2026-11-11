#include "Widgets.h"

#include <algorithm>

namespace soniclab::plugin {
namespace {
constexpr float kReleaseDbPerSecond = 20.0f;
constexpr double kHoldSeconds = 1.5;
constexpr float kLabelWidth = 18.0f;
constexpr float kScaleHeight = 14.0f;
// meter.py turns a bar amber from 0.92 of full scale.
const float kHotDb = juce::Decibels::gainToDecibels(0.92f);
}  // namespace

using namespace theme;

juce::String formatDuration(double seconds) {
    seconds = std::max(0.0, seconds);
    const auto minutes = static_cast<int>(seconds / 60.0);
    return juce::String::formatted("%d:%04.1f", minutes, seconds - 60.0 * minutes);
}

juce::String describeChannels(int channels) {
    return channels == 1 ? juce::String("Mono") : channels == 2 ? juce::String("Stereo") : juce::String(channels) + " ch";
}

void drawLcdFrame(juce::Graphics& g, juce::Rectangle<int> area) {
    g.setColour(kLcdBackground);
    g.fillRect(area);
    g.setColour(kLine);
    g.drawRect(area, 1);
}

void drawConfidence(juce::Graphics& g, juce::Rectangle<int> area, float confidence) {
    constexpr int kSegments = 5;
    const auto gap = 2.0f;
    const auto bounds = area.toFloat();
    const auto width = (bounds.getWidth() - gap * (kSegments - 1)) / kSegments;
    const auto lit = confidence < 0.0f ? 0 : juce::jlimit(0, kSegments, juce::roundToInt(confidence * kSegments));
    for (int i = 0; i < kSegments; ++i) {
        const auto segment = juce::Rectangle<float>(bounds.getX() + static_cast<float>(i) * (width + gap), bounds.getY(),
                                                    width, bounds.getHeight());
        g.setColour(i < lit ? kAmber : kGroove);
        g.fillRect(segment);
    }
}

// ------------------------------------------------------------------ meter

void LevelMeter::update(const std::array<float, kMaxMeteredChannels>& peaks, int channels, double elapsedSeconds) {
    channels_ = std::clamp(channels, 0, kMaxMeteredChannels);
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

}  // namespace soniclab::plugin
