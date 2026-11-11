// Small drawing helpers and components every SonicLab plugin editor uses:
// the input level meter, LCD readouts, confidence bars and text formats.
// Message thread only.
#pragma once

#include <array>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "Theme.h"

namespace soniclab::plugin {

inline constexpr int kMaxMeteredChannels = 2;

juce::String formatDuration(double seconds);   // "2:05.3"
juce::String describeChannels(int channels);  // "Mono", "Stereo", "6 ch"

// QFrame#lcd: the dark inset box behind numeric readouts.
void drawLcdFrame(juce::Graphics& g, juce::Rectangle<int> area);

// Five-segment confidence bar, amber segments for confidence in 0..1. A
// value below zero draws an empty, dimmed bar ("not known").
void drawConfidence(juce::Graphics& g, juce::Rectangle<int> area, float confidence);

// Peak meter with UI-side ballistics: instant attack, ~20 dB/s release and
// a 1.5 s peak hold. Displays -60..+6 dBFS with a dB scale underneath and
// channel labels on the left, in the Python mixer's colours (grey bars,
// amber when hot); ``setHint`` text is drawn over the bars while the input
// is silent.
class LevelMeter final : public juce::Component {
public:
    static constexpr float kFloorDb = -60.0f;
    static constexpr float kCeilingDb = 6.0f;

    void update(const std::array<float, kMaxMeteredChannels>& peaks, int channels, double elapsedSeconds);
    void reset();
    void setHint(const juce::String& hint);
    void paint(juce::Graphics& g) override;

    float displayedDecibels(int channel) const { return levels_[static_cast<std::size_t>(channel)]; }

private:
    std::array<float, kMaxMeteredChannels> levels_{kFloorDb, kFloorDb};
    std::array<float, kMaxMeteredChannels> holds_{kFloorDb, kFloorDb};
    std::array<double, kMaxMeteredChannels> holdAge_{};
    int channels_ = 0;
    juce::String hint_;
};

}  // namespace soniclab::plugin
