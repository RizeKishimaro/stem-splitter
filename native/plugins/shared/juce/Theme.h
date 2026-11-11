// SonicLab's visual theme for every plugin editor, matching the Python app
// (soniclab/ui/theme.py and the timeline's clip colours): near-black
// surfaces, square corners, small upper-case section labels, and amber as
// the only highlight. Colour names follow theme.py.
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace soniclab::plugin::theme {

// soniclab/ui/theme.py
inline const juce::Colour kWindow{0xff0c0d10};
inline const juce::Colour kPanel{0xff12151a};
inline const juce::Colour kPanelRaised{0xff181c22};
inline const juce::Colour kRuler{0xff101318};
inline const juce::Colour kLine{0xff2a3038};
inline const juce::Colour kLineSoft{0xff1e232b};
inline const juce::Colour kText{0xffe6e8ec};
inline const juce::Colour kTextSoft{0xffc5cad3};
inline const juce::Colour kTextDim{0xff8b939f};
inline const juce::Colour kTextDisabled{0xff5e6772};
inline const juce::Colour kAmber{0xffd7b36a};
inline const juce::Colour kAmberDim{0xff8a7040};
inline const juce::Colour kAmberLabel{0xffc6a56a};  // timeline bar numbers
inline const juce::Colour kLcd{0xffe7f2ea};
inline const juce::Colour kLcdBackground{0xff0c0f13};
inline const juce::Colour kWell{0xff0e1114};  // inputs, meter background
inline const juce::Colour kHover{0xff222831};
inline const juce::Colour kHandle{0xff3a424e};
inline const juce::Colour kSelect{0xff243044};
inline const juce::Colour kMenu{0xff161b21};
inline const juce::Colour kGroove{0xff2c333c};
inline const juce::Colour kSliderHandle{0xffc8ced6};
// soniclab/ui/meter.py
inline const juce::Colour kMeter{0xff8e9aad};
inline const juce::Colour kMeterHot{0xffc4a46a};
// soniclab/ui/timeline.py
inline const juce::Colour kClipBody{0xff1c382c};
inline const juce::Colour kClipEdge{0xff3e6a52};
inline const juce::Colour kWaveFill{0xff8fcb9b};
inline const juce::Colour kWaveCentre{juce::Colour::fromRGBA(46, 78, 62, 160)};
inline const juce::Colour kLaneEven{0xff12161b};
inline const juce::Colour kGridBeat{0xff222830};
inline const juce::Colour kBeatMark{juce::Colour::fromRGBA(214, 230, 218, 90)};
inline const juce::Colour kDownbeatMark{juce::Colour::fromRGBA(244, 214, 150, 235)};
inline const juce::Colour kError{0xffc47060};  // timeline ONSET_LATE

// The Python app asks for DejaVu Sans; other platforms fall back to their
// own UI fonts, as Qt does there.
inline juce::String sansName() {
#if JUCE_LINUX || JUCE_BSD
    return "DejaVu Sans";
#else
    return juce::Font::getDefaultSansSerifFontName();
#endif
}

inline juce::String monoName() {
#if JUCE_LINUX || JUCE_BSD
    return "DejaVu Sans Mono";
#elif JUCE_MAC
    return "Menlo";
#else
    return juce::Font::getDefaultMonospacedFontName();
#endif
}

// ``pixels`` is the Qt stylesheet font-size; JUCE heights include
// ascent and descent, so they are a little larger.
inline juce::FontOptions sans(float pixels, bool bold = false) {
    return juce::FontOptions(sansName(), pixels * 1.17f, bold ? juce::Font::bold : juce::Font::plain);
}

inline juce::FontOptions mono(float pixels, bool bold = false) {
    return juce::FontOptions(monoName(), pixels * 1.17f, bold ? juce::Font::bold : juce::Font::plain);
}

// QLabel#sectionLabel: 10px, bold, dim, letter-spaced, upper case.
inline void drawSectionLabel(juce::Graphics& g, const juce::String& text, juce::Rectangle<int> area) {
    g.setColour(kTextDim);
    g.setFont(juce::Font(sans(10.0f, true)).withExtraKerningFactor(0.08f));
    g.drawText(text.toUpperCase(), area, juce::Justification::centredLeft, true);
}

class LookAndFeel final : public juce::LookAndFeel_V4 {
public:
    LookAndFeel() {
        setDefaultSansSerifTypefaceName(sansName());
        setColour(juce::ResizableWindow::backgroundColourId, kWindow);
        setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
        setColour(juce::TextButton::textColourOffId, kTextSoft);
        setColour(juce::TextButton::textColourOnId, juce::Colour(0xfff2f4f7));
        setColour(juce::ToggleButton::textColourId, kText);
        setColour(juce::Label::textColourId, kText);
        setColour(juce::TooltipWindow::backgroundColourId, kMenu);
        setColour(juce::TooltipWindow::textColourId, kText);
        setColour(juce::TooltipWindow::outlineColourId, kLine);
        // QLineEdit / QComboBox / QDoubleSpinBox
        setColour(juce::ComboBox::backgroundColourId, kWell);
        setColour(juce::ComboBox::outlineColourId, kLine);
        setColour(juce::ComboBox::textColourId, kText);
        setColour(juce::ComboBox::arrowColourId, kTextDim);
        setColour(juce::ComboBox::focusedOutlineColourId, kAmberDim);
        setColour(juce::PopupMenu::backgroundColourId, kMenu);
        setColour(juce::PopupMenu::textColourId, kText);
        setColour(juce::PopupMenu::highlightedBackgroundColourId, kSelect);
        setColour(juce::PopupMenu::highlightedTextColourId, juce::Colour(0xfff4f6f8));
        setColour(juce::Slider::textBoxBackgroundColourId, kWell);
        setColour(juce::Slider::textBoxOutlineColourId, kLine);
        setColour(juce::Slider::textBoxTextColourId, kText);
        setColour(juce::Slider::textBoxHighlightColourId, juce::Colour(0xff3a3424));
        setColour(juce::TextEditor::backgroundColourId, kWell);
        setColour(juce::TextEditor::textColourId, kText);
        setColour(juce::TextEditor::outlineColourId, kLine);
        setColour(juce::TextEditor::focusedOutlineColourId, kAmberDim);
        setColour(juce::TextEditor::highlightColourId, juce::Colour(0xff3a3424));
        setColour(juce::CaretComponent::caretColourId, kAmber);
    }

    // QCheckBox: a 12px square, amber when checked.
    void drawToggleButton(juce::Graphics& g, juce::ToggleButton& button, bool highlighted, bool) override {
        const auto bounds = button.getLocalBounds();
        const auto box = juce::Rectangle<float>(0.5f, (static_cast<float>(bounds.getHeight()) - 12.0f) / 2.0f, 12.0f, 12.0f);
        const bool on = button.getToggleState();
        g.setColour(on ? kAmberDim : kWell);
        g.fillRect(box);
        g.setColour(on ? kAmber : (highlighted ? kTextDim : kHandle));
        g.drawRect(box, 1.0f);
        g.setColour(button.isEnabled() ? kText : kTextDisabled);
        g.setFont(sans(12.0f));
        g.drawText(button.getButtonText(), bounds.withTrimmedLeft(19), juce::Justification::centredLeft, false);
    }

    // QToolButton: flat and square, with the mixer buttons' thin border.
    void drawButtonBackground(juce::Graphics& g, juce::Button& button, const juce::Colour&, bool highlighted,
                              bool down) override {
        const auto bounds = button.getLocalBounds().toFloat().reduced(0.5f);
        if (down) g.setColour(kWindow);
        else if (highlighted) g.setColour(kHover);
        else g.setColour(juce::Colours::transparentBlack);
        g.fillRect(bounds);
        g.setColour(button.getToggleState() ? kAmberDim : kLine);
        g.drawRect(bounds, 1.0f);
    }

    juce::Font getTextButtonFont(juce::TextButton&, int) override { return sans(11.0f); }

    // QComboBox: square well with a small chevron.
    void drawComboBox(juce::Graphics& g, int width, int height, bool, int, int, int, int,
                      juce::ComboBox& box) override {
        const auto bounds = juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
        g.setColour(box.findColour(juce::ComboBox::backgroundColourId));
        g.fillRect(bounds);
        g.setColour(box.hasKeyboardFocus(true) ? kAmberDim : kLine);
        g.drawRect(bounds.reduced(0.5f), 1.0f);
        const auto cx = static_cast<float>(width) - 9.0f;
        const auto cy = static_cast<float>(height) / 2.0f;
        juce::Path chevron;
        chevron.startNewSubPath(cx - 3.5f, cy - 1.5f);
        chevron.lineTo(cx, cy + 2.0f);
        chevron.lineTo(cx + 3.5f, cy - 1.5f);
        g.setColour(kTextDim);
        g.strokePath(chevron, juce::PathStrokeType(1.2f));
    }

    juce::Font getComboBoxFont(juce::ComboBox&) override { return mono(11.0f); }
    juce::Font getPopupMenuFont() override { return sans(12.0f); }

    void positionComboBoxText(juce::ComboBox& box, juce::Label& label) override {
        label.setBounds(1, 1, box.getWidth() - 18, box.getHeight() - 2);
        label.setFont(getComboBoxFont(box));
    }

    void drawPopupMenuBackground(juce::Graphics& g, int width, int height) override {
        g.fillAll(kMenu);
        g.setColour(kLine);
        g.drawRect(0, 0, width, height, 1);
    }

    // QSlider: a 2px groove and a narrow light handle.
    void drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height, float sliderPos, float, float,
                          juce::Slider::SliderStyle style, juce::Slider& slider) override {
        if (style != juce::Slider::LinearHorizontal) {
            LookAndFeel_V4::drawLinearSlider(g, x, y, width, height, sliderPos, 0.0f, 0.0f, style, slider);
            return;
        }
        const auto cy = static_cast<float>(y) + static_cast<float>(height) / 2.0f;
        g.setColour(kGroove);
        g.fillRect(juce::Rectangle<float>(static_cast<float>(x), cy - 1.0f, static_cast<float>(width), 2.0f));
        g.setColour(kAmberDim);
        g.fillRect(juce::Rectangle<float>(static_cast<float>(x), cy - 1.0f, sliderPos - static_cast<float>(x), 2.0f));
        g.setColour(slider.isEnabled() ? kSliderHandle : kTextDisabled);
        g.fillRect(juce::Rectangle<float>(sliderPos - 4.0f, cy - 6.0f, 8.0f, 12.0f));
    }

    int getSliderThumbRadius(juce::Slider&) override { return 4; }

    juce::Label* createSliderTextBox(juce::Slider& slider) override {
        auto* label = LookAndFeel_V4::createSliderTextBox(slider);
        label->setFont(mono(11.0f));
        label->setJustificationType(juce::Justification::centred);
        return label;
    }
};

}  // namespace soniclab::plugin::theme
