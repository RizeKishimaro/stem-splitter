// JUCE implementation of the scanner, loader and plugin instance interfaces.
//
// Built only when SONICLAB_WITH_JUCE is ON. Formats are registered per
// platform: AudioUnit on macOS only, VST3 where JUCE_PLUGINHOST_VST3 is set.
// Every call here is message-thread only, except JucePluginInstance::process
// and setParameter, which the PluginChain calls on the audio thread.
#pragma once

#include <memory>

#include <juce_audio_processors/juce_audio_processors.h>

#include "soniclab/host/plugin_scanner.h"

namespace soniclab::host {

PluginDescription fromJuce(const juce::PluginDescription& description);

class JucePluginInstance final : public IPluginInstance {
public:
    // ``scanned`` is the description the plugin was loaded from. It is kept
    // as this instance's identity because the instance's own description can
    // differ: for VST3 it names the binary inside the bundle, so its
    // identifier would not match the scanned list when a chain is restored.
    JucePluginInstance(std::unique_ptr<juce::AudioPluginInstance> instance, const juce::PluginDescription& scanned);
    ~JucePluginInstance() override;

    const PluginDescription& description() const override { return description_; }
    void prepare(double sampleRate, int maxBlockSize) override;
    void release() override;
    int latencySamples() const override;
    void process(float* const* channels, int numChannels, int numFrames) noexcept override;
    int parameterCount() const override;
    ParameterInfo parameterInfo(int index) const override;
    float parameter(int index) const override;
    void setParameter(int index, float normalised) noexcept override;
    std::vector<std::uint8_t> saveState() const override;
    bool restoreState(const std::vector<std::uint8_t>& state) override;

    juce::AudioPluginInstance& juceInstance() { return *instance_; }

private:
    std::unique_ptr<juce::AudioPluginInstance> instance_;
    PluginDescription description_;
    juce::AudioBuffer<float> scratch_;  // sized in prepare(), never on the audio thread
    juce::MidiBuffer midi_;
    bool prepared_ = false;
};

class JucePluginHost final : public IPluginScanner, public IPluginLoader {
public:
    JucePluginHost();

    std::vector<std::string> availableFormats() const override;
    std::vector<PluginDescription> scan(const ScanRequest& request, const ScanProgress& progress,
                                        const std::atomic<bool>* cancel) override;
    std::vector<PluginDescription> knownPlugins() const override;
    std::string saveKnownPlugins() const override;
    bool restoreKnownPlugins(const std::string& saved) override;

    std::unique_ptr<IPluginInstance> load(const PluginDescription& description, double sampleRate,
                                          int maxBlockSize, std::string& error) override;

private:
    juce::AudioPluginFormatManager formats_;
    juce::KnownPluginList known_;
};

}  // namespace soniclab::host
