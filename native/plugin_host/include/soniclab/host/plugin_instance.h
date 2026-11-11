// A loaded plugin, as the chain sees it.
//
// Thread contract, matching juce::AudioProcessor:
//   message thread  prepare, release, parameterInfo, saveState, restoreState
//   audio thread    process, setParameter (only through PluginChain's queue)
// prepare() is called before the instance is published to the audio thread
// and never while it is processing.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "soniclab/host/plugin_description.h"

namespace soniclab::host {

struct ParameterInfo {
    std::string id;
    std::string name;
    float defaultValue = 0.0f;  // normalised 0..1
    bool automatable = true;
};

class IPluginInstance {
public:
    virtual ~IPluginInstance() = default;

    virtual const PluginDescription& description() const = 0;

    virtual void prepare(double sampleRate, int maxBlockSize) = 0;
    virtual void release() = 0;
    virtual int latencySamples() const = 0;

    // In place on ``numChannels`` non-interleaved channels. Realtime safe.
    virtual void process(float* const* channels, int numChannels, int numFrames) noexcept = 0;

    virtual int parameterCount() const = 0;
    virtual ParameterInfo parameterInfo(int index) const = 0;
    virtual float parameter(int index) const = 0;
    virtual void setParameter(int index, float normalised) noexcept = 0;

    virtual std::vector<std::uint8_t> saveState() const = 0;
    virtual bool restoreState(const std::vector<std::uint8_t>& state) = 0;
};

}  // namespace soniclab::host
