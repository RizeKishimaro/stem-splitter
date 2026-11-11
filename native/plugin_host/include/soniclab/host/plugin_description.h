// What the host knows about a plugin without loading it.
//
// JUCE-free mirror of juce::PluginDescription so the chain, its saved state
// and any Python binding can be built and tested without JUCE.
#pragma once

#include <string>

namespace soniclab::host {

struct PluginDescription {
    std::string format;        // "AudioUnit", "VST3", ...
    std::string name;
    std::string manufacturer;
    std::string version;
    std::string identifier;    // juce::PluginDescription::createIdentifierString()
    std::string fileOrIdentifier;
    int uniqueId = 0;
    bool isInstrument = false;
    int numInputChannels = 0;
    int numOutputChannels = 0;
};

}  // namespace soniclab::host
