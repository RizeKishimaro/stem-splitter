// Saved form of a plugin chain: which plugins, in what order, and each
// plugin's opaque state blob (juce::AudioProcessor::getStateInformation).
//
// The text form is line-based UTF-8 so it can sit inside SonicLab's JSON
// project file as a single string later, without changing the project
// schema now. State blobs are base64.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "soniclab/host/plugin_description.h"

namespace soniclab::host {

class IPluginLoader;
class PluginChain;

struct SlotState {
    PluginDescription plugin;
    bool bypassed = false;
    std::vector<std::uint8_t> state;
};

struct ChainState {
    static constexpr int kVersion = 1;
    std::vector<SlotState> slots;
};

// Message thread. Asks every plugin in the chain for its state.
ChainState captureChainState(const PluginChain& chain);

// Message thread. Loads each saved plugin through ``loader``, restores its
// state and appends it to ``chain``. A plugin that is missing or fails to
// load is skipped and described in ``errors``. Returns how many loaded.
int restoreChainState(const ChainState& state, IPluginLoader& loader, PluginChain& chain, double sampleRate,
                      int maxBlockSize, std::vector<std::string>& errors);

std::string encodeChainState(const ChainState& state);
// False on malformed input or an unsupported version; ``out`` is then unchanged.
bool decodeChainState(const std::string& text, ChainState& out);

std::string base64Encode(const std::vector<std::uint8_t>& data);
bool base64Decode(const std::string& text, std::vector<std::uint8_t>& out);

}  // namespace soniclab::host
