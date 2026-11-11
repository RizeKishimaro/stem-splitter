#include "soniclab/host/chain_state.h"

#include <sstream>

#include "soniclab/host/plugin_chain.h"
#include "soniclab/host/plugin_scanner.h"

namespace soniclab::host {
namespace {

constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
constexpr char kHeader[] = "soniclab-chain";

int base64Value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Values are single-line: backslash and newline are escaped.
std::string escape(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out += c;
    }
    return out;
}

bool unescape(const std::string& value, std::string& out) {
    out.clear();
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '\\') {
            out += value[i];
            continue;
        }
        if (++i >= value.size()) return false;
        switch (value[i]) {
            case '\\': out += '\\'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            default: return false;
        }
    }
    return true;
}

bool parseInt(const std::string& text, int& out) {
    try {
        std::size_t used = 0;
        out = std::stoi(text, &used);
        return used == text.size();
    } catch (...) {
        return false;
    }
}

}  // namespace

ChainState captureChainState(const PluginChain& chain) {
    ChainState state;
    for (const auto& slot : chain.slots())
        state.slots.push_back({slot.plugin->description(), slot.bypassed, slot.plugin->saveState()});
    return state;
}

int restoreChainState(const ChainState& state, IPluginLoader& loader, PluginChain& chain, double sampleRate,
                      int maxBlockSize, std::vector<std::string>& errors) {
    int loaded = 0;
    for (const auto& slot : state.slots) {
        std::string error;
        std::shared_ptr<IPluginInstance> plugin = loader.load(slot.plugin, sampleRate, maxBlockSize, error);
        if (!plugin) {
            errors.push_back(slot.plugin.name + ": " + (error.empty() ? "could not be loaded" : error));
            continue;
        }
        if (!slot.state.empty() && !plugin->restoreState(slot.state))
            errors.push_back(slot.plugin.name + ": saved state was rejected, using defaults");
        const auto id = chain.insert(std::move(plugin));
        if (slot.bypassed) chain.setBypassed(id, true);
        ++loaded;
    }
    return loaded;
}

std::string base64Encode(const std::vector<std::uint8_t>& data) {
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        const auto n = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        out += kAlphabet[(n >> 18) & 63];
        out += kAlphabet[(n >> 12) & 63];
        out += kAlphabet[(n >> 6) & 63];
        out += kAlphabet[n & 63];
    }
    const auto rest = data.size() - i;
    if (rest == 1) {
        const auto n = data[i] << 16;
        out += kAlphabet[(n >> 18) & 63];
        out += kAlphabet[(n >> 12) & 63];
        out += "==";
    } else if (rest == 2) {
        const auto n = (data[i] << 16) | (data[i + 1] << 8);
        out += kAlphabet[(n >> 18) & 63];
        out += kAlphabet[(n >> 12) & 63];
        out += kAlphabet[(n >> 6) & 63];
        out += '=';
    }
    return out;
}

bool base64Decode(const std::string& text, std::vector<std::uint8_t>& out) {
    if (text.size() % 4 != 0) return false;
    std::vector<std::uint8_t> result;
    result.reserve(text.size() / 4 * 3);
    for (std::size_t i = 0; i < text.size(); i += 4) {
        int v[4];
        int padding = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = text[i + static_cast<std::size_t>(k)];
            if (c == '=' && i + 4 == text.size() && k >= 2) {
                v[k] = 0;
                ++padding;
            } else {
                if (padding) return false;
                v[k] = base64Value(c);
                if (v[k] < 0) return false;
            }
        }
        const auto n = (v[0] << 18) | (v[1] << 12) | (v[2] << 6) | v[3];
        result.push_back(static_cast<std::uint8_t>((n >> 16) & 0xff));
        if (padding < 2) result.push_back(static_cast<std::uint8_t>((n >> 8) & 0xff));
        if (padding < 1) result.push_back(static_cast<std::uint8_t>(n & 0xff));
    }
    out = std::move(result);
    return true;
}

std::string encodeChainState(const ChainState& state) {
    std::ostringstream out;
    out << kHeader << ' ' << ChainState::kVersion << '\n';
    for (const auto& slot : state.slots) {
        const auto& p = slot.plugin;
        out << "slot\n"
            << "format=" << escape(p.format) << '\n'
            << "name=" << escape(p.name) << '\n'
            << "manufacturer=" << escape(p.manufacturer) << '\n'
            << "version=" << escape(p.version) << '\n'
            << "identifier=" << escape(p.identifier) << '\n'
            << "file=" << escape(p.fileOrIdentifier) << '\n'
            << "uid=" << p.uniqueId << '\n'
            << "instrument=" << (p.isInstrument ? 1 : 0) << '\n'
            << "inputs=" << p.numInputChannels << '\n'
            << "outputs=" << p.numOutputChannels << '\n'
            << "bypassed=" << (slot.bypassed ? 1 : 0) << '\n'
            << "state=" << base64Encode(slot.state) << '\n'
            << "end\n";
    }
    return out.str();
}

bool decodeChainState(const std::string& text, ChainState& out) {
    std::istringstream in(text);
    std::string line;
    if (!std::getline(in, line)) return false;
    const std::string expected = std::string(kHeader) + ' ' + std::to_string(ChainState::kVersion);
    if (line != expected) return false;

    ChainState result;
    SlotState* slot = nullptr;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line == "slot") {
            if (slot) return false;
            result.slots.emplace_back();
            slot = &result.slots.back();
            continue;
        }
        if (line == "end") {
            if (!slot) return false;
            slot = nullptr;
            continue;
        }
        if (!slot) return false;
        const auto eq = line.find('=');
        if (eq == std::string::npos) return false;
        const auto key = line.substr(0, eq);
        std::string value;
        if (!unescape(line.substr(eq + 1), value)) return false;
        auto& p = slot->plugin;
        int number = 0;
        if (key == "format") p.format = value;
        else if (key == "name") p.name = value;
        else if (key == "manufacturer") p.manufacturer = value;
        else if (key == "version") p.version = value;
        else if (key == "identifier") p.identifier = value;
        else if (key == "file") p.fileOrIdentifier = value;
        else if (key == "state") {
            if (!base64Decode(value, slot->state)) return false;
        } else if (parseInt(value, number)) {
            if (key == "uid") p.uniqueId = number;
            else if (key == "instrument") p.isInstrument = number != 0;
            else if (key == "inputs") p.numInputChannels = number;
            else if (key == "outputs") p.numOutputChannels = number;
            else if (key == "bypassed") slot->bypassed = number != 0;
            // Unknown numeric keys are ignored so newer files still load.
        }
        // Unknown text keys are ignored for the same reason.
    }
    if (slot) return false;
    out = std::move(result);
    return true;
}

}  // namespace soniclab::host
