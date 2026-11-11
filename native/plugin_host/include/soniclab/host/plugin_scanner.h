// Discovery and loading, independent of the plugin SDK.
//
// The JUCE implementation is JucePluginHost (juce_plugin_host.h). Scanning
// loads third-party code, so the Phase 2 plan runs it in a child process
// (JUCE's out-of-process scanner pattern) and keeps a dead-man's-pedal file
// so a plugin that crashes the scan is skipped next time.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "soniclab/host/plugin_description.h"
#include "soniclab/host/plugin_instance.h"

namespace soniclab::host {

struct ScanRequest {
    std::vector<std::string> formats;      // empty means every registered format
    std::vector<std::string> searchPaths;  // empty means each format's defaults
    bool rescanKnown = false;
    std::string deadMansPedalFile;         // plugins that crashed a scan
};

// progress is 0..1; currentPlugin names the file being probed.
using ScanProgress = std::function<void(float progress, const std::string& currentPlugin)>;

class IPluginScanner {
public:
    virtual ~IPluginScanner() = default;
    virtual std::vector<std::string> availableFormats() const = 0;
    virtual std::vector<PluginDescription> scan(const ScanRequest& request, const ScanProgress& progress,
                                                const std::atomic<bool>* cancel) = 0;
    virtual std::vector<PluginDescription> knownPlugins() const = 0;

    // The scanner's cache, so a restart need not rescan.
    virtual std::string saveKnownPlugins() const = 0;
    virtual bool restoreKnownPlugins(const std::string& saved) = 0;
};

class IPluginLoader {
public:
    virtual ~IPluginLoader() = default;
    // Message thread. The instance comes back prepared for sampleRate/maxBlockSize.
    virtual std::unique_ptr<IPluginInstance> load(const PluginDescription& description, double sampleRate,
                                                  int maxBlockSize, std::string& error) = 0;
};

}  // namespace soniclab::host
