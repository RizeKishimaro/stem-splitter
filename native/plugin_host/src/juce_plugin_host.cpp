// Not compiled in Phase 1 on machines without JUCE; see native/README.md.
#include "soniclab/host/juce_plugin_host.h"

#include <algorithm>

namespace soniclab::host {

PluginDescription fromJuce(const juce::PluginDescription& d) {
    PluginDescription out;
    out.format = d.pluginFormatName.toStdString();
    out.name = d.name.toStdString();
    out.manufacturer = d.manufacturerName.toStdString();
    out.version = d.version.toStdString();
    out.identifier = d.createIdentifierString().toStdString();
    out.fileOrIdentifier = d.fileOrIdentifier.toStdString();
    out.uniqueId = d.uniqueId;
    out.isInstrument = d.isInstrument;
    out.numInputChannels = d.numInputChannels;
    out.numOutputChannels = d.numOutputChannels;
    return out;
}

// ---------------------------------------------------------------- instance

JucePluginInstance::JucePluginInstance(std::unique_ptr<juce::AudioPluginInstance> instance,
                                       const juce::PluginDescription& scanned)
    : instance_(std::move(instance)), description_(fromJuce(scanned)) {}

JucePluginInstance::~JucePluginInstance() { release(); }

void JucePluginInstance::prepare(double sampleRate, int maxBlockSize) {
    if (prepared_) instance_->releaseResources();
    const auto channels = std::max({2, instance_->getTotalNumInputChannels(), instance_->getTotalNumOutputChannels()});
    scratch_.setSize(channels, maxBlockSize, false, true, false);
    midi_.ensureSize(2048);
    instance_->setRateAndBufferSizeDetails(sampleRate, maxBlockSize);
    instance_->prepareToPlay(sampleRate, maxBlockSize);
    prepared_ = true;
}

void JucePluginInstance::release() {
    if (!prepared_) return;
    instance_->releaseResources();
    prepared_ = false;
}

int JucePluginInstance::latencySamples() const { return instance_->getLatencySamples(); }

void JucePluginInstance::process(float* const* channels, int numChannels, int numFrames) noexcept {
    // The plugin may want more channels than the track has, so it always
    // processes the preallocated scratch buffer.
    numFrames = std::min(numFrames, scratch_.getNumSamples());
    const auto used = std::min(numChannels, scratch_.getNumChannels());
    for (int ch = 0; ch < scratch_.getNumChannels(); ++ch) {
        if (ch < used) scratch_.copyFrom(ch, 0, channels[ch], numFrames);
        else scratch_.clear(ch, 0, numFrames);
    }
    juce::AudioBuffer<float> view(scratch_.getArrayOfWritePointers(), scratch_.getNumChannels(), numFrames);
    midi_.clear();
    instance_->processBlock(view, midi_);
    for (int ch = 0; ch < used; ++ch) juce::FloatVectorOperations::copy(channels[ch], view.getReadPointer(ch), numFrames);
}

int JucePluginInstance::parameterCount() const { return instance_->getParameters().size(); }

ParameterInfo JucePluginInstance::parameterInfo(int index) const {
    ParameterInfo info;
    if (auto* p = instance_->getParameters()[index]) {
        if (auto* withId = dynamic_cast<juce::HostedAudioProcessorParameter*>(p))
            info.id = withId->getParameterID().toStdString();
        info.name = p->getName(128).toStdString();
        info.defaultValue = p->getDefaultValue();
        info.automatable = p->isAutomatable();
    }
    return info;
}

float JucePluginInstance::parameter(int index) const {
    auto* p = instance_->getParameters()[index];
    return p != nullptr ? p->getValue() : 0.0f;
}

void JucePluginInstance::setParameter(int index, float normalised) noexcept {
    // setValue, not setValueNotifyingHost: this runs on the audio thread and
    // the host's own UI already knows about the change it queued.
    if (auto* p = instance_->getParameters()[index]) p->setValue(normalised);
}

std::vector<std::uint8_t> JucePluginInstance::saveState() const {
    juce::MemoryBlock block;
    instance_->getStateInformation(block);
    const auto* bytes = static_cast<const std::uint8_t*>(block.getData());
    return {bytes, bytes + block.getSize()};
}

bool JucePluginInstance::restoreState(const std::vector<std::uint8_t>& state) {
    if (state.empty()) return false;
    instance_->setStateInformation(state.data(), static_cast<int>(state.size()));
    return true;
}

// -------------------------------------------------------------------- host

JucePluginHost::JucePluginHost() {
#if JUCE_PLUGINHOST_VST3
    formats_.addFormat(new juce::VST3PluginFormat());
#endif
#if JUCE_PLUGINHOST_AU && JUCE_MAC
    formats_.addFormat(new juce::AudioUnitPluginFormat());
#endif
}

std::vector<std::string> JucePluginHost::availableFormats() const {
    std::vector<std::string> out;
    for (auto* format : formats_.getFormats()) out.push_back(format->getName().toStdString());
    return out;
}

std::vector<PluginDescription> JucePluginHost::scan(const ScanRequest& request, const ScanProgress& progress,
                                                    const std::atomic<bool>* cancel) {
    const juce::File pedal(request.deadMansPedalFile);
    for (auto* format : formats_.getFormats()) {
        const auto name = format->getName().toStdString();
        if (!request.formats.empty() &&
            std::find(request.formats.begin(), request.formats.end(), name) == request.formats.end())
            continue;
        juce::FileSearchPath paths;
        for (const auto& path : request.searchPaths) paths.add(juce::File(path));
        if (paths.getNumPaths() == 0) paths = format->getDefaultLocationsToSearch();

        juce::PluginDirectoryScanner scanner(known_, *format, paths, true, pedal, false);
        juce::String current;
        while (!(cancel != nullptr && cancel->load())) {
            if (progress) progress(scanner.getProgress(), scanner.getNextPluginFileThatWillBeScanned().toStdString());
            if (!scanner.scanNextFile(!request.rescanKnown, current)) break;
        }
    }
    return knownPlugins();
}

std::vector<PluginDescription> JucePluginHost::knownPlugins() const {
    std::vector<PluginDescription> out;
    for (const auto& type : known_.getTypes()) out.push_back(fromJuce(type));
    return out;
}

std::string JucePluginHost::saveKnownPlugins() const {
    if (auto xml = known_.createXml()) return xml->toString().toStdString();
    return {};
}

bool JucePluginHost::restoreKnownPlugins(const std::string& saved) {
    auto xml = juce::parseXML(juce::String(saved));
    if (xml == nullptr) return false;
    known_.recreateFromXml(*xml);
    return true;
}

std::unique_ptr<IPluginInstance> JucePluginHost::load(const PluginDescription& description, double sampleRate,
                                                      int maxBlockSize, std::string& error) {
    auto type = known_.getTypeForIdentifierString(juce::String(description.identifier));
    if (type == nullptr) {
        error = "not in the scanned plugin list; rescan plugins";
        return nullptr;
    }
    juce::String message;
    // Synchronous instantiation suits AUv2 and VST3. AUv3 extensions may need
    // createPluginInstanceAsync; that is Phase 2 work.
    auto instance = formats_.createPluginInstance(*type, sampleRate, maxBlockSize, message);
    if (instance == nullptr) {
        error = message.toStdString();
        return nullptr;
    }
    auto wrapped = std::make_unique<JucePluginInstance>(std::move(instance), *type);
    wrapped->prepare(sampleRate, maxBlockSize);
    return wrapped;
}

}  // namespace soniclab::host
