#include "AnalysisProcessorBase.h"

#include <algorithm>

namespace soniclab::plugin {
namespace {
constexpr std::size_t kMarkerHistory = 4096;
}  // namespace

juce::AudioProcessorValueTreeState::ParameterLayout AnalysisProcessorBase::withEnabled(
    juce::AudioProcessorValueTreeState::ParameterLayout layout) {
    // Audio always passes through unchanged; this switches metering and
    // analysis on or off. Version hint 1 keeps the AU/VST3 parameter IDs stable.
    layout.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{kEnabledId, 1}, "Enabled", true));
    return layout;
}

AnalysisProcessorBase::AnalysisProcessorBase(Options options)
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      stateType_(options.stateType),
      stateVersion_(options.stateVersion),
      parameters_(*this, nullptr, options.stateType, withEnabled(std::move(options.parameters))),
      enabled_(parameters_.getRawParameterValue(kEnabledId)),
      makeAnalyzer_(std::move(options.makeAnalyzer)),
      pipeline_(makeAnalyzer_(*this, AnalyzerUse::Live)),
      fileAnalysis_([this] { return makeAnalyzer_(*this, AnalyzerUse::File); }) {
    jassert(enabled_ != nullptr);
    markers_.reserve(kMarkerHistory);
    parameters_.state.setProperty("version", stateVersion_, nullptr);
}

AnalysisProcessorBase::~AnalysisProcessorBase() { pipeline_.stop(); }

const std::atomic<float>* AnalysisProcessorBase::rawParameter(const juce::String& id) const {
    return parameters_.getRawParameterValue(id);
}

void AnalysisProcessorBase::prepareToPlay(double sampleRate, int) {
    // Hosts call this with processing stopped, so restarting the worker is
    // safe. It may not be the message thread, so markers_ is not touched here.
    meteredChannels_.store(std::min(getTotalNumInputChannels(), kMaxMeteredChannels));
    for (auto& peak : inputPeaks_) peak.take();
    pipeline_.start(sampleRate);
    generation_.fetch_add(1, std::memory_order_release);
}

void AnalysisProcessorBase::releaseResources() { pipeline_.stop(); }

void AnalysisProcessorBase::resetLiveAnalysis() noexcept {
    pipeline_.requestAnalyzerReset();
    generation_.fetch_add(1, std::memory_order_release);
}

bool AnalysisProcessorBase::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo()) return false;
    return layouts.getMainInputChannelSet() == out;
}

void AnalysisProcessorBase::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {
    // Realtime: no locks, no allocation, no waiting. Everything below is
    // arithmetic on the host's buffer, atomics, and a lock-free FIFO push.
    juce::ScopedNoDenormals noDenormals;
    midi.clear();  // not a MIDI plugin; never pass stray events on

    const auto samples = buffer.getNumSamples();
    // Never index past what the host actually gave us.
    const auto inputs = std::min(getTotalNumInputChannels(), buffer.getNumChannels());
    const auto outputs = std::min(getTotalNumOutputChannels(), buffer.getNumChannels());

    // Pass-through: input and output share the buffer, so the audio is left
    // as it is. Output channels without a matching input are silenced.
    for (auto ch = inputs; ch < outputs; ++ch) buffer.clear(ch, 0, samples);

    if (samples <= 0 || inputs <= 0 || !analysisEnabled()) return;

    for (int ch = 0; ch < std::min(inputs, kMaxMeteredChannels); ++ch)
        inputPeaks_[static_cast<std::size_t>(ch)].raise(buffer.getMagnitude(ch, 0, samples));

    pipeline_.pushAudio(buffer.getArrayOfReadPointers(), inputs, samples);
}

std::array<float, AnalysisProcessorBase::kMaxMeteredChannels> AnalysisProcessorBase::takeInputPeaks() noexcept {
    std::array<float, kMaxMeteredChannels> peaks{};
    for (std::size_t ch = 0; ch < peaks.size(); ++ch) peaks[ch] = inputPeaks_[ch].take();
    return peaks;
}

void AnalysisProcessorBase::getStateInformation(juce::MemoryBlock& destData) {
    // Host thread (message thread in Logic). copyState is thread-safe in APVTS.
    auto state = parameters_.copyState();
    state.setProperty("version", stateVersion_, nullptr);
    if (auto xml = state.createXml()) copyXmlToBinary(*xml, destData);
}

void AnalysisProcessorBase::setStateInformation(const void* data, int sizeInBytes) {
    auto xml = getXmlFromBinary(data, sizeInBytes);
    if (xml == nullptr || !xml->hasTagName(stateType_)) return;  // unknown data: keep defaults
    auto state = juce::ValueTree::fromXml(*xml);
    const int saved = state.getProperty("version", 1);
    if (saved > stateVersion_) return;  // from a newer plugin: do not guess
    if (saved < stateVersion_) upgradeState(state, saved);
    state.setProperty("version", stateVersion_, nullptr);
    parameters_.replaceState(state);
}

void AnalysisProcessorBase::upgradeState(juce::ValueTree&, int) {}

const std::vector<AnalysisMarker>& AnalysisProcessorBase::collectMarkers() {
    const auto generation = generation_.load(std::memory_order_acquire);
    if (generation != seenGeneration_) {
        seenGeneration_ = generation;
        markers_.clear();
    }
    AnalysisMarker batch[256];
    std::size_t count = 0;
    while ((count = pipeline_.popMarkers(batch, std::size(batch))) > 0) {
        for (std::size_t i = 0; i < count; ++i) {
            if (markers_.size() == kMarkerHistory) markers_.erase(markers_.begin());
            markers_.push_back(batch[i]);
        }
    }
    return markers_;
}

}  // namespace soniclab::plugin
