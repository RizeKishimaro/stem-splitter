#include "PluginProcessor.h"

#include <algorithm>

#include "PluginEditor.h"

namespace soniclab::beat_analyzer {
namespace {
constexpr std::size_t kMarkerHistory = 4096;
constexpr int kStateVersion = 2;  // 1: version only (Phase 1); 2: + parameters
const juce::Identifier kStateType{"SonicLabBeatAnalyzer"};
}  // namespace

juce::AudioProcessorValueTreeState::ParameterLayout BeatAnalyzerProcessor::createLayout() {
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    // Audio always passes through unchanged; this switches metering and
    // analysis on or off. Version hint 1 keeps the AU/VST3 parameter IDs stable.
    layout.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{kEnabledId, 1}, "Enabled", true));
    return layout;
}

BeatAnalyzerProcessor::BeatAnalyzerProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters_(*this, nullptr, kStateType, createLayout()),
      enabled_(parameters_.getRawParameterValue(kEnabledId)),
      pipeline_(makeAnalyzer()),
      fileAnalysis_(&BeatAnalyzerProcessor::makeAnalyzer) {
    jassert(enabled_ != nullptr);
    markers_.reserve(kMarkerHistory);
    parameters_.state.setProperty("version", kStateVersion, nullptr);
}

std::unique_ptr<OnlineAnalyzer> BeatAnalyzerProcessor::makeAnalyzer() {
    return std::make_unique<PlaceholderAnalyzer>();
}

BeatAnalyzerProcessor::~BeatAnalyzerProcessor() { pipeline_.stop(); }

void BeatAnalyzerProcessor::prepareToPlay(double sampleRate, int) {
    // Hosts call this with processing stopped, so restarting the worker is
    // safe. It may not be the message thread, so markers_ is not touched here.
    meteredChannels_.store(std::min(getTotalNumInputChannels(), kMaxMeteredChannels));
    for (auto& peak : inputPeaks_) peak.take();
    pipeline_.start(sampleRate);
    generation_.fetch_add(1, std::memory_order_release);
}

void BeatAnalyzerProcessor::releaseResources() { pipeline_.stop(); }

bool BeatAnalyzerProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo()) return false;
    return layouts.getMainInputChannelSet() == out;
}

void BeatAnalyzerProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {
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

std::array<float, BeatAnalyzerProcessor::kMaxMeteredChannels> BeatAnalyzerProcessor::takeInputPeaks() noexcept {
    std::array<float, kMaxMeteredChannels> peaks{};
    for (std::size_t ch = 0; ch < peaks.size(); ++ch) peaks[ch] = inputPeaks_[ch].take();
    return peaks;
}

juce::AudioProcessorEditor* BeatAnalyzerProcessor::createEditor() { return new BeatAnalyzerEditor(*this); }

void BeatAnalyzerProcessor::getStateInformation(juce::MemoryBlock& destData) {
    // Host thread (message thread in Logic). copyState is thread-safe in APVTS.
    auto state = parameters_.copyState();
    state.setProperty("version", kStateVersion, nullptr);
    if (auto xml = state.createXml()) copyXmlToBinary(*xml, destData);
}

void BeatAnalyzerProcessor::setStateInformation(const void* data, int sizeInBytes) {
    auto xml = getXmlFromBinary(data, sizeInBytes);
    if (xml == nullptr || !xml->hasTagName(kStateType)) return;  // unknown data: keep defaults
    auto state = juce::ValueTree::fromXml(*xml);
    // Phase 1 sessions saved only a version and no parameters; replaceState
    // then leaves each parameter at its default.
    state.setProperty("version", kStateVersion, nullptr);
    parameters_.replaceState(state);
}

const std::vector<AnalysisMarker>& BeatAnalyzerProcessor::collectMarkers() {
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

}  // namespace soniclab::beat_analyzer

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new soniclab::beat_analyzer::BeatAnalyzerProcessor();
}
