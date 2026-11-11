// Headless tests of BeatAnalyzerProcessor through the JUCE AudioProcessor API,
// the same calls an AU/VST3 wrapper makes. They run wherever JUCE builds
// (Linux included); they do not exercise the AU wrapper itself.
// Bit-exact pass-through checks compare floats with == on purpose; JUCE's
// recommended warning flags (inherited from the plugin target) flag that.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "test_framework.h"

// ---- allocation tracking: any heap allocation on a tracked thread counts.
namespace {
thread_local bool tracking = false;
std::atomic<int> trackedAllocations{0};

void* allocate(std::size_t size) {
    if (tracking) trackedAllocations.fetch_add(1);
    if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
    throw std::bad_alloc();
}
}  // namespace

void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using soniclab::beat_analyzer::BeatAnalyzerEditor;
using soniclab::beat_analyzer::BeatAnalyzerProcessor;
using soniclab::beat_analyzer::FileAnalysis;
using soniclab::beat_analyzer::FileAnalysisResult;

namespace {

constexpr double kRate = 48000.0;
constexpr int kBlock = 512;

std::unique_ptr<BeatAnalyzerProcessor> makeProcessor(const juce::AudioChannelSet& layout) {
    auto processor = std::make_unique<BeatAnalyzerProcessor>();
    juce::AudioProcessor::BusesLayout buses;
    buses.inputBuses.add(layout);
    buses.outputBuses.add(layout);
    CHECK(processor->setBusesLayout(buses));
    processor->setRateAndBufferSizeDetails(kRate, kBlock);
    processor->prepareToPlay(kRate, kBlock);
    return processor;
}

juce::AudioBuffer<float> noise(int channels, int samples, float amplitude, unsigned seed) {
    juce::AudioBuffer<float> buffer(channels, samples);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-amplitude, amplitude);
    for (int ch = 0; ch < channels; ++ch)
        for (int i = 0; i < samples; ++i) buffer.setSample(ch, i, dist(rng));
    return buffer;
}

bool identical(const juce::AudioBuffer<float>& a, const juce::AudioBuffer<float>& b) {
    if (a.getNumChannels() != b.getNumChannels() || a.getNumSamples() != b.getNumSamples()) return false;
    for (int ch = 0; ch < a.getNumChannels(); ++ch)
        if (std::memcmp(a.getReadPointer(ch), b.getReadPointer(ch), sizeof(float) * static_cast<std::size_t>(a.getNumSamples())) != 0)
            return false;
    return true;
}

void setEnabled(BeatAnalyzerProcessor& processor, bool on) {
    auto* parameter = processor.parameters().getParameter(BeatAnalyzerProcessor::kEnabledId);
    CHECK(parameter != nullptr);
    parameter->setValueNotifyingHost(on ? 1.0f : 0.0f);
}

}  // namespace

TEST("plugin.identity_matches_au_registration") {
    BeatAnalyzerProcessor processor;
    CHECK_EQ(processor.getName(), juce::String("SonicLab Beat Analyzer"));
    CHECK_EQ(juce::String(JucePlugin_Manufacturer), juce::String("SonicLab"));
    const auto fourCC = [](int code) {
        const char chars[] = {static_cast<char>((code >> 24) & 0xff), static_cast<char>((code >> 16) & 0xff),
                              static_cast<char>((code >> 8) & 0xff), static_cast<char>(code & 0xff), 0};
        return juce::String(chars);
    };
    CHECK_EQ(fourCC(JucePlugin_ManufacturerCode), juce::String("Snlb"));
    CHECK_EQ(fourCC(JucePlugin_PluginCode), juce::String("Btan"));
    CHECK_EQ(juce::String(JucePlugin_AUMainType), juce::String(static_cast<int>('aufx')));
    CHECK_EQ(juce::String(JucePlugin_VersionString), juce::String("0.1.1"));
    CHECK(!processor.acceptsMidi());
    CHECK(!processor.producesMidi());
    CHECK(!processor.isMidiEffect());
    CHECK_EQ(JucePlugin_IsSynth, 0);
    CHECK_EQ(JucePlugin_WantsMidiInput, 0);
    CHECK_EQ(JucePlugin_ProducesMidiOutput, 0);
}

TEST("plugin.bus_layouts_mono_and_stereo_only") {
    BeatAnalyzerProcessor processor;
    const auto supports = [&](const juce::AudioChannelSet& in, const juce::AudioChannelSet& out) {
        juce::AudioProcessor::BusesLayout buses;
        buses.inputBuses.add(in);
        buses.outputBuses.add(out);
        return processor.checkBusesLayoutSupported(buses);
    };
    CHECK(supports(juce::AudioChannelSet::mono(), juce::AudioChannelSet::mono()));
    CHECK(supports(juce::AudioChannelSet::stereo(), juce::AudioChannelSet::stereo()));
    CHECK(!supports(juce::AudioChannelSet::mono(), juce::AudioChannelSet::stereo()));
    CHECK(!supports(juce::AudioChannelSet::stereo(), juce::AudioChannelSet::mono()));
    CHECK(!supports(juce::AudioChannelSet::create5point1(), juce::AudioChannelSet::create5point1()));
    CHECK_EQ(processor.getTotalNumInputChannels(), 2);  // default layout is stereo
}

TEST("plugin.stereo_passes_through_bit_exact") {
    auto processor = makeProcessor(juce::AudioChannelSet::stereo());
    juce::MidiBuffer midi;
    for (unsigned block = 0; block < 20; ++block) {
        auto buffer = noise(2, kBlock, 1.2f, block);  // includes overs: must not be clipped
        juce::AudioBuffer<float> original(buffer);
        processor->processBlock(buffer, midi);
        CHECK(identical(buffer, original));
    }
    processor->releaseResources();
}

TEST("plugin.mono_passes_through_bit_exact") {
    auto processor = makeProcessor(juce::AudioChannelSet::mono());
    CHECK_EQ(processor->getTotalNumInputChannels(), 1);
    CHECK_EQ(processor->meteredChannels(), 1);
    juce::MidiBuffer midi;
    auto buffer = noise(1, kBlock, 0.5f, 7);
    juce::AudioBuffer<float> original(buffer);
    processor->processBlock(buffer, midi);
    CHECK(identical(buffer, original));
    processor->releaseResources();
}

TEST("plugin.odd_block_sizes_and_empty_blocks") {
    auto processor = makeProcessor(juce::AudioChannelSet::stereo());
    juce::MidiBuffer midi;
    for (int samples : {0, 1, 7, 255, 256, 257, kBlock}) {
        auto buffer = noise(2, samples, 0.3f, static_cast<unsigned>(samples));
        juce::AudioBuffer<float> original(buffer);
        processor->processBlock(buffer, midi);
        CHECK(identical(buffer, original));
    }
    processor->releaseResources();
}

TEST("plugin.process_block_never_allocates") {
    auto processor = makeProcessor(juce::AudioChannelSet::stereo());
    juce::MidiBuffer midi;
    midi.ensureSize(256);
    auto buffer = noise(2, kBlock, 0.5f, 3);
    trackedAllocations.store(0);
    tracking = true;
    for (int i = 0; i < 200; ++i) processor->processBlock(buffer, midi);
    tracking = false;
    CHECK_EQ(trackedAllocations.load(), 0);

    setEnabled(*processor, false);
    tracking = true;
    for (int i = 0; i < 50; ++i) processor->processBlock(buffer, midi);
    tracking = false;
    CHECK_EQ(trackedAllocations.load(), 0);
    processor->releaseResources();
}

TEST("plugin.midi_is_dropped") {
    auto processor = makeProcessor(juce::AudioChannelSet::stereo());
    juce::MidiBuffer midi;
    midi.addEvent(juce::MidiMessage::noteOn(1, 60, 0.8f), 0);
    auto buffer = noise(2, 64, 0.1f, 1);
    processor->processBlock(buffer, midi);
    CHECK(midi.isEmpty());
    processor->releaseResources();
}

TEST("plugin.input_level_is_measured_off_the_audio_thread") {
    auto processor = makeProcessor(juce::AudioChannelSet::stereo());
    juce::MidiBuffer midi;
    juce::AudioBuffer<float> buffer(2, kBlock);
    buffer.clear();
    buffer.setSample(0, 100, 0.5f);
    buffer.setSample(1, 200, -0.25f);
    processor->processBlock(buffer, midi);
    auto peaks = processor->takeInputPeaks();
    CHECK_EQ(peaks[0], 0.5f);
    CHECK_EQ(peaks[1], 0.25f);
    peaks = processor->takeInputPeaks();  // taking resets
    CHECK_EQ(peaks[0], 0.0f);

    processor->releaseResources();  // joins the worker, which drains the FIFO first
    CHECK_EQ(processor->pipeline().framesAnalyzed(), std::int64_t{kBlock});
}

TEST("plugin.disabled_passes_audio_but_does_not_measure") {
    auto processor = makeProcessor(juce::AudioChannelSet::stereo());
    setEnabled(*processor, false);
    CHECK(!processor->analysisEnabled());
    juce::MidiBuffer midi;
    auto buffer = noise(2, kBlock, 0.8f, 11);
    juce::AudioBuffer<float> original(buffer);
    processor->processBlock(buffer, midi);
    CHECK(identical(buffer, original));
    CHECK_EQ(processor->takeInputPeaks()[0], 0.0f);
    processor->releaseResources();
    CHECK_EQ(processor->pipeline().framesAnalyzed(), std::int64_t{0});
}

TEST("plugin.state_round_trip_restores_parameter") {
    BeatAnalyzerProcessor source;
    setEnabled(source, false);
    juce::MemoryBlock saved;
    source.getStateInformation(saved);
    CHECK(saved.getSize() > 0);

    BeatAnalyzerProcessor restored;
    CHECK(restored.analysisEnabled());
    restored.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
    CHECK(!restored.analysisEnabled());

    setEnabled(source, true);
    source.getStateInformation(saved);
    restored.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
    CHECK(restored.analysisEnabled());
}

TEST("plugin.bad_or_old_state_keeps_defaults") {
    BeatAnalyzerProcessor processor;
    const char garbage[] = "not a plugin state";
    processor.setStateInformation(garbage, static_cast<int>(sizeof(garbage)));
    CHECK(processor.analysisEnabled());
    processor.setStateInformation(nullptr, 0);
    CHECK(processor.analysisEnabled());

    // Phase 1 builds wrote a bare ValueTree stream with only a version.
    juce::ValueTree old("SonicLabBeatAnalyzer");
    old.setProperty("version", 1, nullptr);
    juce::MemoryOutputStream stream;
    old.writeToStream(stream);
    processor.setStateInformation(stream.getData(), static_cast<int>(stream.getDataSize()));
    CHECK(processor.analysisEnabled());
}

TEST("plugin.editor_builds_and_paints_headless") {
    auto processor = makeProcessor(juce::AudioChannelSet::stereo());
    juce::MidiBuffer midi;
    auto buffer = noise(2, kBlock, 0.5f, 5);
    processor->processBlock(buffer, midi);
    std::unique_ptr<juce::AudioProcessorEditor> editor(processor->createEditorIfNeeded());
    CHECK(editor != nullptr);
    CHECK_EQ(editor->getWidth(), 560);
    auto image = editor->createComponentSnapshot(editor->getLocalBounds());
    CHECK(image.isValid());
    if (const char* path = std::getenv("SONICLAB_EDITOR_SNAPSHOT")) {
        juce::File file(path);
        file.deleteFile();
        juce::FileOutputStream out(file);
        juce::PNGImageFormat().writeImageToStream(image, out);
    }
    processor->editorBeingDeleted(editor.get());
    editor.reset();
    processor->releaseResources();
}

// ---- dropped-file analysis

namespace {

// A 24-bit WAV in the temp folder, deleted when the test ends: a sine of
// ``amplitude`` on the left channel and half that on the right.
struct TempWav {
    juce::File file;
    TempWav(double rate, int channels, double seconds, float amplitude, const juce::String& extension = ".wav") {
        file = juce::File::getSpecialLocation(juce::File::tempDirectory)
                   .getNonexistentChildFile("soniclab_test", extension, false);
        const auto frames = static_cast<int>(rate * seconds);
        juce::AudioBuffer<float> audio(channels, frames);
        for (int ch = 0; ch < channels; ++ch)
            for (int i = 0; i < frames; ++i)
                audio.setSample(ch, i, amplitude / static_cast<float>(ch + 1) *
                                           std::sin(2.0f * juce::MathConstants<float>::pi * 440.0f *
                                                    static_cast<float>(i) / static_cast<float>(rate)));
        juce::WavAudioFormat wav;
        auto stream = file.createOutputStream();
        CHECK(stream != nullptr);
        std::unique_ptr<juce::AudioFormatWriter> writer(
            wav.createWriterFor(stream.release(), rate, static_cast<unsigned int>(channels), 24, {}, 0));
        CHECK(writer != nullptr);
        CHECK(writer->writeFromAudioSampleBuffer(audio, 0, frames));
    }
    ~TempWav() { file.deleteFile(); }
};

FileAnalysisResult waitForFile(FileAnalysis& analysis) {
    for (int i = 0; i < 2000; ++i) {
        auto result = analysis.snapshot();
        if (result.status != FileAnalysisResult::Status::Analysing) return result;
        juce::Thread::sleep(5);
    }
    throw soniclab::test::Failure{"file analysis did not finish within 10 s"};
}

}  // namespace

TEST("file_analysis.reads_wav_and_builds_overview") {
    TempWav wav(44100.0, 2, 1.5, 0.5f);
    FileAnalysis analysis(&BeatAnalyzerProcessor::makeAnalyzer);
    CHECK(FileAnalysis::canAnalyse(wav.file));
    analysis.start(wav.file);
    const auto result = waitForFile(analysis);
    CHECK(result.status == FileAnalysisResult::Status::Done);
    CHECK_EQ(result.channels, 2);
    CHECK_EQ(result.sampleRate, 44100.0);
    CHECK_EQ(result.lengthFrames, std::int64_t{66150});
    CHECK(std::abs(result.lengthSeconds() - 1.5) < 1e-9);
    CHECK(std::abs(result.peak - 0.5f) < 1e-3f);
    CHECK_EQ(result.overview.size(), std::size_t{FileAnalysis::kOverviewBuckets});
    // Mono downmix of 0.5 and 0.25 sines swings +-0.375. A bucket is shorter
    // than one cycle, so check the envelope over all buckets.
    float lowest = 0.0f;
    float highest = 0.0f;
    for (const auto& [low, high] : result.overview) {
        CHECK(low <= high);
        lowest = std::min(lowest, low);
        highest = std::max(highest, high);
    }
    CHECK(std::abs(highest - 0.375f) < 1e-3f);
    CHECK(std::abs(lowest + 0.375f) < 1e-3f);
    CHECK(result.markers.empty());  // the placeholder analyzer reports nothing
    CHECK_EQ(analysis.progress(), 1.0f);
}

TEST("file_analysis.reports_unreadable_and_missing_files") {
    FileAnalysis analysis(&BeatAnalyzerProcessor::makeAnalyzer);
    const auto garbage = juce::File::getSpecialLocation(juce::File::tempDirectory)
                             .getNonexistentChildFile("soniclab_garbage", ".wav", false);
    CHECK(garbage.replaceWithText("not audio at all"));
    analysis.start(garbage);
    auto result = waitForFile(analysis);
    CHECK(result.status == FileAnalysisResult::Status::Failed);
    CHECK(result.error.isNotEmpty());
    garbage.deleteFile();

    analysis.start(garbage);
    result = waitForFile(analysis);
    CHECK(result.status == FileAnalysisResult::Status::Failed);
    CHECK(result.error.contains("no longer exists"));

    CHECK(!FileAnalysis::canAnalyse(juce::File("/tmp/notes.txt")));
    CHECK(FileAnalysis::canAnalyse(juce::File("/tmp/LOUD.WAV")));
    CHECK(FileAnalysis::supportedFormatsText().contains("WAV"));
}

TEST("file_analysis.restart_and_clear_cancel_cleanly") {
    TempWav longer(48000.0, 2, 20.0, 0.25f);
    TempWav shorter(22050.0, 1, 0.5, 0.8f);
    FileAnalysis analysis(&BeatAnalyzerProcessor::makeAnalyzer);
    const auto before = analysis.version();
    analysis.start(longer.file);
    analysis.start(shorter.file);  // cancels the first run
    const auto result = waitForFile(analysis);
    CHECK(analysis.version() != before);
    CHECK(result.status == FileAnalysisResult::Status::Done);
    CHECK(result.file == shorter.file);
    CHECK_EQ(result.channels, 1);
    CHECK(std::abs(result.peak - 0.8f) < 1e-3f);

    analysis.start(longer.file);
    analysis.clear();
    CHECK(analysis.snapshot().status == FileAnalysisResult::Status::Empty);
}

TEST("plugin.editor_accepts_dropped_audio_files") {
    TempWav wav(48000.0, 2, 2.0, 0.5f);
    auto processor = makeProcessor(juce::AudioChannelSet::stereo());
    std::unique_ptr<juce::AudioProcessorEditor> editor(processor->createEditorIfNeeded());
    auto* beatEditor = dynamic_cast<BeatAnalyzerEditor*>(editor.get());
    CHECK(beatEditor != nullptr);
    CHECK(beatEditor->isInterestedInFileDrag({wav.file.getFullPathName()}));
    CHECK(!beatEditor->isInterestedInFileDrag({"/tmp/readme.txt"}));
    CHECK(beatEditor->isInterestedInFileDrag({"/tmp/readme.txt", wav.file.getFullPathName()}));

    // Live processing is untouched while a file is analysed.
    beatEditor->filesDropped({"/tmp/readme.txt", wav.file.getFullPathName()}, 10, 10);
    juce::MidiBuffer midi;
    auto buffer = noise(2, kBlock, 0.5f, 9);
    const auto reference = buffer;
    tracking = true;
    processor->processBlock(buffer, midi);
    tracking = false;
    CHECK_EQ(trackedAllocations.exchange(0), 0);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < kBlock; ++i) CHECK_EQ(buffer.getSample(ch, i), reference.getSample(ch, i));

    const auto result = waitForFile(processor->fileAnalysis());
    CHECK(result.status == FileAnalysisResult::Status::Done);
    CHECK(result.file == wav.file);

    auto image = editor->createComponentSnapshot(editor->getLocalBounds());
    CHECK(image.isValid());
    if (const char* path = std::getenv("SONICLAB_EDITOR_FILE_SNAPSHOT")) {
        juce::File file(path);
        file.deleteFile();
        juce::FileOutputStream out(file);
        juce::PNGImageFormat().writeImageToStream(image, out);
    }

    // The dropped file is a view, not a setting: it is not saved with the state.
    juce::MemoryBlock state;
    processor->getStateInformation(state);
    auto xml = juce::AudioProcessor::getXmlFromBinary(state.getData(), static_cast<int>(state.getSize()));
    CHECK(xml != nullptr);
    CHECK(!xml->toString().contains(wav.file.getFileName()));

    processor->editorBeingDeleted(editor.get());
    editor.reset();
    processor->releaseResources();
}

int main(int argc, char** argv) {
    juce::ScopedJuceInitialiser_GUI juce;
    const char* prefix = argc > 1 ? argv[1] : "";
    int run = 0;
    int failed = 0;
    for (const auto& test : soniclab::test::registry()) {
        if (std::strncmp(test.name.c_str(), prefix, std::strlen(prefix)) != 0) continue;
        ++run;
        try {
            test.body();
            std::cout << "[ ok ] " << test.name << '\n';
        } catch (const soniclab::test::Failure& failure) {
            ++failed;
            std::cout << "[FAIL] " << test.name << "\n       " << failure.message << '\n';
        }
    }
    std::cout << run << " tests, " << failed << " failed\n";
    return run > 0 && failed == 0 ? 0 : 1;
}
