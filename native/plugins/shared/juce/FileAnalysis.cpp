#include "FileAnalysis.h"

#include <algorithm>
#include <cmath>

namespace soniclab::plugin {
namespace {
constexpr int kReadFrames = 8192;

// Lower-case extensions without dots, from every format JUCE registers here.
const juce::StringArray& supportedExtensions() {
    static const juce::StringArray extensions = [] {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        juce::StringArray out;
        for (auto* format : formats)
            for (auto extension : format->getFileExtensions())
                out.addIfNotAlreadyThere(extension.trimCharactersAtStart(".").toLowerCase());
        return out;
    }();
    return extensions;
}

class CollectingSink final : public MarkerSink {
public:
    explicit CollectingSink(std::vector<AnalysisMarker>& markers) : markers_(markers) {}
    void emit(const AnalysisMarker& marker) override {
        if (markers_.size() < FileAnalysis::kMaxMarkers) markers_.push_back(marker);
    }

private:
    std::vector<AnalysisMarker>& markers_;
};
}  // namespace

FileAnalysis::FileAnalysis(AnalyzerFactory makeAnalyzer) : makeAnalyzer_(std::move(makeAnalyzer)) {}

FileAnalysis::~FileAnalysis() { cancel(); }

bool FileAnalysis::canAnalyse(const juce::File& file) {
    return supportedExtensions().contains(file.getFileExtension().trimCharactersAtStart(".").toLowerCase());
}

juce::String FileAnalysis::supportedFormatsText() {
    juce::StringArray shown;
    for (const auto& extension : supportedExtensions())
        if (extension != "aif" && extension != "bwf") shown.add(extension.toUpperCase());
    return shown.joinIntoString(", ");
}

juce::String FileAnalysis::chooserPattern() {
    juce::StringArray patterns;
    for (const auto& extension : supportedExtensions()) patterns.add("*." + extension);
    return patterns.joinIntoString(";");
}

void FileAnalysis::start(const juce::File& file) {
    cancel();
    FileAnalysisResult starting;
    starting.status = FileAnalysisResult::Status::Analysing;
    starting.file = file;
    progress_.store(0.0f, std::memory_order_relaxed);
    publish(std::move(starting));
    cancel_.store(false);
    worker_ = std::thread([this, file] { run(file); });
}

void FileAnalysis::clear() {
    cancel();
    publish({});
}

FileAnalysisResult FileAnalysis::snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return result_;
}

void FileAnalysis::cancel() {
    cancel_.store(true);
    if (worker_.joinable()) worker_.join();
}

void FileAnalysis::publish(FileAnalysisResult result) {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        result_ = std::move(result);
    }
    version_.fetch_add(1, std::memory_order_acq_rel);
}

void FileAnalysis::run(juce::File file) {
    FileAnalysisResult result;
    result.file = file;
    const auto fail = [&](const juce::String& message) {
        result.status = FileAnalysisResult::Status::Failed;
        result.error = message;
        publish(std::move(result));
    };

    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
    if (reader == nullptr)
        return fail(file.existsAsFile() ? "Could not read this file as audio." : "The file no longer exists.");
    if (reader->lengthInSamples <= 0 || reader->numChannels == 0 || reader->sampleRate <= 0.0)
        return fail("The file contains no audio.");

    result.formatName = reader->getFormatName();
    result.sampleRate = reader->sampleRate;
    result.channels = static_cast<int>(reader->numChannels);
    result.lengthFrames = reader->lengthInSamples;
    result.overview.assign(kOverviewBuckets, {0.0f, 0.0f});
    std::vector<bool> touched(kOverviewBuckets, false);

    auto analyzer = makeAnalyzer_();
    analyzer->reset(result.sampleRate);
    CollectingSink sink(result.markers);
    AudioChunk chunk;

    juce::AudioBuffer<float> block(result.channels, kReadFrames);
    std::vector<float> mono(kReadFrames);
    const auto length = result.lengthFrames;
    for (std::int64_t position = 0; position < length; position += kReadFrames) {
        if (cancel_.load()) return;
        const auto frames = static_cast<int>(std::min<std::int64_t>(kReadFrames, length - position));
        if (!reader->read(&block, 0, frames, position, true, true)) return fail("Reading the file failed part way.");

        std::fill(mono.begin(), mono.begin() + frames, 0.0f);
        const auto gain = 1.0f / static_cast<float>(result.channels);
        for (int ch = 0; ch < result.channels; ++ch) {
            result.peak = std::max(result.peak, block.getMagnitude(ch, 0, frames));
            juce::FloatVectorOperations::addWithMultiply(mono.data(), block.getReadPointer(ch), gain, frames);
        }

        for (int i = 0; i < frames; ++i) {
            const auto frame = position + i;
            const auto bucket = static_cast<std::size_t>(frame * kOverviewBuckets / length);
            auto& [low, high] = result.overview[bucket];
            const auto sample = mono[static_cast<std::size_t>(i)];
            if (!touched[bucket]) {
                low = high = sample;
                touched[bucket] = true;
            } else {
                low = std::min(low, sample);
                high = std::max(high, sample);
            }
        }

        // Same chunking and stamping the live pipeline uses, so an analyzer
        // sees a file exactly as it would see that audio played in.
        for (int offset = 0; offset < frames; offset += kChunkFrames) {
            chunk.firstSample = position + offset;
            chunk.frames = std::min(kChunkFrames, frames - offset);
            std::copy_n(mono.begin() + offset, chunk.frames, chunk.mono);
            analyzer->analyze(chunk, sink);
        }
        progress_.store(static_cast<float>(static_cast<double>(position + frames) / static_cast<double>(length)),
                        std::memory_order_relaxed);
    }

    analyzer->finish(sink);
    result.summary = analyzer->summary();
    result.status = FileAnalysisResult::Status::Done;
    publish(std::move(result));
}

}  // namespace soniclab::plugin
