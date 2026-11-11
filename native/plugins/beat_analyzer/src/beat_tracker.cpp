#include "beat_analyzer/beat_tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace soniclab::beat_analyzer {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kFramesPerSecond = 100.0;
constexpr double kLiveWindowSeconds = 12.0;
constexpr double kLiveKeepSeconds = 16.0;
constexpr double kLiveMinimumSeconds = 4.0;
constexpr double kLiveUpdateSeconds = 0.5;
constexpr double kLiveLatencySeconds = 0.25;

// In-place iterative radix-2 FFT; ``data.size()`` is a power of two.
void fft(std::vector<std::complex<float>>& data) {
    const auto n = data.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        auto bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
    for (std::size_t length = 2; length <= n; length <<= 1) {
        const auto angle = -2.0 * kPi / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0; start < n; start += length) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < length / 2; ++k) {
                const auto even = data[start + k];
                const auto odd = data[start + k + length / 2] * std::complex<float>(w);
                data[start + k] = even + odd;
                data[start + k + length / 2] = even - odd;
                w *= step;
            }
        }
    }
}

int toBin(double hz, int size, double rate) { return static_cast<int>(std::lround(hz * size / rate)); }

float mean(const std::vector<float>& values) {
    return values.empty() ? 0.0f
                          : static_cast<float>(std::accumulate(values.begin(), values.end(), 0.0) /
                                               static_cast<double>(values.size()));
}

// Zero mean, unit variance; all zeros when the values do not vary.
void standardise(std::vector<float>& values) {
    const auto m = mean(values);
    double variance = 0.0;
    for (const auto v : values) variance += (v - m) * (v - m);
    const auto sd = values.empty() ? 0.0 : std::sqrt(variance / static_cast<double>(values.size()));
    for (auto& v : values) v = sd > 1e-9 ? static_cast<float>((v - m) / sd) : 0.0f;
}

// Parabolic peak refinement around index i of ``values``.
double refinePeak(const std::vector<float>& values, std::size_t i) {
    if (i == 0 || i + 1 >= values.size()) return static_cast<double>(i);
    const double a = values[i - 1];
    const double b = values[i];
    const double c = values[i + 1];
    const double denominator = a - 2.0 * b + c;
    if (std::abs(denominator) < 1e-12) return static_cast<double>(i);
    return static_cast<double>(i) + std::clamp(0.5 * (a - c) / denominator, -0.5, 0.5);
}

}  // namespace

// ------------------------------------------------------------- features

void OnsetFeatures::reset(double sampleRate) {
    rate_ = sampleRate;
    hop_ = std::max(1, static_cast<int>(std::lround(sampleRate / kFramesPerSecond)));
    size_ = sampleRate <= 50000.0 ? 2048 : sampleRate <= 100000.0 ? 4096 : 8192;
    window_.resize(static_cast<std::size_t>(size_));
    for (int i = 0; i < size_; ++i)
        window_[static_cast<std::size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / size_));
    // Half a window of silence first, so frame i is centred on sample i * hop.
    pending_.assign(static_cast<std::size_t>(size_ / 2), 0.0f);
    spectrum_.assign(static_cast<std::size_t>(size_), {});
    previous_.clear();
    const auto nyquist = size_ / 2;
    bandEdges_.clear();
    lowBands_ = 0;
    for (int b = 0; b <= kBands; ++b) {
        const auto hz = 40.0 * std::pow(11000.0 / 40.0, static_cast<double>(b) / kBands);
        auto bin = std::clamp(toBin(hz, size_, rate_), 1, nyquist);
        if (!bandEdges_.empty()) bin = std::max(bin, bandEdges_.back() + 1);
        bandEdges_.push_back(bin);
        if (b > 0 && hz <= 160.0) lowBands_ = b;
    }
    lowBands_ = std::max(lowBands_, 1);
    first_ = 0;
    flux_.clear();
    lowFlux_.clear();
    bands_.clear();
}

void OnsetFeatures::push(const float* samples, int count) {
    if (size_ == 0 || count <= 0) return;
    pending_.insert(pending_.end(), samples, samples + count);
    std::size_t consumed = 0;
    while (pending_.size() - consumed >= static_cast<std::size_t>(size_)) {
        for (int i = 0; i < size_; ++i)
            spectrum_[static_cast<std::size_t>(i)] = pending_[consumed + static_cast<std::size_t>(i)] * window_[static_cast<std::size_t>(i)];
        computeFrame();
        consumed += static_cast<std::size_t>(hop_);
    }
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(consumed));
}

void OnsetFeatures::computeFrame() {
    fft(spectrum_);
    const auto bins = size_ / 2 + 1;
    // Band level: mean log-compressed magnitude. The compression decides
    // what counts as an onset: much stronger and quiet, wide-band hi-hats
    // outweigh the kick, which puts beats on the off-beat in dance music
    // (measured against Madmom, see scripts/compare_beats.py).
    constexpr float kCompression = 15.0f;
    const auto scale = 4.0f / static_cast<float>(size_);
    std::vector<float> current(static_cast<std::size_t>(kBands), 0.0f);
    for (int b = 0; b < kBands; ++b) {
        const auto from = bandEdges_[static_cast<std::size_t>(b)];
        const auto to = std::min(bins, std::max(from + 1, bandEdges_[static_cast<std::size_t>(b) + 1]));
        float sum = 0.0f;
        for (int k = from; k < to; ++k) sum += std::log1p(kCompression * std::abs(spectrum_[static_cast<std::size_t>(k)]) * scale);
        current[static_cast<std::size_t>(b)] = to > from ? sum / static_cast<float>(to - from) : 0.0f;
    }

    float flux = 0.0f;
    float low = 0.0f;
    if (!previous_.empty()) {
        for (int b = 0; b < kBands; ++b) {
            const auto rise = std::max(0.0f, current[static_cast<std::size_t>(b)] - previous_[static_cast<std::size_t>(b)]);
            flux += rise;
            if (b < lowBands_) low += rise;
        }
    }
    flux_.push_back(flux);
    lowFlux_.push_back(low);
    bands_.insert(bands_.end(), current.begin(), current.end());
    previous_ = std::move(current);
}

void OnsetFeatures::discardBefore(std::int64_t frame) {
    if (frame <= first_) return;
    const auto count = static_cast<std::size_t>(std::min<std::int64_t>(frame - first_, static_cast<std::int64_t>(flux_.size())));
    flux_.erase(flux_.begin(), flux_.begin() + static_cast<std::ptrdiff_t>(count));
    lowFlux_.erase(lowFlux_.begin(), lowFlux_.begin() + static_cast<std::ptrdiff_t>(count));
    bands_.erase(bands_.begin(), bands_.begin() + static_cast<std::ptrdiff_t>(count * kBands));
    first_ += static_cast<std::int64_t>(count);
}

// --------------------------------------------------------------- tempo

std::vector<float> normaliseEnvelope(const float* flux, std::size_t count) {
    std::vector<float> envelope(flux, flux + count);
    const auto m = mean(envelope);
    double variance = 0.0;
    for (const auto v : envelope) variance += (v - m) * (v - m);
    const auto sd = count > 1 ? std::sqrt(variance / static_cast<double>(count - 1)) : 0.0;
    for (auto& v : envelope) v = sd > 1e-9 ? static_cast<float>(v / sd) : 0.0f;
    return envelope;
}

TempoEstimate estimateTempo(const std::vector<float>& envelope, double framesPerSecond, double minBpm,
                            double maxBpm) {
    const auto n = envelope.size();
    if (framesPerSecond <= 0.0 || static_cast<double>(n) < 3.0 * framesPerSecond) return {};
    minBpm = std::clamp(minBpm, 20.0, 300.0);
    maxBpm = std::clamp(maxBpm, minBpm + 1.0, 400.0);

    const auto minLag = std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(60.0 * framesPerSecond / maxBpm)));
    auto maxLag = static_cast<std::size_t>(std::ceil(60.0 * framesPerSecond / minBpm));
    maxLag = std::min(maxLag, n / 2 - 1);
    if (minLag >= maxLag) return {};

    std::vector<float> x(envelope);
    const auto m = mean(x);
    for (auto& v : x) v -= m;
    const auto acLength = std::min(n - 1, 3 * maxLag + 2);
    std::vector<float> ac(acLength + 1);
    for (std::size_t lag = 0; lag <= acLength; ++lag) {
        double sum = 0.0;
        for (std::size_t t = 0; t + lag < n; ++t) sum += static_cast<double>(x[t]) * x[t + lag];
        ac[lag] = static_cast<float>(sum / static_cast<double>(n - lag));
    }
    if (ac[0] <= 1e-9f) return {};
    for (auto& v : ac) v /= ac.front();

    const auto at = [&](std::size_t lag) { return lag <= acLength ? ac[lag] : 0.0f; };
    std::vector<double> score(maxLag + 1, 0.0);
    std::size_t best = 0;
    for (auto lag = minLag; lag <= maxLag; ++lag) {
        // A beat period also repeats at two and three periods.
        const double harmonic = at(lag) + 0.5 * at(2 * lag) + 0.25 * at(3 * lag);
        const auto bpm = 60.0 * framesPerSecond / static_cast<double>(lag);
        const auto octaves = std::log2(bpm / 120.0);
        score[lag] = harmonic * std::exp(-0.5 * octaves * octaves);
        if (best == 0 || score[lag] > score[best]) best = lag;
    }
    if (score[best] <= 0.0) return {};

    // Octave check. Alternating kick and snare repeat most strongly every
    // two beats, which would halve the tempo. Move to a pulse two or three
    // times faster when it is a clear peak of its own and the prior likes
    // that tempo better.
    const auto prior = [&](std::size_t lag) {
        const auto octaves = std::log2(60.0 * framesPerSecond / static_cast<double>(lag) / 120.0);
        return std::exp(-0.5 * octaves * octaves);
    };
    for (bool moved = true; moved;) {
        moved = false;
        for (const std::size_t divisor : {std::size_t{2}, std::size_t{3}}) {
            const auto centre = (best + divisor / 2) / divisor;
            if (centre < minLag + 1 || centre + 1 > maxLag) continue;
            auto candidate = centre;
            for (auto lag = centre - 1; lag <= centre + 1; ++lag)
                if (ac[lag] > ac[candidate]) candidate = lag;
            const bool isPeak = ac[candidate] >= ac[candidate - 1] && ac[candidate] >= ac[candidate + 1];
            // A snare a sixth as strong as the kick still gives 0.3 here.
            if (isPeak && ac[candidate] >= 0.15f * ac[best] && ac[candidate] > 0.03f && prior(candidate) > prior(best)) {
                best = candidate;
                moved = true;
                break;
            }
        }
    }

    // Sub-frame period from the peaks at one and two periods.
    auto peak = best;
    for (auto lag = best > minLag ? best - 1 : best; lag <= std::min(best + 1, acLength); ++lag)
        if (ac[lag] > ac[peak]) peak = lag;
    double period = refinePeak(ac, peak);
    if (2 * best + 2 <= acLength) {
        auto second = 2 * best - 2;
        for (auto lag = 2 * best - 2; lag <= 2 * best + 2; ++lag)
            if (ac[lag] > ac[second]) second = lag;
        period = 0.5 * (period + refinePeak(ac, second) / 2.0);
    }

    // Confidence: how periodic the envelope is at this lag, and how far
    // this tempo beats the best one not related to it by a simple ratio.
    double alternative = 0.0;
    for (auto lag = minLag; lag <= maxLag; ++lag) {
        const auto ratio = static_cast<double>(lag) / static_cast<double>(best);
        bool related = false;
        for (const double r : {1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0, 3.0})
            related = related || std::abs(ratio - r) <= 0.06 * r;
        if (!related) alternative = std::max(alternative, score[lag]);
    }
    const auto periodicity = std::clamp(static_cast<double>(ac[best]) / 0.3, 0.0, 1.0);
    const auto distinct = std::clamp(2.0 * (1.0 - std::max(0.0, alternative) / score[best]), 0.0, 1.0);

    TempoEstimate out;
    out.periodFrames = period;
    out.bpm = 60.0 * framesPerSecond / period;
    out.confidence = static_cast<float>(periodicity * distinct);
    return out;
}

// --------------------------------------------------------------- beats

std::vector<double> trackBeats(const std::vector<float>& envelope, double periodFrames, double tightness) {
    const auto n = static_cast<std::ptrdiff_t>(envelope.size());
    if (n < 2 || periodFrames <= 1.0) return {};

    // Local score: the envelope smoothed by a Gaussian of period / 32.
    const auto sigma = periodFrames / 32.0;
    const auto radius = std::max<std::ptrdiff_t>(1, static_cast<std::ptrdiff_t>(std::ceil(3.0 * sigma)));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    for (auto d = -radius; d <= radius; ++d)
        kernel[static_cast<std::size_t>(d + radius)] = std::exp(-0.5 * (static_cast<double>(d) / sigma) * (static_cast<double>(d) / sigma));
    std::vector<float> local(envelope.size(), 0.0f);
    for (std::ptrdiff_t t = 0; t < n; ++t) {
        double sum = 0.0;
        for (auto d = std::max(-radius, -t); d <= std::min(radius, n - 1 - t); ++d)
            sum += envelope[static_cast<std::size_t>(t + d)] * kernel[static_cast<std::size_t>(d + radius)];
        local[static_cast<std::size_t>(t)] = static_cast<float>(sum);
    }
    const auto threshold = 0.01f * *std::max_element(local.begin(), local.end());

    // cum[t] = local[t] + best over previous beats half to two periods back
    // of cum[previous] minus a penalty for straying from the period.
    const auto nearest = std::max<std::ptrdiff_t>(1, std::lround(periodFrames / 2.0));
    const auto farthest = std::max(nearest, static_cast<std::ptrdiff_t>(std::lround(2.0 * periodFrames)));
    std::vector<double> penalty(static_cast<std::size_t>(farthest + 1), 0.0);
    for (auto d = nearest; d <= farthest; ++d) {
        const auto stray = std::log(static_cast<double>(d) / periodFrames);
        penalty[static_cast<std::size_t>(d)] = tightness * stray * stray;
    }
    std::vector<double> cum(envelope.size(), 0.0);
    std::vector<std::ptrdiff_t> back(envelope.size(), -1);
    bool firstBeat = true;
    for (std::ptrdiff_t t = 0; t < n; ++t) {
        double best = -std::numeric_limits<double>::infinity();
        std::ptrdiff_t from = -1;
        for (auto d = nearest; d <= farthest; ++d) {
            const auto previous = t - d;
            const auto value = -penalty[static_cast<std::size_t>(d)] + (previous >= 0 ? cum[static_cast<std::size_t>(previous)] : 0.0);
            if (value > best) {
                best = value;
                from = previous;
            }
        }
        cum[static_cast<std::size_t>(t)] = local[static_cast<std::size_t>(t)] + best;
        if (firstBeat && local[static_cast<std::size_t>(t)] < threshold) continue;
        firstBeat = false;
        back[static_cast<std::size_t>(t)] = from >= 0 ? from : -1;
    }

    // Last beat: the last local maximum of cum above half the median maximum.
    std::vector<double> maxima;
    std::vector<bool> isMax(envelope.size(), false);
    for (std::ptrdiff_t t = 0; t < n; ++t) {
        const auto i = static_cast<std::size_t>(t);
        const bool left = t == 0 || cum[i] > cum[i - 1];
        const bool right = t == n - 1 || cum[i] >= cum[i + 1];
        if (left && right) {
            isMax[i] = true;
            maxima.push_back(cum[i]);
        }
    }
    if (maxima.empty()) return {};
    std::nth_element(maxima.begin(), maxima.begin() + static_cast<std::ptrdiff_t>(maxima.size() / 2), maxima.end());
    const auto median = maxima[maxima.size() / 2];
    std::ptrdiff_t last = -1;
    for (std::ptrdiff_t t = n - 1; t >= 0; --t)
        if (isMax[static_cast<std::size_t>(t)] && 2.0 * cum[static_cast<std::size_t>(t)] > median) {
            last = t;
            break;
        }
    if (last < 0) return {};

    std::vector<double> beats;
    for (auto t = last; t >= 0; t = back[static_cast<std::size_t>(t)]) beats.push_back(static_cast<double>(t));
    std::reverse(beats.begin(), beats.end());

    // Trim weak beats at both ends (fade-ins, fade-outs, silence).
    double energy = 0.0;
    for (const auto b : beats) energy += static_cast<double>(local[static_cast<std::size_t>(b)]) * local[static_cast<std::size_t>(b)];
    const auto floor = 0.5 * std::sqrt(energy / static_cast<double>(beats.size()));
    const auto weak = [&](double b) { return local[static_cast<std::size_t>(b)] <= floor; };
    while (!beats.empty() && weak(beats.front())) beats.erase(beats.begin());
    while (!beats.empty() && weak(beats.back())) beats.pop_back();
    return beats;
}

// --------------------------------------------------------------- meter

MeterEstimate estimateMeter(const std::vector<double>& beats, const float* lowFlux, const float* flux,
                            const float* bands, std::size_t frames, int fixedBeatsPerBar) {
    const auto count = beats.size();
    if (count < 6 || frames == 0) return {};

    const auto frameOf = [&](double beat) {
        return static_cast<std::size_t>(std::clamp<double>(std::round(beat), 0.0, static_cast<double>(frames - 1)));
    };
    const auto peakAround = [&](const float* values, double beat) {
        const auto centre = frameOf(beat);
        float best = 0.0f;
        for (auto i = centre >= 2 ? centre - 2 : 0; i <= std::min(frames - 1, centre + 2); ++i) best = std::max(best, values[i]);
        return best;
    };
    const auto bandMean = [&](double from, double to, std::vector<double>& out) {
        out.assign(OnsetFeatures::kBands, 0.0);
        const auto a = frameOf(from);
        const auto b = std::max(a + 1, frameOf(to));
        for (auto i = a; i < b && i < frames; ++i)
            for (int k = 0; k < OnsetFeatures::kBands; ++k) out[static_cast<std::size_t>(k)] += bands[i * OnsetFeatures::kBands + static_cast<std::size_t>(k)];
        for (auto& v : out) v /= static_cast<double>(b - a);
    };

    // Per beat: bass onset, overall onset, and how much the spectrum of the
    // beat after it differs from the beat before (chord changes).
    std::vector<float> bass(count), onset(count), change(count, 0.0f);
    std::vector<double> before, after;
    for (std::size_t j = 0; j < count; ++j) {
        bass[j] = peakAround(lowFlux, beats[j]);
        onset[j] = peakAround(flux, beats[j]);
        if (j == 0 || j + 1 == count) continue;
        bandMean(beats[j - 1], beats[j], before);
        bandMean(beats[j], beats[j + 1], after);
        double distance = 0.0;
        for (std::size_t k = 0; k < before.size(); ++k) distance += (after[k] - before[k]) * (after[k] - before[k]);
        change[j] = static_cast<float>(std::sqrt(distance));
    }
    // The first and last beat have no neighbour on one side: give them the average.
    const auto inner = std::accumulate(change.begin() + 1, change.end() - 1, 0.0f) / static_cast<float>(count - 2);
    change.front() = change.back() = inner;
    standardise(bass);
    standardise(onset);
    standardise(change);
    std::vector<float> accent(count);
    for (std::size_t j = 0; j < count; ++j) accent[j] = 0.8f * bass[j] + 1.0f * change[j] + 0.3f * onset[j];

    struct Fit {
        int beatsPerBar = 0;
        int phase = 0;
        double best = -1e9;
        double second = -1e9;
    };
    const auto fit = [&](int m) {
        Fit out;
        out.beatsPerBar = m;
        if (count < static_cast<std::size_t>(2 * m)) return out;
        for (int p = 0; p < m; ++p) {
            double on = 0.0, off = 0.0;
            int onCount = 0, offCount = 0;
            for (std::size_t j = 0; j < count; ++j) {
                if (static_cast<int>(j % static_cast<std::size_t>(m)) == p) {
                    on += accent[j];
                    ++onCount;
                } else {
                    off += accent[j];
                    ++offCount;
                }
            }
            const auto contrast = on / std::max(1, onCount) - off / std::max(1, offCount);
            if (contrast > out.best) {
                out.second = out.best;
                out.best = contrast;
                out.phase = p;
            } else if (contrast > out.second) {
                out.second = contrast;
            }
        }
        return out;
    };

    MeterEstimate out;
    Fit chosen;
    if (fixedBeatsPerBar >= 2) {
        chosen = fit(fixedBeatsPerBar);
        if (chosen.best <= -1e8) return {};
        out.meterConfidence = 1.0f;
    } else {
        const auto three = fit(3);
        const auto four = fit(4);
        if (four.best <= -1e8) return {};
        // Four beats per bar unless three fits clearly better.
        chosen = three.best > four.best + 0.15 ? three : four;
        const auto margin = std::abs(four.best - three.best);
        out.meterConfidence = static_cast<float>(std::clamp(margin / 0.6, 0.0, 1.0) * std::clamp(chosen.best / 0.6, 0.0, 1.0));
    }
    out.beatsPerBar = chosen.beatsPerBar;
    out.phase = chosen.phase;
    out.downbeatConfidence = static_cast<float>(std::clamp(chosen.best - chosen.second, 0.0, 1.0));
    return out;
}

// -------------------------------------------------------------- tracker

BeatTracker::BeatTracker(Mode mode, BeatTrackerSettings settings) : mode_(mode), settings_(settings) {}

double BeatTracker::setting(const std::atomic<float>* value, double fallback) const {
    return value != nullptr ? static_cast<double>(value->load(std::memory_order_relaxed)) : fallback;
}

void BeatTracker::reset(double sampleRate) {
    features_.reset(sampleRate);
    haveBase_ = false;
    sampleBase_ = 0;
    lastUpdate_ = 0;
    lastEmitted_ = -1e9;
    lastPosition_ = 0;
    lastBeatsPerBar_ = 0;
    smoothedBpm_ = 0.0;
    publish({}, {});
}

void BeatTracker::analyze(const AudioChunk& chunk, MarkerSink& sink) {
    if (!haveBase_) {
        sampleBase_ = chunk.firstSample;
        haveBase_ = true;
    }
    features_.push(chunk.mono, chunk.frames);
    if (mode_ != Mode::Live) return;
    const auto fps = features_.framesPerSecond();
    const auto end = features_.endFrame();
    if (static_cast<double>(end - lastUpdate_) < kLiveUpdateSeconds * fps) return;
    if (static_cast<double>(end - features_.firstFrame()) < kLiveMinimumSeconds * fps) return;
    update(sink, false);
    lastUpdate_ = end;
    features_.discardBefore(end - static_cast<std::int64_t>(kLiveKeepSeconds * fps));
}

void BeatTracker::finish(MarkerSink& sink) {
    if (mode_ == Mode::File) update(sink, true);
}

AnalysisSummary BeatTracker::summary() const {
    AnalysisSummary out;
    out.tempoBpm = bpm_.load(std::memory_order_relaxed);
    out.tempoConfidence = tempoConfidence_.load(std::memory_order_relaxed);
    out.beatsPerBar = beatsPerBar_.load(std::memory_order_relaxed);
    out.meterConfidence = meterConfidence_.load(std::memory_order_relaxed);
    out.downbeatConfidence = downbeatConfidence_.load(std::memory_order_relaxed);
    return out;
}

void BeatTracker::publish(const TempoEstimate& tempo, const MeterEstimate& meter) {
    bpm_.store(static_cast<float>(tempo.bpm), std::memory_order_relaxed);
    tempoConfidence_.store(tempo.confidence, std::memory_order_relaxed);
    beatsPerBar_.store(meter.beatsPerBar, std::memory_order_relaxed);
    meterConfidence_.store(meter.meterConfidence, std::memory_order_relaxed);
    downbeatConfidence_.store(meter.downbeatConfidence, std::memory_order_relaxed);
}

void BeatTracker::update(MarkerSink& sink, bool final) {
    const auto fps = features_.framesPerSecond();
    const auto first = features_.firstFrame();
    const auto end = features_.endFrame();
    const auto start = final ? first : std::max(first, end - static_cast<std::int64_t>(kLiveWindowSeconds * fps));
    const auto offset = static_cast<std::size_t>(start - first);
    const auto frames = static_cast<std::size_t>(end - start);
    if (frames == 0) return;

    auto minBpm = setting(settings_.minBpm, 60.0);
    auto maxBpm = setting(settings_.maxBpm, 200.0);
    if (minBpm > maxBpm) std::swap(minBpm, maxBpm);
    const auto choice = static_cast<int>(std::lround(setting(settings_.meterChoice, 0.0)));
    const auto fixedBeatsPerBar = choice > 0 ? std::clamp(choice + 1, 2, 7) : 0;

    const auto envelope = normaliseEnvelope(features_.flux().data() + offset, frames);
    auto tempo = estimateTempo(envelope, fps, minBpm, maxBpm);
    if (tempo.bpm <= 0.0 || tempo.confidence < 0.05f) {
        smoothedBpm_ = 0.0;
        publish({}, {});
        return;
    }
    const auto period = tempo.periodFrames;
    if (!final) {
        // Live readout: smooth small changes, follow real tempo changes.
        if (smoothedBpm_ > 0.0 && std::abs(tempo.bpm - smoothedBpm_) / smoothedBpm_ < 0.05)
            smoothedBpm_ = 0.8 * smoothedBpm_ + 0.2 * tempo.bpm;
        else
            smoothedBpm_ = tempo.bpm;
        tempo.bpm = smoothedBpm_;
    }

    const auto beats = trackBeats(envelope, period);
    const auto meter = estimateMeter(beats, features_.lowFlux().data() + offset, features_.flux().data() + offset,
                                     features_.bands().data() + offset * OnsetFeatures::kBands, frames, fixedBeatsPerBar);
    publish(tempo, meter);

    const bool meterUsable = meter.beatsPerBar > 0 && meter.downbeatConfidence >= 0.15f &&
                             (fixedBeatsPerBar > 0 || meter.meterConfidence >= 0.25f);
    const auto latest = static_cast<double>(end) - kLiveLatencySeconds * fps;
    for (std::size_t j = 0; j < beats.size(); ++j) {
        const auto frame = static_cast<double>(start) + beats[j];
        if (!final && (frame > latest || frame <= lastEmitted_ + 0.6 * period)) continue;

        int position = 0;
        if (meter.beatsPerBar > 0 && (final || meterUsable)) {
            const auto m = meter.beatsPerBar;
            position = ((static_cast<int>(j) - meter.phase) % m + m) % m + 1;
        } else if (lastPosition_ > 0 && lastBeatsPerBar_ > 0) {
            position = lastPosition_ % lastBeatsPerBar_ + 1;  // keep counting through a weak estimate
        }
        const auto strength = std::clamp(envelope[static_cast<std::size_t>(beats[j])] / 4.0f, 0.0f, 1.0f);
        const auto sample = sampleBase_ + static_cast<std::int64_t>(std::llround(frame * features_.hop()));
        sink.emit(makeBeatMarker(sample, static_cast<std::uint8_t>(position), strength));

        lastEmitted_ = frame;
        lastPosition_ = position;
        if (meter.beatsPerBar > 0 && (final || meterUsable)) lastBeatsPerBar_ = meter.beatsPerBar;
    }
}

}  // namespace soniclab::beat_analyzer
