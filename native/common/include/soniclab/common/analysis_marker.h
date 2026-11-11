// Analysis markers passed from an analysis worker to the UI or host.
//
// Positions are in samples from the start of the analysed stream, at the
// stream's own sample rate. This matches SonicLab's Python side, where bar
// export rounds bar edges to samples once (see soniclab/core/bar_map.py) and
// beat positions count from 1 at the downbeat (soniclab/core/beat_analysis.py).
#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace soniclab {

enum class MarkerKind : std::uint8_t {
    Onset = 0,
    Beat = 1,
    Downbeat = 2,
    BarLine = 3,
};

struct AnalysisMarker {
    std::int64_t samplePosition = 0;
    float strength = 0.0f;       // 0..1, detector confidence or salience
    MarkerKind kind = MarkerKind::Onset;
    std::uint8_t beatInBar = 0;  // 1 is the downbeat, 0 when unknown

    double seconds(double sampleRate) const noexcept {
        return sampleRate > 0.0 ? static_cast<double>(samplePosition) / sampleRate : 0.0;
    }
};

static_assert(std::is_trivially_copyable_v<AnalysisMarker>, "markers travel through SpscRing");

std::string_view markerKindName(MarkerKind kind) noexcept;
bool markerKindFromName(std::string_view name, MarkerKind& out) noexcept;

// A beat marker is a downbeat when its position in the bar is 1.
AnalysisMarker makeBeatMarker(std::int64_t samplePosition, std::uint8_t beatInBar, float strength) noexcept;

}  // namespace soniclab
