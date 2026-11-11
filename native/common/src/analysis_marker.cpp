#include "soniclab/common/analysis_marker.h"

namespace soniclab {

std::string_view markerKindName(MarkerKind kind) noexcept {
    switch (kind) {
        case MarkerKind::Onset: return "onset";
        case MarkerKind::Beat: return "beat";
        case MarkerKind::Downbeat: return "downbeat";
        case MarkerKind::BarLine: return "bar_line";
    }
    return "unknown";
}

bool markerKindFromName(std::string_view name, MarkerKind& out) noexcept {
    for (auto kind : {MarkerKind::Onset, MarkerKind::Beat, MarkerKind::Downbeat, MarkerKind::BarLine}) {
        if (markerKindName(kind) == name) {
            out = kind;
            return true;
        }
    }
    return false;
}

AnalysisMarker makeBeatMarker(std::int64_t samplePosition, std::uint8_t beatInBar, float strength) noexcept {
    AnalysisMarker marker;
    marker.samplePosition = samplePosition;
    marker.beatInBar = beatInBar;
    marker.strength = strength;
    marker.kind = beatInBar == 1 ? MarkerKind::Downbeat : MarkerKind::Beat;
    return marker;
}

}  // namespace soniclab
