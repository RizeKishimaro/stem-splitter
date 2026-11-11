// Fixed-size commands from the UI thread to the audio thread.
//
// The Python engine changes clip gain, pan, mute and the transport by taking
// the same RLock the PortAudio callback holds. A native engine instead
// queues these POD commands into an SpscRing and the callback applies them
// at the start of each block, so the UI can never stall the audio thread.
#pragma once

#include <cstdint>
#include <type_traits>

namespace soniclab {

enum class CommandType : std::uint8_t {
    Play,
    Pause,
    Stop,
    Seek,            // frame
    SetTrackGain,    // track, value
    SetTrackPan,     // track, value (-1..1)
    SetTrackMuted,   // track, value (0 or 1)
    SetTrackSolo,    // track, value (0 or 1)
    SetOutputGain,   // value
    SetMasterMuted,  // value (0 or 1)
    SetLoop,         // frame = start, frameEnd = end (exclusive)
    ClearLoop,
    SetParameter,    // track = slot, index = parameter, value = normalised 0..1
};

struct EngineCommand {
    CommandType type = CommandType::Pause;
    std::int32_t track = -1;
    std::int32_t index = -1;
    float value = 0.0f;
    std::int64_t frame = 0;
    std::int64_t frameEnd = 0;
};

static_assert(std::is_trivially_copyable_v<EngineCommand>, "commands travel through SpscRing");

}  // namespace soniclab
