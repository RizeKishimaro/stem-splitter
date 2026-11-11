// An ordered insert chain whose layout changes without locking the audio thread.
//
// The message thread edits a list of slots, then publishes an immutable
// snapshot through an atomic pointer. The audio thread picks the snapshot up
// at the start of the next block and hands the one it replaced back through
// an SPSC ring, so it is freed on the message thread by collectGarbage().
// Loading, preparing and destroying plugins therefore never happen on the
// audio thread, and the audio thread never waits for the message thread.
//
// Parameter changes travel the same way: queued on the message thread,
// applied by the audio thread before it processes the block. Slots are
// addressed by a stable id so a queued change still reaches the right plugin
// after the chain is reordered.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "soniclab/common/spsc_ring.h"
#include "soniclab/host/plugin_instance.h"

namespace soniclab::host {

using SlotId = std::uint32_t;

struct SlotInfo {
    SlotId id = 0;
    bool bypassed = false;
    std::shared_ptr<IPluginInstance> plugin;
};

class PluginChain {
public:
    PluginChain();
    // The audio thread must have stopped calling process().
    ~PluginChain();

    PluginChain(const PluginChain&) = delete;
    PluginChain& operator=(const PluginChain&) = delete;

    // Message thread, with the audio device stopped. Prepares every plugin.
    void prepare(double sampleRate, int maxBlockSize);
    void release();

    // Message thread. The plugin is prepared here if the chain is prepared.
    SlotId insert(std::shared_ptr<IPluginInstance> plugin, int position = -1);
    bool remove(SlotId id);
    bool move(SlotId id, int newPosition);
    bool setBypassed(SlotId id, bool bypassed);
    const std::vector<SlotInfo>& slots() const { return slots_; }
    int latencySamples() const;

    // Message thread. False when the queue is full; the caller may retry.
    bool queueParameter(SlotId id, int index, float normalised);

    // Message thread. Frees snapshots the audio thread has finished with.
    void collectGarbage();

    // Audio thread.
    void process(float* const* channels, int numChannels, int numFrames) noexcept;

private:
    struct Snapshot {
        struct Entry {
            SlotId id;
            bool bypassed;
            IPluginInstance* plugin;
        };
        std::vector<Entry> entries;
        std::vector<std::shared_ptr<IPluginInstance>> owners;  // keeps plugins alive
    };

    struct ParameterChange {
        SlotId id;
        std::int32_t index;
        float value;
    };

    void publish();
    int indexOf(SlotId id) const;

    std::vector<SlotInfo> slots_;
    SlotId nextId_ = 1;
    double sampleRate_ = 0.0;
    int maxBlockSize_ = 0;

    std::atomic<Snapshot*> pending_{nullptr};
    Snapshot* current_ = nullptr;  // owned by the audio thread while running
    SpscRing<Snapshot*> retired_;
    SpscRing<ParameterChange> parameters_;
};

}  // namespace soniclab::host
