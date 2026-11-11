#include "soniclab/host/plugin_chain.h"

#include <algorithm>

namespace soniclab::host {

PluginChain::PluginChain() : retired_(64), parameters_(1024) {}

PluginChain::~PluginChain() {
    delete pending_.exchange(nullptr);
    delete current_;
    collectGarbage();
    release();
}

void PluginChain::prepare(double sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;
    for (auto& slot : slots_) slot.plugin->prepare(sampleRate, maxBlockSize);
}

void PluginChain::release() {
    for (auto& slot : slots_) slot.plugin->release();
    sampleRate_ = 0.0;
    maxBlockSize_ = 0;
}

SlotId PluginChain::insert(std::shared_ptr<IPluginInstance> plugin, int position) {
    if (!plugin) return 0;
    if (sampleRate_ > 0.0) plugin->prepare(sampleRate_, maxBlockSize_);
    SlotInfo slot{nextId_++, false, std::move(plugin)};
    const auto size = static_cast<int>(slots_.size());
    const auto at = (position < 0 || position > size) ? size : position;
    slots_.insert(slots_.begin() + at, std::move(slot));
    publish();
    return slots_[static_cast<std::size_t>(at)].id;
}

bool PluginChain::remove(SlotId id) {
    const auto index = indexOf(id);
    if (index < 0) return false;
    slots_.erase(slots_.begin() + index);
    publish();
    // The plugin is released when the audio thread retires the old snapshot
    // and collectGarbage() drops the last reference.
    return true;
}

bool PluginChain::move(SlotId id, int newPosition) {
    const auto index = indexOf(id);
    if (index < 0) return false;
    auto slot = std::move(slots_[static_cast<std::size_t>(index)]);
    slots_.erase(slots_.begin() + index);
    const auto size = static_cast<int>(slots_.size());
    const auto at = std::clamp(newPosition, 0, size);
    slots_.insert(slots_.begin() + at, std::move(slot));
    publish();
    return true;
}

bool PluginChain::setBypassed(SlotId id, bool bypassed) {
    const auto index = indexOf(id);
    if (index < 0) return false;
    slots_[static_cast<std::size_t>(index)].bypassed = bypassed;
    publish();
    return true;
}

int PluginChain::latencySamples() const {
    int total = 0;
    for (const auto& slot : slots_)
        if (!slot.bypassed) total += slot.plugin->latencySamples();
    return total;
}

bool PluginChain::queueParameter(SlotId id, int index, float normalised) {
    return parameters_.push({id, index, std::clamp(normalised, 0.0f, 1.0f)});
}

void PluginChain::collectGarbage() {
    Snapshot* old = nullptr;
    while (retired_.pop(old)) delete old;
}

void PluginChain::publish() {
    auto* snapshot = new Snapshot;
    snapshot->entries.reserve(slots_.size());
    snapshot->owners.reserve(slots_.size());
    for (const auto& slot : slots_) {
        snapshot->entries.push_back({slot.id, slot.bypassed, slot.plugin.get()});
        snapshot->owners.push_back(slot.plugin);
    }
    // A snapshot the audio thread never picked up is still ours to free.
    delete pending_.exchange(snapshot, std::memory_order_acq_rel);
    collectGarbage();
}

int PluginChain::indexOf(SlotId id) const {
    for (std::size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].id == id) return static_cast<int>(i);
    return -1;
}

void PluginChain::process(float* const* channels, int numChannels, int numFrames) noexcept {
    // Only swap when the old snapshot has somewhere to go; otherwise keep the
    // current one for another block rather than free memory here.
    if (retired_.freeSpace() > 0) {
        if (auto* next = pending_.exchange(nullptr, std::memory_order_acq_rel)) {
            if (current_ != nullptr) retired_.push(current_);
            current_ = next;
        }
    }
    if (current_ == nullptr) return;

    ParameterChange change{};
    while (parameters_.pop(change)) {
        for (const auto& entry : current_->entries) {
            if (entry.id == change.id) {
                entry.plugin->setParameter(change.index, change.value);
                break;
            }
        }
    }

    for (const auto& entry : current_->entries)
        if (!entry.bypassed) entry.plugin->process(channels, numChannels, numFrames);
}

}  // namespace soniclab::host
