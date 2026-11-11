// Peak meter value written by the audio thread and taken by the UI.
//
// Replaces the pattern in the Python engine where take_peaks() and the
// callback share an RLock: the audio thread raises the peak with a CAS loop,
// the UI swaps it back to zero. Neither side blocks the other.
#pragma once

#include <atomic>
#include <cmath>

namespace soniclab {

class AtomicPeak {
public:
    // Audio thread.
    void raise(float value) noexcept {
        value = std::fabs(value);
        auto current = peak_.load(std::memory_order_relaxed);
        while (value > current &&
               !peak_.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
        }
    }

    // UI thread: read and reset in one step.
    float take() noexcept { return peak_.exchange(0.0f, std::memory_order_relaxed); }

    float peek() const noexcept { return peak_.load(std::memory_order_relaxed); }

private:
    std::atomic<float> peak_{0.0f};
    static_assert(std::atomic<float>::is_always_lock_free, "peak meter must be lock-free");
};

}  // namespace soniclab
