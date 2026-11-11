// Lock-free single-producer / single-consumer ring buffer.
//
// One thread pushes, one other thread pops. Neither side ever locks,
// allocates or blocks, so either side may be a realtime audio callback.
// Storage is allocated once in the constructor, on a non-realtime thread.
//
// Capacity is rounded up to a power of two. Indices grow without wrapping
// (64-bit) and are masked on access, so "full" and "empty" never alias.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>

namespace soniclab {

// Fixed rather than std::hardware_destructive_interference_size, which GCC
// warns is ABI-unstable and Apple Silicon reports inconsistently.
inline constexpr std::size_t kCacheLine = 64;

template <typename T>
class SpscRing {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SpscRing holds trivially copyable values so copies never allocate");

public:
    explicit SpscRing(std::size_t minimumCapacity)
        : capacity_(roundUpToPowerOfTwo(std::max<std::size_t>(minimumCapacity, 2))),
          mask_(capacity_ - 1),
          storage_(std::make_unique<T[]>(capacity_)) {}

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    std::size_t capacity() const noexcept { return capacity_; }

    // Producer side.
    bool push(const T& value) noexcept { return pushBulk(&value, 1) == 1; }

    // Copies as many of ``count`` values as fit. Returns how many were written.
    std::size_t pushBulk(const T* values, std::size_t count) noexcept {
        const auto write = write_.load(std::memory_order_relaxed);
        const auto read = read_.load(std::memory_order_acquire);
        const auto space = capacity_ - static_cast<std::size_t>(write - read);
        const auto n = std::min(count, space);
        copyIn(write, values, n);
        write_.store(write + n, std::memory_order_release);
        return n;
    }

    std::size_t freeSpace() const noexcept {
        return capacity_ - static_cast<std::size_t>(write_.load(std::memory_order_relaxed) -
                                                    read_.load(std::memory_order_acquire));
    }

    // Consumer side.
    bool pop(T& out) noexcept { return popBulk(&out, 1) == 1; }

    std::size_t popBulk(T* out, std::size_t count) noexcept {
        const auto read = read_.load(std::memory_order_relaxed);
        const auto write = write_.load(std::memory_order_acquire);
        const auto n = std::min(count, static_cast<std::size_t>(write - read));
        copyOut(read, out, n);
        read_.store(read + n, std::memory_order_release);
        return n;
    }

    std::size_t available() const noexcept {
        return static_cast<std::size_t>(write_.load(std::memory_order_acquire) -
                                        read_.load(std::memory_order_relaxed));
    }

    // Consumer side: drop everything currently queued.
    void discardAll() noexcept { read_.store(write_.load(std::memory_order_acquire), std::memory_order_release); }

private:
    static std::size_t roundUpToPowerOfTwo(std::size_t value) noexcept {
        std::size_t result = 1;
        while (result < value) result <<= 1;
        return result;
    }

    void copyIn(std::uint64_t index, const T* values, std::size_t n) noexcept {
        const auto start = static_cast<std::size_t>(index) & mask_;
        const auto first = std::min(n, capacity_ - start);
        std::copy_n(values, first, storage_.get() + start);
        if (n > first) std::copy_n(values + first, n - first, storage_.get());
    }

    void copyOut(std::uint64_t index, T* out, std::size_t n) const noexcept {
        const auto start = static_cast<std::size_t>(index) & mask_;
        const auto first = std::min(n, capacity_ - start);
        std::copy_n(storage_.get() + start, first, out);
        if (n > first) std::copy_n(storage_.get(), n - first, out + first);
    }

    const std::size_t capacity_;
    const std::size_t mask_;
    std::unique_ptr<T[]> storage_;
    alignas(kCacheLine) std::atomic<std::uint64_t> write_{0};
    alignas(kCacheLine) std::atomic<std::uint64_t> read_{0};
};

}  // namespace soniclab
