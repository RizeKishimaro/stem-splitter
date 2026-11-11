#include <thread>
#include <vector>

#include "soniclab/common/analysis_marker.h"
#include "soniclab/common/atomic_peak.h"
#include "soniclab/common/engine_command.h"
#include "soniclab/common/spsc_ring.h"
#include "test_framework.h"

using namespace soniclab;

TEST("common.ring.capacity_rounds_up") {
    SpscRing<int> ring(5);
    CHECK_EQ(ring.capacity(), std::size_t{8});
    CHECK_EQ(ring.available(), std::size_t{0});
    CHECK_EQ(ring.freeSpace(), std::size_t{8});
}

TEST("common.ring.push_pop_and_full") {
    SpscRing<int> ring(4);
    for (int i = 0; i < 4; ++i) CHECK(ring.push(i));
    CHECK(!ring.push(99));
    int value = -1;
    for (int i = 0; i < 4; ++i) {
        CHECK(ring.pop(value));
        CHECK_EQ(value, i);
    }
    CHECK(!ring.pop(value));
}

TEST("common.ring.bulk_wraps_around") {
    SpscRing<float> ring(8);
    std::vector<float> in(6), out(6);
    for (int round = 0; round < 10; ++round) {
        for (int i = 0; i < 6; ++i) in[i] = static_cast<float>(round * 10 + i);
        CHECK_EQ(ring.pushBulk(in.data(), in.size()), std::size_t{6});
        CHECK_EQ(ring.popBulk(out.data(), out.size()), std::size_t{6});
        CHECK(in == out);
    }
}

TEST("common.ring.bulk_partial_when_nearly_full") {
    SpscRing<int> ring(4);
    int in[6] = {1, 2, 3, 4, 5, 6};
    CHECK_EQ(ring.pushBulk(in, 6), std::size_t{4});
    int out[6] = {};
    CHECK_EQ(ring.popBulk(out, 6), std::size_t{4});
    CHECK_EQ(out[3], 4);
}

TEST("common.ring.discard_all") {
    SpscRing<int> ring(4);
    ring.push(1);
    ring.push(2);
    ring.discardAll();
    CHECK_EQ(ring.available(), std::size_t{0});
    CHECK(ring.push(3));
}

TEST("common.ring.threaded_sequence_is_preserved") {
    // One producer, one consumer, a small ring so both sides wrap constantly.
    constexpr std::uint64_t kCount = 2'000'000;
    SpscRing<std::uint64_t> ring(64);
    std::thread producer([&] {
        std::uint64_t next = 0;
        while (next < kCount)
            if (ring.push(next)) ++next;
    });
    std::uint64_t expected = 0;
    bool ordered = true;
    std::uint64_t value = 0;
    while (expected < kCount) {
        if (ring.pop(value)) {
            ordered = ordered && value == expected;
            ++expected;
        }
    }
    producer.join();
    CHECK(ordered);
    CHECK_EQ(ring.available(), std::size_t{0});
}

TEST("common.ring.carries_engine_commands") {
    SpscRing<EngineCommand> ring(16);
    EngineCommand command;
    command.type = CommandType::SetTrackGain;
    command.track = 3;
    command.value = 0.5f;
    CHECK(ring.push(command));
    EngineCommand out;
    CHECK(ring.pop(out));
    CHECK(out.type == CommandType::SetTrackGain);
    CHECK_EQ(out.track, 3);
    CHECK_EQ(out.value, 0.5f);
}

TEST("common.marker.kind_names_round_trip") {
    for (auto kind : {MarkerKind::Onset, MarkerKind::Beat, MarkerKind::Downbeat, MarkerKind::BarLine}) {
        MarkerKind parsed = MarkerKind::Onset;
        CHECK(markerKindFromName(markerKindName(kind), parsed));
        CHECK(parsed == kind);
    }
    MarkerKind unused;
    CHECK(!markerKindFromName("tempo", unused));
}

TEST("common.marker.beat_one_is_downbeat") {
    CHECK(makeBeatMarker(0, 1, 1.0f).kind == MarkerKind::Downbeat);
    CHECK(makeBeatMarker(22050, 2, 1.0f).kind == MarkerKind::Beat);
    CHECK_NEAR(makeBeatMarker(22050, 2, 1.0f).seconds(44100.0), 0.5, 1e-12);
    CHECK_EQ(AnalysisMarker{}.seconds(0.0), 0.0);
}

TEST("common.peak.raise_and_take") {
    AtomicPeak peak;
    peak.raise(0.25f);
    peak.raise(-0.75f);
    peak.raise(0.5f);
    CHECK_EQ(peak.take(), 0.75f);
    CHECK_EQ(peak.take(), 0.0f);
}

TEST("common.peak.concurrent_raise_keeps_maximum") {
    AtomicPeak peak;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&peak, t] {
            for (int i = 0; i < 10000; ++i) peak.raise(static_cast<float>(i % 100) / 100.0f + static_cast<float>(t) * 0.001f);
        });
    for (auto& thread : threads) thread.join();
    CHECK_NEAR(peak.peek(), 0.99 + 0.003, 1e-6);
}
