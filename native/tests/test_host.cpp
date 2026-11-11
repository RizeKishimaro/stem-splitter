#include <atomic>
#include <map>
#include <thread>

#include "soniclab/host/chain_state.h"
#include "soniclab/host/plugin_chain.h"
#include "soniclab/host/plugin_scanner.h"
#include "test_framework.h"

using namespace soniclab::host;

namespace {

// Multiplies every sample by a gain parameter; stands in for a real plugin.
class GainPlugin final : public IPluginInstance {
public:
    explicit GainPlugin(std::string name, float gain = 1.0f, int latency = 0) : gain_(gain), latency_(latency) {
        description_.format = "Test";
        description_.name = std::move(name);
        description_.identifier = "Test-" + description_.name;
    }
    ~GainPlugin() override { ++destroyed; }

    const PluginDescription& description() const override { return description_; }
    void prepare(double sampleRate, int maxBlockSize) override {
        preparedRate = sampleRate;
        preparedBlock = maxBlockSize;
    }
    void release() override { preparedRate = 0.0; }
    int latencySamples() const override { return latency_; }
    void process(float* const* channels, int numChannels, int numFrames) noexcept override {
        const float gain = gain_.load();
        for (int ch = 0; ch < numChannels; ++ch)
            for (int i = 0; i < numFrames; ++i) channels[ch][i] *= gain;
    }
    int parameterCount() const override { return 1; }
    ParameterInfo parameterInfo(int) const override { return {"gain", "Gain", 1.0f, true}; }
    float parameter(int) const override { return gain_.load(); }
    void setParameter(int, float normalised) noexcept override { gain_.store(normalised); }
    std::vector<std::uint8_t> saveState() const override {
        return {static_cast<std::uint8_t>(gain_.load() * 255.0f + 0.5f)};
    }
    bool restoreState(const std::vector<std::uint8_t>& state) override {
        if (state.size() != 1) return false;
        gain_.store(static_cast<float>(state[0]) / 255.0f);
        return true;
    }

    double preparedRate = 0.0;
    int preparedBlock = 0;
    static inline int destroyed = 0;

private:
    PluginDescription description_;
    std::atomic<float> gain_;
    int latency_;
};

class TestLoader final : public IPluginLoader {
public:
    std::unique_ptr<IPluginInstance> load(const PluginDescription& description, double sampleRate,
                                          int maxBlockSize, std::string& error) override {
        if (description.name == "Missing") {
            error = "not installed";
            return nullptr;
        }
        auto plugin = std::make_unique<GainPlugin>(description.name);
        plugin->prepare(sampleRate, maxBlockSize);
        return plugin;
    }
};

float runBlock(PluginChain& chain, float input) {
    float left[4] = {input, input, input, input};
    float right[4] = {input, input, input, input};
    float* channels[2] = {left, right};
    chain.process(channels, 2, 4);
    return left[3];
}

}  // namespace

TEST("host.chain.empty_chain_passes_through") {
    PluginChain chain;
    CHECK_EQ(runBlock(chain, 0.5f), 0.5f);
}

TEST("host.chain.insert_prepares_and_processes_in_order") {
    PluginChain chain;
    chain.prepare(48000.0, 256);
    auto half = std::make_shared<GainPlugin>("half", 0.5f, 10);
    auto quarter = std::make_shared<GainPlugin>("quarter", 0.25f, 5);
    chain.insert(half);
    chain.insert(quarter);
    CHECK_EQ(half->preparedRate, 48000.0);
    CHECK_EQ(quarter->preparedBlock, 256);
    CHECK_EQ(runBlock(chain, 1.0f), 0.125f);
    CHECK_EQ(chain.latencySamples(), 15);
}

TEST("host.chain.bypass_move_and_remove") {
    PluginChain chain;
    const auto a = chain.insert(std::make_shared<GainPlugin>("a", 0.5f));
    const auto b = chain.insert(std::make_shared<GainPlugin>("b", 0.5f));
    CHECK(chain.setBypassed(a, true));
    CHECK_EQ(runBlock(chain, 1.0f), 0.5f);
    CHECK(chain.move(b, 0));
    CHECK_EQ(chain.slots().front().id, b);
    CHECK(chain.remove(b));
    CHECK_EQ(runBlock(chain, 1.0f), 1.0f);  // only the bypassed slot is left
    CHECK(!chain.remove(b));
}

TEST("host.chain.removed_plugin_is_freed_off_the_audio_thread") {
    const int before = GainPlugin::destroyed;
    {
        PluginChain chain;
        const auto id = chain.insert(std::make_shared<GainPlugin>("x"));
        runBlock(chain, 1.0f);  // audio thread takes the snapshot holding x
        chain.remove(id);
        runBlock(chain, 1.0f);  // audio thread retires it
        CHECK_EQ(GainPlugin::destroyed, before);
        chain.collectGarbage();  // message thread frees it
        CHECK_EQ(GainPlugin::destroyed, before + 1);
    }
}

TEST("host.chain.queued_parameter_follows_slot_after_reorder") {
    PluginChain chain;
    const auto a = chain.insert(std::make_shared<GainPlugin>("a", 1.0f));
    chain.insert(std::make_shared<GainPlugin>("b", 1.0f));
    runBlock(chain, 1.0f);
    chain.move(a, 1);
    CHECK(chain.queueParameter(a, 0, 0.25f));
    CHECK_EQ(runBlock(chain, 1.0f), 0.25f);
    CHECK_EQ(chain.slots().back().plugin->parameter(0), 0.25f);
}

TEST("host.chain.edits_while_audio_thread_runs") {
    PluginChain chain;
    chain.prepare(44100.0, 4);
    std::atomic<bool> stop{false};
    std::atomic<bool> bad{false};
    std::thread audio([&] {
        while (!stop.load()) {
            const float out = runBlock(chain, 1.0f);
            // Every plugin is gain 1, so any torn snapshot would show up as a crash
            // under sanitizers rather than a value; the value must stay 1.
            if (out != 1.0f) bad.store(true);
        }
    });
    for (int i = 0; i < 2000; ++i) {
        const auto id = chain.insert(std::make_shared<GainPlugin>("p" + std::to_string(i)));
        if (i % 3 == 0) chain.remove(id);
        chain.collectGarbage();
    }
    stop.store(true);
    audio.join();
    CHECK(!bad.load());
}

TEST("host.state.base64_round_trip") {
    for (std::size_t size = 0; size < 10; ++size) {
        std::vector<std::uint8_t> data(size);
        for (std::size_t i = 0; i < size; ++i) data[i] = static_cast<std::uint8_t>(i * 37 + 250);
        std::vector<std::uint8_t> back;
        CHECK(base64Decode(base64Encode(data), back));
        CHECK(back == data);
    }
    CHECK_EQ(base64Encode({'M', 'a', 'n'}), std::string("TWFu"));
    std::vector<std::uint8_t> unused;
    CHECK(!base64Decode("abc", unused));
    CHECK(!base64Decode("a=bc", unused));
}

TEST("host.state.capture_encode_decode_restore") {
    PluginChain chain;
    chain.insert(std::make_shared<GainPlugin>("first", 0.2f));
    const auto second = chain.insert(std::make_shared<GainPlugin>("line\nbreak\\name", 1.0f));
    chain.setBypassed(second, true);

    const auto text = encodeChainState(captureChainState(chain));
    ChainState decoded;
    CHECK(decodeChainState(text, decoded));
    CHECK_EQ(decoded.slots.size(), std::size_t{2});
    CHECK_EQ(decoded.slots[1].plugin.name, std::string("line\nbreak\\name"));
    CHECK(decoded.slots[1].bypassed);

    decoded.slots.push_back({});
    decoded.slots.back().plugin.name = "Missing";

    PluginChain restored;
    TestLoader loader;
    std::vector<std::string> errors;
    CHECK_EQ(restoreChainState(decoded, loader, restored, 44100.0, 512, errors), 2);
    CHECK_EQ(errors.size(), std::size_t{1});
    CHECK_NEAR(restored.slots()[0].plugin->parameter(0), 0.2, 1.0 / 255.0);
    CHECK(restored.slots()[1].bypassed);
}

TEST("host.state.rejects_bad_input") {
    ChainState out;
    CHECK(!decodeChainState("", out));
    CHECK(!decodeChainState("soniclab-chain 2\n", out));
    CHECK(!decodeChainState("soniclab-chain 1\nslot\nname=x\n", out));  // no end
    CHECK(!decodeChainState("soniclab-chain 1\nname=x\n", out));        // key outside slot
    CHECK(!decodeChainState("soniclab-chain 1\nslot\nstate=***\nend\n", out));
    CHECK(decodeChainState("soniclab-chain 1\nslot\nfuture_key=1\nend\n", out));
}
