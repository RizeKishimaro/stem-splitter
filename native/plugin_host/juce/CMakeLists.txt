# JUCE plugin-hosting adapter. AudioUnit hosting is enabled on macOS only;
# VST3 hosting follows SONICLAB_BUILD_VST3 (see native/LICENSING.md).
add_library(soniclab_host_juce STATIC
    ../src/juce_plugin_host.cpp)
add_library(soniclab::host_juce ALIAS soniclab_host_juce)
target_link_libraries(soniclab_host_juce
    PUBLIC
        soniclab::host_core
        juce::juce_audio_processors
        juce::juce_audio_utils
    PRIVATE
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

if(APPLE)
    set(_au 1)
else()
    set(_au 0)
endif()
if(SONICLAB_BUILD_VST3)
    set(_vst3 1)
else()
    set(_vst3 0)
endif()

target_compile_definitions(soniclab_host_juce PUBLIC
    JUCE_PLUGINHOST_AU=${_au}
    JUCE_PLUGINHOST_VST3=${_vst3}
    JUCE_PLUGINHOST_LADSPA=0
    JUCE_PLUGINHOST_LV2=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0)
