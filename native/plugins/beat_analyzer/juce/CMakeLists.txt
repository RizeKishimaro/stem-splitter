# SonicLab Beat Analyzer as a JUCE plugin.
#
# macOS: AU (AUv2 .component) and Standalone; VST3 with SONICLAB_BUILD_VST3.
# Linux: Standalone; VST3 with SONICLAB_BUILD_VST3. AU is never built on Linux.
# Codes: manufacturer "Snlb", subtype "Btan", type aufx (effect), so
#   auval -v aufx Btan Snlb
# AU artefact: SonicLabBeatAnalyzer_artefacts/<Config>/AU/SonicLab Beat Analyzer.component
option(SONICLAB_COPY_PLUGIN_AFTER_BUILD
    "Copy built plugins to the user's plug-in folders (macOS: ~/Library/Audio/Plug-Ins/Components)" OFF)
set(_formats Standalone)
if(APPLE)
    list(APPEND _formats AU)
endif()
if(SONICLAB_BUILD_VST3)
    list(APPEND _formats VST3)
endif()

juce_add_plugin(SonicLabBeatAnalyzer
    PRODUCT_NAME "SonicLab Beat Analyzer"
    VERSION 0.1.1
    COMPANY_NAME "SonicLab"
    PLUGIN_NAME "SonicLab Beat Analyzer"
    DESCRIPTION "Pass-through input meter and beat analysis scaffold"
    BUNDLE_ID "org.soniclab.beatanalyzer"
    PLUGIN_MANUFACTURER_CODE Snlb
    PLUGIN_CODE Btan
    FORMATS ${_formats}
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    AU_MAIN_TYPE kAudioUnitType_Effect
    VST3_CATEGORIES Fx Analyzer
    COPY_PLUGIN_AFTER_BUILD ${SONICLAB_COPY_PLUGIN_AFTER_BUILD})

target_sources(SonicLabBeatAnalyzer PRIVATE
    ../src/PluginProcessor.cpp
    ../src/PluginEditor.cpp
    ../src/FileAnalysis.cpp)

target_compile_definitions(SonicLabBeatAnalyzer PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(SonicLabBeatAnalyzer
    PRIVATE
        soniclab::beat_analyzer_core
        juce::juce_audio_utils
        juce::juce_gui_basics
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

# Headless tests of the real processor (not the AU wrapper), built wherever
# JUCE is: they link the plugin's shared code like a host would.
if(SONICLAB_BUILD_TESTS)
    # Links the plugin's compiled shared code (the same objects the AU and
    # Standalone wrappers link) and compiles with the plugin's own JUCE
    # configuration, so the tests see exactly what a host loads.
    add_executable(SonicLabBeatAnalyzerPluginTests test_plugin.cpp)
    target_include_directories(SonicLabBeatAnalyzerPluginTests PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ../src
        $<TARGET_PROPERTY:juce::juce_core,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(SonicLabBeatAnalyzerPluginTests PRIVATE
        $<TARGET_PROPERTY:SonicLabBeatAnalyzer,COMPILE_DEFINITIONS>)
    target_link_libraries(SonicLabBeatAnalyzerPluginTests PRIVATE
        SonicLabBeatAnalyzer
        soniclab::beat_analyzer_core
        juce::juce_recommended_config_flags)
    add_test(NAME beat_analyzer_plugin COMMAND SonicLabBeatAnalyzerPluginTests)
    set_tests_properties(beat_analyzer_plugin PROPERTIES TIMEOUT 120)
endif()
