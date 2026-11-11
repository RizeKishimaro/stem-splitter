# SonicLab Beat Analyzer as a JUCE plugin.
#
# macOS: AU (AUv2 .component) and Standalone; VST3 with SONICLAB_BUILD_VST3.
# Linux: Standalone; VST3 with SONICLAB_BUILD_VST3. AU is never built on Linux.
# Codes: manufacturer "Snlb", subtype "Btan", type aufx (effect), so
#   auval -v aufx Btan Snlb
# AU artefact: SonicLabBeatAnalyzer_artefacts/<Config>/AU/SonicLab Beat Analyzer.component
soniclab_add_analysis_plugin(SonicLabBeatAnalyzer
    NAME "SonicLab Beat Analyzer"
    CODE Btan
    BUNDLE_SUFFIX beatanalyzer
    VERSION 0.2.0
    DESCRIPTION "Tempo, beat, downbeat and bar-length analysis of live audio and dropped files"
    SOURCES
        ../src/PluginProcessor.cpp
        ../src/PluginEditor.cpp
    LINK soniclab::beat_analyzer_core)

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
        ${SONICLAB_PLUGIN_SHARED_DIR}
        $<TARGET_PROPERTY:juce::juce_core,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(SonicLabBeatAnalyzerPluginTests PRIVATE
        $<TARGET_PROPERTY:SonicLabBeatAnalyzer,COMPILE_DEFINITIONS>)
    target_link_libraries(SonicLabBeatAnalyzerPluginTests PRIVATE
        SonicLabBeatAnalyzer
        soniclab::analysis_core
        soniclab::beat_analyzer_core
        juce::juce_recommended_config_flags)
    add_test(NAME beat_analyzer_plugin COMMAND SonicLabBeatAnalyzerPluginTests)
    set_tests_properties(beat_analyzer_plugin PROPERTIES TIMEOUT 120)
endif()
