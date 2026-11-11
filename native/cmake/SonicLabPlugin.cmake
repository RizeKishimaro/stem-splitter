# soniclab_add_analysis_plugin(): one call per SonicLab analysis plugin, so
# every plugin gets the same identity, formats, JUCE settings and shared code.
#
#   soniclab_add_analysis_plugin(SonicLabBeatAnalyzer
#       NAME "SonicLab Beat Analyzer"     # product, plugin and bundle name
#       CODE Btan                         # 4-char AU subtype, unique per plugin
#       BUNDLE_SUFFIX beatanalyzer        # -> org.soniclab.beatanalyzer
#       VERSION 0.1.1
#       DESCRIPTION "..."
#       SOURCES src/PluginProcessor.cpp src/PluginEditor.cpp
#       LINK soniclab::beat_analyzer_core)
#
# All plugins share manufacturer code Snlb, company "SonicLab", AU type aufx
# (effect, so Logic offers them on audio tracks), mono and stereo
# pass-through, no MIDI. Formats: AU on macOS, Standalone everywhere, VST3
# with SONICLAB_BUILD_VST3.
set(SONICLAB_PLUGIN_SHARED_DIR "${CMAKE_CURRENT_LIST_DIR}/../plugins/shared/juce")

option(SONICLAB_COPY_PLUGIN_AFTER_BUILD
    "Copy built plugins to the user's plug-in folders (macOS: ~/Library/Audio/Plug-Ins/Components)" OFF)

function(soniclab_add_analysis_plugin target)
    cmake_parse_arguments(PLUGIN "" "NAME;CODE;BUNDLE_SUFFIX;VERSION;DESCRIPTION" "SOURCES;LINK" ${ARGN})
    foreach(required NAME CODE BUNDLE_SUFFIX VERSION)
        if(NOT PLUGIN_${required})
            message(FATAL_ERROR "soniclab_add_analysis_plugin(${target}): ${required} is required")
        endif()
    endforeach()

    set(formats Standalone)
    if(APPLE)
        list(APPEND formats AU)
    endif()
    if(SONICLAB_BUILD_VST3)
        list(APPEND formats VST3)
    endif()

    juce_add_plugin(${target}
        PRODUCT_NAME "${PLUGIN_NAME}"
        VERSION ${PLUGIN_VERSION}
        COMPANY_NAME "SonicLab"
        PLUGIN_NAME "${PLUGIN_NAME}"
        DESCRIPTION "${PLUGIN_DESCRIPTION}"
        BUNDLE_ID "org.soniclab.${PLUGIN_BUNDLE_SUFFIX}"
        PLUGIN_MANUFACTURER_CODE Snlb
        PLUGIN_CODE ${PLUGIN_CODE}
        FORMATS ${formats}
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        AU_MAIN_TYPE kAudioUnitType_Effect
        VST3_CATEGORIES Fx Analyzer
        COPY_PLUGIN_AFTER_BUILD ${SONICLAB_COPY_PLUGIN_AFTER_BUILD})

    # Shared sources are compiled into each plugin rather than a library, so
    # they see that plugin's JUCE configuration (JucePlugin_* macros).
    target_sources(${target} PRIVATE
        ${PLUGIN_SOURCES}
        ${SONICLAB_PLUGIN_SHARED_DIR}/AnalysisEditorBase.cpp
        ${SONICLAB_PLUGIN_SHARED_DIR}/AnalysisProcessorBase.cpp
        ${SONICLAB_PLUGIN_SHARED_DIR}/AnalysisViews.cpp
        ${SONICLAB_PLUGIN_SHARED_DIR}/FileAnalysis.cpp
        ${SONICLAB_PLUGIN_SHARED_DIR}/Widgets.cpp)
    target_include_directories(${target} PRIVATE ${SONICLAB_PLUGIN_SHARED_DIR})

    target_compile_definitions(${target} PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0)

    target_link_libraries(${target}
        PRIVATE
            soniclab::analysis_core
            ${PLUGIN_LINK}
            juce::juce_audio_utils
            juce::juce_gui_basics
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_lto_flags
            juce::juce_recommended_warning_flags)
endfunction()
