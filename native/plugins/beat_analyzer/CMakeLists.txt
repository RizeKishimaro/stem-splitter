# The analyzer's realtime-to-worker pipeline. JUCE-free so it is tested on
# every build; the JUCE plugin wrapper is in juce/.
add_library(soniclab_beat_analyzer_core STATIC
    src/analysis_pipeline.cpp)
add_library(soniclab::beat_analyzer_core ALIAS soniclab_beat_analyzer_core)
target_include_directories(soniclab_beat_analyzer_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(soniclab_beat_analyzer_core
    PUBLIC soniclab::common
    PRIVATE soniclab_warnings)
set_target_properties(soniclab_beat_analyzer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
