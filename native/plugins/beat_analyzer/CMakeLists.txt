# The Beat Analyzer's beat tracker. JUCE-free so it is tested on every build;
# the JUCE plugin is in juce/.
add_library(soniclab_beat_analyzer_core STATIC
    src/beat_tracker.cpp)
add_library(soniclab::beat_analyzer_core ALIAS soniclab_beat_analyzer_core)
target_include_directories(soniclab_beat_analyzer_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(soniclab_beat_analyzer_core
    PUBLIC soniclab::analysis_core
    PRIVATE soniclab_warnings)
set_target_properties(soniclab_beat_analyzer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Development tool: the tracker on raw audio from stdin, for
# scripts/compare_beats.py.
if(SONICLAB_BUILD_TESTS)
    add_executable(soniclab_beat_cli ${PROJECT_SOURCE_DIR}/tools/beat_cli.cpp)
    target_link_libraries(soniclab_beat_cli PRIVATE soniclab::beat_analyzer_core soniclab_warnings)
endif()
