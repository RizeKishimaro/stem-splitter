# Code shared by every SonicLab analysis plugin.
#
#   soniclab::analysis_core   JUCE-free: realtime-to-worker AnalysisPipeline and
#                             the OnlineAnalyzer interface. Tested on every build.
#   juce/                     JUCE sources (theme, widgets, processor and editor
#                             bases, dropped-file analysis). Each plugin compiles
#                             them itself through soniclab_add_analysis_plugin()
#                             in cmake/SonicLabPlugin.cmake, so they always see
#                             that plugin's own JUCE configuration.
add_library(soniclab_analysis_core STATIC
    src/analysis_pipeline.cpp)
add_library(soniclab::analysis_core ALIAS soniclab_analysis_core)
target_include_directories(soniclab_analysis_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(soniclab_analysis_core
    PUBLIC soniclab::common Threads::Threads
    PRIVATE soniclab_warnings)
set_target_properties(soniclab_analysis_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
