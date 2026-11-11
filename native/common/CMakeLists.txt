# Realtime-safe code shared by the engine, host and plugins. No JUCE.
add_library(soniclab_common STATIC
    src/analysis_marker.cpp)
add_library(soniclab::common ALIAS soniclab_common)
target_include_directories(soniclab_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(soniclab_common PUBLIC cxx_std_17)
target_link_libraries(soniclab_common
    PUBLIC Threads::Threads
    PRIVATE soniclab_warnings)
set_target_properties(soniclab_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
