# Phase 3 prototype: the C ABI shared library Python loads with ctypes.
# Only the symbols marked SL_API are exported.
add_library(soniclab_engine SHARED
    src/engine_capi.cpp)
add_library(soniclab::engine ALIAS soniclab_engine)
target_include_directories(soniclab_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(soniclab_engine PRIVATE SONICLAB_ENGINE_BUILD)
target_link_libraries(soniclab_engine
    PRIVATE soniclab::common soniclab_warnings)
set_target_properties(soniclab_engine PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 0)
