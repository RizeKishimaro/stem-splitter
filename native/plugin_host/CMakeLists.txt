# SDK-independent hosting core: interfaces, the lock-free plugin chain and
# chain state. The JUCE adapter lives in juce/ and is added only with JUCE.
add_library(soniclab_host_core STATIC
    src/plugin_chain.cpp
    src/chain_state.cpp)
add_library(soniclab::host_core ALIAS soniclab_host_core)
target_include_directories(soniclab_host_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(soniclab_host_core
    PUBLIC soniclab::common
    PRIVATE soniclab_warnings)
set_target_properties(soniclab_host_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
