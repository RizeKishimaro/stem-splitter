# SonicLab native code: realtime-safe shared code, plugin-host interfaces and
# the Beat Analyzer plugin. Independent of the Python package one level up.
#
# Without JUCE (the default) this builds and tests the shared code, the
# JUCE-free host core and the analyzer pipeline with nothing but a C++17
# compiler. SONICLAB_WITH_JUCE=ON adds the JUCE host adapter and the plugin;
# see README.md for how JUCE is obtained and LICENSING.md before shipping.
cmake_minimum_required(VERSION 3.22)

project(SonicLabNative
    VERSION 0.1.0
    DESCRIPTION "SonicLab native audio components"
    LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
endif()

option(SONICLAB_BUILD_TESTS "Build the native unit tests" ON)
option(SONICLAB_BUILD_ENGINE_CAPI "Build the C ABI engine prototype (libsoniclab_engine)" ON)
set(SONICLAB_PYTHON "" CACHE FILEPATH "Python with numpy and soniclab importable; adds the ctypes tests to CTest")
option(SONICLAB_WITH_JUCE "Build JUCE targets (plugin host adapter, Beat Analyzer plugin)" OFF)
option(SONICLAB_FETCH_JUCE "Download JUCE with FetchContent when SONICLAB_JUCE_DIR is not set" OFF)
set(SONICLAB_JUCE_DIR "" CACHE PATH "Path to a JUCE source checkout (takes precedence over fetching)")
set(SONICLAB_JUCE_TAG "8.0.4" CACHE STRING "JUCE release tag used by SONICLAB_FETCH_JUCE")
option(SONICLAB_BUILD_VST3 "Also build the plugin as VST3 (see LICENSING.md)" OFF)
option(SONICLAB_SANITIZE_THREAD "Build with ThreadSanitizer" OFF)
option(SONICLAB_SANITIZE_ADDRESS "Build with AddressSanitizer and UBSan" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Warnings for SonicLab's own targets only; JUCE has its own flags.
add_library(soniclab_warnings INTERFACE)
if(MSVC)
    target_compile_options(soniclab_warnings INTERFACE /W4)
else()
    target_compile_options(soniclab_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                                                       -Wno-sign-conversion)
endif()

if(SONICLAB_SANITIZE_THREAD AND SONICLAB_SANITIZE_ADDRESS)
    message(FATAL_ERROR "ThreadSanitizer and AddressSanitizer cannot be combined")
endif()
if(SONICLAB_SANITIZE_THREAD)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
elseif(SONICLAB_SANITIZE_ADDRESS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)

# Before any add_subdirectory: add_test in a directory processed earlier
# than enable_testing() is silently ignored.
if(SONICLAB_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(common)
add_subdirectory(plugin_host)
add_subdirectory(plugins/shared)
add_subdirectory(plugins/beat_analyzer)
add_subdirectory(inference)
if(SONICLAB_BUILD_ENGINE_CAPI)
    add_subdirectory(engine_capi)
endif()

if(SONICLAB_WITH_JUCE)
    # JUCE compiles some C sources (bundled codecs, zlib); C must be enabled
    # at the top level or the generate step fails.
    enable_language(C)
    include(cmake/SonicLabJuce.cmake)
    soniclab_add_juce()
    include(cmake/SonicLabPlugin.cmake)
    add_subdirectory(plugin_host/juce)
    add_subdirectory(plugins/beat_analyzer/juce)
endif()

if(SONICLAB_BUILD_TESTS)
    add_subdirectory(tests)
endif()

message(STATUS "SonicLab native: JUCE targets ${SONICLAB_WITH_JUCE}, tests ${SONICLAB_BUILD_TESTS}, inference ${SONICLAB_WITH_INFERENCE}")
