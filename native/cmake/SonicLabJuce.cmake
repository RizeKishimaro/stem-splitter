# Locates JUCE for SONICLAB_WITH_JUCE builds. In order of preference:
#
#   1. SONICLAB_JUCE_DIR   a JUCE source tree you checked out yourself.
#                          The JUCE presets default it to native/third_party/JUCE,
#                          which native/scripts/get_juce.sh fills.
#   2. find_package(JUCE)  a JUCE you installed with `cmake --install`
#   3. SONICLAB_FETCH_JUCE FetchContent of the pinned SONICLAB_JUCE_TAG
#
# Nothing is downloaded unless SONICLAB_FETCH_JUCE is ON. JUCE is dual
# licensed (AGPLv3 or commercial); read native/LICENSING.md first.

function(soniclab_add_juce)
    if(TARGET juce::juce_core)
        return()
    endif()

    if(SONICLAB_JUCE_DIR AND EXISTS "${SONICLAB_JUCE_DIR}/CMakeLists.txt")
        message(STATUS "JUCE: using source tree ${SONICLAB_JUCE_DIR}")
        add_subdirectory("${SONICLAB_JUCE_DIR}" "${CMAKE_BINARY_DIR}/_juce" EXCLUDE_FROM_ALL)
        return()
    endif()

    find_package(JUCE CONFIG QUIET)
    if(JUCE_FOUND)
        message(STATUS "JUCE: using installed package ${JUCE_DIR}")
        return()
    endif()

    if(NOT SONICLAB_FETCH_JUCE)
        message(FATAL_ERROR
            "SONICLAB_WITH_JUCE is ON but JUCE was not found "
            "(no JUCE source tree at '${SONICLAB_JUCE_DIR}').\n"
            "Run native/scripts/get_juce.sh (clones the pinned ${SONICLAB_JUCE_TAG} into native/third_party/JUCE), "
            "or set -DSONICLAB_JUCE_DIR=<path to JUCE>, "
            "or allow a download with -DSONICLAB_FETCH_JUCE=ON.")
    endif()

    include(FetchContent)
    message(STATUS "JUCE: fetching tag ${SONICLAB_JUCE_TAG} from GitHub")
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG ${SONICLAB_JUCE_TAG}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(JUCE)
endfunction()
