# Official ONNX Runtime CPU builds. The shared library is downloaded on
# configure; it is not committed. Set SONICLAB_ONNXRUNTIME_ROOT to a
# directory that already contains include/ and lib/ to skip the download.
#
# Only the Linux x86_64 archive hash below was computed on the development
# machine. Other archives use the same upstream version and have not been
# executed here. Core ML is not part of the Linux package.

set(SONICLAB_ONNXRUNTIME_VERSION "1.20.1" CACHE STRING "ONNX Runtime release to download")

function(soniclab_onnxruntime_resolve out_root)
    if(SONICLAB_ONNXRUNTIME_ROOT)
        set(${out_root} "${SONICLAB_ONNXRUNTIME_ROOT}" PARENT_SCOPE)
        return()
    endif()

    set(version "${SONICLAB_ONNXRUNTIME_VERSION}")
    set(hash "")
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(archive "onnxruntime-osx-arm64-${version}.tgz")
    elseif(APPLE)
        set(archive "onnxruntime-osx-x86_64-${version}.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(archive "onnxruntime-linux-aarch64-${version}.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(archive "onnxruntime-linux-x64-${version}.tgz")
        set(hash "SHA256=67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105")
    else()
        message(FATAL_ERROR
            "No ONNX Runtime archive is selected for ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}. "
            "Set SONICLAB_ONNXRUNTIME_ROOT to a local build.")
    endif()

    include(FetchContent)
    set(declare_args
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${version}/${archive}")
    if(hash)
        list(APPEND declare_args URL_HASH "${hash}")
    endif()
    FetchContent_Declare(soniclab_onnxruntime ${declare_args})
    FetchContent_GetProperties(soniclab_onnxruntime)
    if(NOT soniclab_onnxruntime_POPULATED)
        FetchContent_Populate(soniclab_onnxruntime)
    endif()

    set(root "${soniclab_onnxruntime_SOURCE_DIR}")
    if(NOT EXISTS "${root}/include/onnxruntime_cxx_api.h")
        file(GLOB nested LIST_DIRECTORIES true "${root}/onnxruntime-*")
        list(LENGTH nested nested_count)
        if(nested_count EQUAL 1)
            set(root "${nested}")
        endif()
    endif()
    if(NOT EXISTS "${root}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime headers were not found under ${soniclab_onnxruntime_SOURCE_DIR}")
    endif()
    set(${out_root} "${root}" PARENT_SCOPE)
endfunction()
