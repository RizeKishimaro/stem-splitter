# Float32 model loading and a one-thread inference worker. ONNX Runtime is
# an implementation detail of this library. Plugins include
# soniclab/inference/*.h and do not link the runtime themselves, except
# through this target.
option(SONICLAB_WITH_INFERENCE "Build the ONNX Runtime inference library" ON)
if(NOT SONICLAB_WITH_INFERENCE)
    return()
endif()

set(SONICLAB_ONNXRUNTIME_ROOT "" CACHE PATH "Prebuilt ONNX Runtime root (include/ and lib/). Empty downloads the pinned release.")
include(${PROJECT_SOURCE_DIR}/cmake/OnnxRuntime.cmake)
soniclab_onnxruntime_resolve(SONICLAB_ORT_ROOT)

add_library(soniclab_onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(soniclab_onnxruntime PROPERTIES
    IMPORTED_LOCATION "${SONICLAB_ORT_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${SONICLAB_ORT_ROOT}/include")

add_library(soniclab_inference STATIC
    src/tensor.cpp
    src/model.cpp
    src/service.cpp)
add_library(soniclab::inference ALIAS soniclab_inference)
target_include_directories(soniclab_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(soniclab_inference SYSTEM PRIVATE "${SONICLAB_ORT_ROOT}/include")
target_link_libraries(soniclab_inference
    PUBLIC Threads::Threads
    PRIVATE soniclab_onnxruntime soniclab_warnings)
set_target_properties(soniclab_inference PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(SONICLAB_BUILD_TESTS)
    add_executable(soniclab_inference_tests
        ${PROJECT_SOURCE_DIR}/tests/test_main.cpp
        ${PROJECT_SOURCE_DIR}/tests/test_inference.cpp)
    target_link_libraries(soniclab_inference_tests PRIVATE soniclab::inference soniclab_warnings)
    target_compile_definitions(soniclab_inference_tests PRIVATE
        SONICLAB_INFERENCE_TESTDATA="${CMAKE_CURRENT_SOURCE_DIR}/testdata")
    set_target_properties(soniclab_inference_tests PROPERTIES
        BUILD_RPATH "${SONICLAB_ORT_ROOT}/lib")
    add_test(NAME inference COMMAND soniclab_inference_tests inference.)
    set_tests_properties(inference PROPERTIES
        TIMEOUT 120
        ENVIRONMENT "LD_LIBRARY_PATH=${SONICLAB_ORT_ROOT}/lib")
    if(SONICLAB_SANITIZE_ADDRESS)
        set_property(TEST inference APPEND PROPERTY ENVIRONMENT
            "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp")
    endif()
endif()
