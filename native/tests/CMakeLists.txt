add_executable(soniclab_native_tests
    test_main.cpp
    test_common.cpp
    test_host.cpp
    test_beat_analyzer.cpp)
target_link_libraries(soniclab_native_tests PRIVATE
    soniclab::common
    soniclab::host_core
    soniclab::beat_analyzer_core
    soniclab_warnings)

# One CTest entry per group; the runner filters by name prefix.
foreach(group common host beat_analyzer)
    add_test(NAME ${group} COMMAND soniclab_native_tests ${group}.)
    set_tests_properties(${group} PROPERTIES TIMEOUT 60)
endforeach()

if(TARGET soniclab_engine)
    # The C ABI exercised from C++, the same way ctypes calls it.
    add_executable(soniclab_engine_capi_tests test_main.cpp test_engine_capi.cpp)
    target_link_libraries(soniclab_engine_capi_tests PRIVATE soniclab::engine soniclab_warnings Threads::Threads)
    add_test(NAME engine_capi COMMAND soniclab_engine_capi_tests engine_capi.)
    set_tests_properties(engine_capi PROPERTIES TIMEOUT 60)

    if(SONICLAB_PYTHON)
        set(_python_env "SONICLAB_NATIVE_LIB=$<TARGET_FILE:soniclab_engine>")
        set(_python_test ON)
        if(SONICLAB_SANITIZE_ADDRESS)
            # Python is not instrumented, so the ASan runtime must be loaded
            # before it. GCC ships it as a shared library; other compilers
            # link it statically, so the ctypes test is skipped for them.
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
                execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
                                OUTPUT_VARIABLE _asan OUTPUT_STRIP_TRAILING_WHITESPACE)
                execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libubsan.so
                                OUTPUT_VARIABLE _ubsan OUTPUT_STRIP_TRAILING_WHITESPACE)
                list(APPEND _python_env "LD_PRELOAD=${_asan} ${_ubsan}" "ASAN_OPTIONS=detect_leaks=0")
            else()
                set(_python_test OFF)
            endif()
        endif()
        if(_python_test)
            add_test(NAME python_ctypes
                COMMAND ${SONICLAB_PYTHON} -m unittest discover -s ${PROJECT_SOURCE_DIR}/python -p "test_*.py" -v)
            set_tests_properties(python_ctypes PROPERTIES TIMEOUT 120 ENVIRONMENT "${_python_env}")
        endif()
    endif()
endif()
