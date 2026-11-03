cmake_minimum_required(VERSION 3.17)
project(seq2waveform LANGUAGES CXX)

# Single-config generators need an explicit optimized build type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(S2W_BUILD_MODULE "Build the Python extension module" ON)
option(S2W_BUILD_TESTS "Build the C++ core tests" OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/src/seq2waveform_core.cmake)  # -> seq2waveform_core

if(S2W_BUILD_MODULE)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)

    # FREE_THREADED keeps the GIL off on supported interpreters.
    # Review shared-state access before adding mutable caches or globals.
    nanobind_add_module(_core FREE_THREADED bindings/seq2waveform_wrapper.cpp)
    target_link_libraries(_core PRIVATE seq2waveform_core)

    install(TARGETS _core LIBRARY DESTINATION seq2waveform)
endif()

# Stub generation imports the module, so skip it for cross-compiled binaries.
set(_generate_stubs ${S2W_BUILD_MODULE})
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE _host_arch
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL _host_arch)
        set(_generate_stubs OFF)
    endif()
endif()

if(_generate_stubs)
    nanobind_add_stub(_core_stub
        MODULE _core
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/seq2waveform/_core.pyi"
        PYTHON_PATH "$<TARGET_FILE_DIR:_core>"
        DEPENDS _core
    )
endif()

if(S2W_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
