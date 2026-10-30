cmake_minimum_required(VERSION 3.17)
project(seq2waveform LANGUAGES CXX)

# Make and Ninja give no optimisation at all with an empty build type. Multi-config generators
# (Visual Studio, Xcode) ignore this and take --config at build time, so don't force it there.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ON for a wheel build, which is every `pip install`. OFF builds the core library and the C++
# tests alone, which need neither Python nor nanobind -- the configuration CI uses to check that
# the core compiles and passes on a toolchain, without installing a Python stack to do it.
option(S2W_BUILD_MODULE "Build the Python extension module" ON)
# OFF by default so `pip install` stays lean; `pixi run test-cpp` turns it on.
option(S2W_BUILD_TESTS "Build the C++ core tests" OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/src/seq2waveform_core.cmake)  # -> seq2waveform_core

if(S2W_BUILD_MODULE)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Detect nanobind -- prefer the installed package, fall back to find_package.
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)

    # FREE_THREADED: the module declares it doesn't need the GIL, so 3.14t keeps it disabled.
    # nanobind turns the option off itself on a non-free-threaded interpreter, so it's a no-op
    # there. Safe because nothing here holds mutable shared state: no globals, no caches, every
    # public method const, results freshly allocated. Revisit if that stops being true.
    nanobind_add_module(_core FREE_THREADED bindings/seq2waveform_wrapper.cpp)
    target_link_libraries(_core PRIVATE seq2waveform_core)

    install(TARGETS _core LIBRARY DESTINATION seq2waveform)
endif()

# Stub generation imports the built module, so it only works for native builds. Skip when
# cross-compiling on macOS (e.g. building x86_64 on an arm64 runner), and when there is no module.
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

# This is the whole reason the C++ port gets documented stubs for free: the .pyi is regenerated
# from the built module on every build, docstrings and NDArray annotations included.
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
