cmake_minimum_required(VERSION 3.17)
project(seq2waveform LANGUAGES CXX)

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

    # FREE_THREADED declares that this module does not need the GIL, which is what lets a
    # free-threaded interpreter (3.14t) keep it disabled instead of re-enabling it at import and
    # warning. nanobind sets the option to FALSE by itself when the interpreter is not
    # free-threaded, so this is unconditional and a no-op on a normal CPython build.
    #
    # What makes the claim true, audited rather than assumed: there is no namespace-scope mutable
    # state in src/ or bindings/, no `mutable` members, no lazily-computed or cached fields, and
    # no Python objects held in C++. Every public method on the shared types (Sequence, DsvSet,
    # PiecewiseLinear and the INF/DSV readers) is const, so two threads calling into one object
    # are both reading. The single function-local static -- `static const DsvChannel empty` in
    # dsvset.cpp -- is const and initialised once, which C++11 makes thread-safe. Results are
    # freshly allocated arrays handed to numpy with their own capsule owners, so no buffer is
    # shared between calls. The two std::fprintf(stderr) warnings are locked by the C runtime;
    # concurrent warnings can interleave in the text but cannot race.
    #
    # If any of that changes -- a cache, a lazily filled field, a non-const accessor -- this flag
    # has to be reconsidered, and nb::ft_mutex is the tool for it.
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
