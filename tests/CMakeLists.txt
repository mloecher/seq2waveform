# C++ core tests. Enabled with -DS2W_BUILD_TESTS=ON (see `pixi run test-cpp`).
# Add one executable per ported module and register it with ctest.

add_executable(test_pwl test_pwl.cpp)
target_link_libraries(test_pwl PRIVATE seq2waveform_core)
add_test(NAME pwl COMMAND test_pwl)

add_executable(test_parser test_parser.cpp)
target_link_libraries(test_parser PRIVATE seq2waveform_core)
add_test(NAME parser COMMAND test_parser)

# Guards the vendored fast_float (see extern/fast_float/README.md): are fast_float and this
# toolchain's std::from_chars bit-identical on real shape data, and what does each cost? Run it
# after bumping the vendored version. Not a ctest -- it needs a dataset path.
add_executable(bench_scan bench_scan.cpp)
target_link_libraries(bench_scan PRIVATE seq2waveform_core)
target_include_directories(bench_scan PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../extern)
