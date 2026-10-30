# C++ core tests. Enabled with -DS2W_BUILD_TESTS=ON (see `pixi run test-cpp`).
# Add one executable per ported module and register it with ctest.

add_executable(test_pwl test_pwl.cpp)
target_link_libraries(test_pwl PRIVATE seq2waveform_core)
add_test(NAME pwl COMMAND test_pwl)

add_executable(test_parser test_parser.cpp)
target_link_libraries(test_parser PRIVATE seq2waveform_core)
add_test(NAME parser COMMAND test_parser)

add_executable(test_waveform test_waveform.cpp)
target_link_libraries(test_waveform PRIVATE seq2waveform_core)
add_test(NAME waveform COMMAND test_waveform)

add_executable(test_sample test_sample.cpp)
target_link_libraries(test_sample PRIVATE seq2waveform_core)
add_test(NAME sample COMMAND test_sample)

add_executable(test_dsv test_dsv.cpp)
target_link_libraries(test_dsv PRIVATE seq2waveform_core)
add_test(NAME dsv COMMAND test_dsv)

add_executable(test_inf test_inf.cpp)
target_link_libraries(test_inf PRIVATE seq2waveform_core)
add_test(NAME inf COMMAND test_inf)

# Guards the vendored fast_float (see extern/fast_float/README.md): are fast_float and this
# toolchain's std::from_chars bit-identical on real shape data, and what does each cost? Run it
# after bumping the vendored version. Not a ctest -- it needs a dataset path.
add_executable(bench_scan bench_scan.cpp)
target_link_libraries(bench_scan PRIVATE seq2waveform_core)
target_include_directories(bench_scan PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../extern)
