# C++ core tests. Enabled with -DS2W_BUILD_TESTS=ON (see `pixi run test-cpp`).
# Add one executable per ported module and register it with ctest.

add_executable(test_pwl test_pwl.cpp)
target_link_libraries(test_pwl PRIVATE seq2waveform_core)
add_test(NAME pwl COMMAND test_pwl)
