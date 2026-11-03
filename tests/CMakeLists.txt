foreach(name IN ITEMS pwl parser waveform sample dsv inf dsvset)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE seq2waveform_core)
    add_test(NAME ${name} COMMAND test_${name})
endforeach()

add_executable(bench_scan bench_scan.cpp)
target_link_libraries(bench_scan PRIVATE seq2waveform_core)
target_include_directories(bench_scan PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../extern)

add_executable(bench_expand bench_expand.cpp)
target_link_libraries(bench_expand PRIVATE seq2waveform_core)
