set(SEQ2WAVEFORM_CORE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pwl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/parser.cpp
    ${CMAKE_CURRENT_LIST_DIR}/waveform.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sample.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dsv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/inf.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dsvset.cpp
)

add_library(seq2waveform_core STATIC ${SEQ2WAVEFORM_CORE_SOURCES})
target_compile_features(seq2waveform_core PUBLIC cxx_std_17)
set_target_properties(seq2waveform_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(seq2waveform_core PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

# Default to vendored headers for offline source builds; allow packagers to unbundle.
option(S2W_USE_SYSTEM_FAST_FLOAT "Use an installed fast_float instead of the vendored copy" OFF)
if(S2W_USE_SYSTEM_FAST_FLOAT)
    find_package(FastFloat CONFIG REQUIRED)
    target_link_libraries(seq2waveform_core PRIVATE FastFloat::fast_float)
else()
    target_include_directories(seq2waveform_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../extern)
endif()
target_compile_options(seq2waveform_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/MP>)  # parallel compile
target_compile_options(seq2waveform_core PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)

set(_s2w_gnuish "$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>")

# Disable FMA contraction to keep golden results stable across architectures.
# PUBLIC also covers inline functions compiled by callers.
target_compile_options(seq2waveform_core PUBLIC $<${_s2w_gnuish}:-ffp-contract=off>)

target_compile_options(seq2waveform_core PRIVATE $<${_s2w_gnuish}:-Wall;-Wextra>)
