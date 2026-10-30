# The seq2waveform_core static library, shared by the Python build (../CMakeLists.txt) and the
# C++ tests (../tests/CMakeLists.txt). Add every ported translation unit here -- the bindings and
# the tests then both pick it up with no further wiring.
#
# cxx_std_17 matches gropt's core target, so the two libraries can be linked into one build if a
# shared waveform core is ever wanted.

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

# fast_float supplies from_chars for doubles (see src/include/seq2waveform/scan.hpp and
# extern/fast_float/README.md). Vendored by default so the sdist and the conda recipe build with no
# network; ON is for packagers who unbundle. PRIVATE: it is an implementation detail of scan.cpp and
# appears in no public header.
option(S2W_USE_SYSTEM_FAST_FLOAT "Use an installed fast_float instead of the vendored copy" OFF)
if(S2W_USE_SYSTEM_FAST_FLOAT)
    find_package(FastFloat CONFIG REQUIRED)
    target_link_libraries(seq2waveform_core PRIVATE FastFloat::fast_float)
else()
    target_include_directories(seq2waveform_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../extern)
endif()
target_compile_options(seq2waveform_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/MP>)  # parallel compile
# MSVC: C++ exception unwinding. Normally in CMake's defaults, but absent if CMAKE_CXX_FLAGS is overridden.
target_compile_options(seq2waveform_core PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)

set(_s2w_gnuish "$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>")

# No fused multiply-add, so the golden snapshots hold across compilers. GCC and Clang contract
# `a + b*c` into one FMA by default, which changes the last bit. x86-64 has no FMA in its
# baseline so it never showed there, but arm64 does -- the trapezoid sums in pwl.cpp would have
# moved. MSVC is left alone: it emits no FMA at its x64 default, so it already matches.
# PUBLIC because the inline helpers in the public headers compile in callers' TUs too.
target_compile_options(seq2waveform_core PUBLIC $<${_s2w_gnuish}:-ffp-contract=off>)

# Warnings where CI will see them; development is on MSVC, which stays at /W3. Not -Werror.
target_compile_options(seq2waveform_core PRIVATE $<${_s2w_gnuish}:-Wall;-Wextra>)
