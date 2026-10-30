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

# No fused multiply-add. This is what makes the golden snapshots portable, so it is a correctness
# setting here and not a tuning knob.
#
# The snapshots in tests/golden/ pin every output bit-for-bit, and they were written by MSVC on
# x86-64. GCC defaults to -ffp-contract=fast and Clang to =on, both of which let the compiler fuse
# `a + b * c` into a single FMA with one rounding instead of two -- a different last bit. On
# x86-64 that is mostly moot, because the base ISA has no FMA instruction to fuse into; on arm64
# it is not, because FMA is in the baseline. So the Apple Silicon wheel is exactly where this
# would have bitten, and the trapezoid accumulations in pwl.cpp (`cum[i-1] + 0.5*(v0+v1)*dt`) and
# the k-space sums in waveform.cpp are exactly the expressions it would have changed.
#
# Pinning it off on the GNU-like compilers makes them reproduce what MSVC already produces, rather
# than changing what MSVC produces -- the committed snapshots stay valid. MSVC is deliberately left
# alone: its x64 default targets SSE2 and emits no FMA, so it already behaves this way, and
# touching its FP flags would risk moving the very numbers the snapshots pin.
#
# The cost is losing FMA in those loops on arm64 and on any AVX2 build. Unmeasured here, since
# there is no arm64 machine to measure on, and the hot paths are dominated by parsing and memory
# traffic rather than arithmetic. If an arm64 benchmark ever says otherwise, the alternative is a
# second set of snapshots for arm64, which is a much worse trade.
#
# PUBLIC, because arithmetic also compiles in consumers' translation units through the inline
# helpers in the public headers; the bindings and the C++ tests must see the same rule.
target_compile_options(seq2waveform_core PUBLIC $<${_s2w_gnuish}:-ffp-contract=off>)

# Warnings on the compilers this is not developed against, where CI is the only reader. Not
# -Werror: the first gcc/clang build of this code has never happened, and a warning should start a
# conversation rather than block the matrix. MSVC stays at its default /W3 so the local build does
# not become noisy; /W4 is a one-word change here if that is ever wanted.
target_compile_options(seq2waveform_core PRIVATE $<${_s2w_gnuish}:-Wall;-Wextra>)
