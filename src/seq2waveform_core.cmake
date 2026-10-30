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
    # Port order (see ../PORTING.md): waveform.cpp, sample.cpp, dsv.cpp, inf.cpp
)

add_library(seq2waveform_core STATIC ${SEQ2WAVEFORM_CORE_SOURCES})
target_compile_features(seq2waveform_core PUBLIC cxx_std_17)
set_target_properties(seq2waveform_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(seq2waveform_core PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_options(seq2waveform_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/MP>)  # parallel compile
# MSVC: C++ exception unwinding. Normally in CMake's defaults, but absent if CMAKE_CXX_FLAGS is overridden.
target_compile_options(seq2waveform_core PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
