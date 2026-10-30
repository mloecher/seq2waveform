// Internals of inf.cpp, exposed only so the tests can reach them.
//
// `merge_contiguous` undoes Siemens's `MaxAdcSegmentLength` chunking and the Rust unit test drives
// it directly with hand-built segments, which is far clearer than synthesising an INF with a split
// readout. `split_cell` is the whole of the table-cell grammar. Nothing outside inf.cpp and
// tests/test_inf.cpp should include this.

#pragma once

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "seq2waveform/inf.hpp"

namespace s2w {
namespace detail {

/// Merge consecutive ADC segments that are contiguous in time and share a dwell.
std::vector<AdcSegment> merge_contiguous(const std::vector<AdcSegment>& segs);

/// Split a cell like `[label]: a/ b` into the trimmed `(a, b)` after the `[label]:` prefix. Empty
/// when there is no `/`, or when the first part is blank.
std::optional<std::pair<std::string_view, std::string_view>> split_cell(std::string_view s);

}  // namespace detail
}  // namespace s2w
