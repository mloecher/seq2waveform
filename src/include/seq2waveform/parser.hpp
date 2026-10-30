// Pulseq `.seq` file parser (v1.4/1.5).
//
// Plain text in `[SECTION]` blocks. Walk the file once, dispatching each line to the current
// section's row parser to fill a Sequence's event libraries; cross-references (block -> event id)
// resolve later, in the renderer. Per-table unit scales are applied here, so the result is in
// canonical units (seconds, Hz, Hz/m, radians).

#pragma once

#include <istream>
#include <string>
#include <string_view>

#include "seq2waveform/model.hpp"

namespace s2w {

/// Parse a sequence already held in memory. The fastest entry point: it walks the text in place,
/// without copying a line at a time.
///
/// Throws `ParseError` on a malformed line, naming the 1-based line number.
Sequence parse_text(std::string_view text);

/// Parse a sequence from any input stream, reading it fully and handing it to `parse_text`.
///
/// The I/O-agnostic entry point, as in Rust: the bindings pass a file, tests pass a string.
Sequence parse(std::istream& in);

/// Parse a sequence from a file on disk. Throws `std::runtime_error` if it cannot be opened.
Sequence parse_file(const std::string& path);

}  // namespace s2w
