// Pulseq `.seq` file parser (v1.4/1.5).
//
// Plain text in `[SECTION]` blocks. Walk the file once, dispatching each line to the current
// section's row parser to fill a Sequence's event libraries; cross-references (block -> event id)
// resolve later, in the renderer. Per-table unit scales are applied here, so the result is in
// canonical units (seconds, Hz, Hz/m, radians).

#pragma once

#include <istream>
#include <string>

#include "seq2waveform/model.hpp"

namespace s2w {

/// Parse a sequence from any input stream.
///
/// The I/O-agnostic entry point, as in Rust: the bindings pass a file, tests pass a string.
/// Throws `ParseError` on a malformed line, naming the 1-based line number.
Sequence parse(std::istream& in);

/// Parse a sequence from a file on disk. Throws `std::runtime_error` if it cannot be opened.
Sequence parse_file(const std::string& path);

}  // namespace s2w
