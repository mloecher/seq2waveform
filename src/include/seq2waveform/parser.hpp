// Pulseq .seq parser using the 1.5 event layout. Units are converted to seconds,
// Hz, Hz/m and radians here; event references are resolved during waveform expansion.

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

/// Buffer an input stream and pass it to parse_text.
Sequence parse(std::istream& in);

/// Parse a sequence from a file on disk. Throws `std::runtime_error` if it cannot be opened.
Sequence parse_file(const std::string& path);

}  // namespace s2w
