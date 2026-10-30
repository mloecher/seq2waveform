// Locale-independent scanning of the plain-text sequence formats (.seq now, .dsv/.inf later).
//
// Reproducing Rust's `str::parse` semantics here is a requirement, not a nicety: a token that
// parses to a different double on either side would make every downstream comparison drift, and
// the drift would look like a bug in the waveform code rather than in the reader. Three things
// matter.
//
// * The whole token must be consumed. "1.5x" is a failure, not 1.5 -- Rust's parse() rejects
//   trailing characters, where C's strtod would quietly stop early.
// * A leading '+' is accepted. Rust accepts it; `std::from_chars` does not, so it is stripped.
// * Rounding is correctly rounded and locale-independent. `strtod` honours the C locale's decimal
//   separator, so under a comma locale it would read "1e-05" as 1 -- `from_chars` never does.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace s2w {

/// Trim ASCII whitespace from both ends.
///
/// Rust's `str::trim` trims Unicode whitespace. The two agree on ASCII input, which is all a
/// Pulseq text file holds; see PORTING.md on non-ASCII input.
std::string_view trim(std::string_view s);

/// True when `s` begins with `prefix`.
bool starts_with(std::string_view s, std::string_view prefix);

/// Split on runs of ASCII whitespace, dropping empty fields -- Rust's `split_whitespace`.
///
/// The returned views borrow from `s`, so it has to outlive them.
std::vector<std::string_view> fields(std::string_view s);

/// Split at the first whitespace character: `{before, trim(after)}`, or `{s, ""}` if there is
/// none. Rust's `split_once(char::is_whitespace)` together with the trim its callers apply.
std::pair<std::string_view, std::string_view> split_first_space(std::string_view s);

/// Parse the whole of `s` as a double, leaving `out` untouched and returning false if it is not a
/// complete valid number.
///
/// One divergence from Rust, unreachable in real data: a token that overflows or underflows the
/// double range ("1e999") fails here, where Rust saturates to infinity or zero.
bool parse_f64(std::string_view s, double& out);

/// Parse the whole of `s` as a u32 -- Rust's `parse::<u32>()`. Rejects a `-` sign, a decimal
/// point and overflow; accepts a leading `+`, as Rust does.
bool parse_u32(std::string_view s, std::uint32_t& out);

/// Parse the whole of `s` as a size_t -- Rust's `parse::<usize>()`, same rules as `parse_u32`.
bool parse_usize(std::string_view s, std::size_t& out);

/// Rust's `as i64` cast on a double: saturating at the int64 bounds, with NaN mapping to 0.
///
/// A plain `static_cast` is undefined behaviour outside the range, so a corrupt shape repeat count
/// would be a crash here and a defined (if useless) number in Rust. Shared so every float-to-int
/// cast in the port goes through the same rule.
std::int64_t saturating_i64(double x);

}  // namespace s2w
