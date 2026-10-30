// Locale-independent scanning of the plain-text sequence formats (.seq now, .dsv/.inf later).
//
// Reproducing Rust's `str::parse` semantics here is a requirement, not a nicety: a token that
// parses to a different double on either side would make every downstream comparison drift, and
// the drift would look like a bug in the waveform code rather than in the reader. Three things
// matter.
//
// * The whole token must be consumed. "1.5x" is a failure, not 1.5 -- Rust's parse() rejects
//   trailing characters, where C's strtod would quietly stop early.
// * A leading '+' is accepted. Rust accepts it; `from_chars` does not, so it is stripped.
// * Rounding is correct and locale-independent. `strtod` honours the C locale's decimal separator,
//   so under a comma locale it would read "1e-05" as 1 -- `from_chars` never does.
//
// Doubles go through vendored fast_float rather than `std::from_chars` (integers still use the
// standard one). Three reasons, in `extern/fast_float/README.md`: libc++ only gained the
// floating-point overload in 2025 and Xcode's copy lags, which would break macOS wheels; the three
// standard libraries disagree with each other on out-of-range input; and Rust's own `dec2flt` is
// this same Eisel-Lemire algorithm, so the port and the oracle now parse floats identically by
// construction rather than by coincidence. Correct rounding is a uniqueness guarantee -- there is
// exactly one nearest double -- and `tests/bench_scan.cpp` confirms it empirically.

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
