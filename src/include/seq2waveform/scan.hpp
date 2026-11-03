// Locale-independent scanners for .seq, .dsv and INF text. Numeric parsers consume
// the whole token and accept a leading +. Doubles use the vendored fast_float parser.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace s2w {

/// Trim ASCII whitespace from both ends.
std::string_view trim(std::string_view s);

/// True when `s` begins with `prefix`.
bool starts_with(std::string_view s, std::string_view prefix);

/// Split on ASCII whitespace. Returned views borrow from s.
std::vector<std::string_view> fields(std::string_view s);

/// Split at the first whitespace: {before, trim(after)}, or {s, ""} if none.
std::pair<std::string_view, std::string_view> split_first_space(std::string_view s);

/// Parse the whole of `s` as a double, leaving `out` untouched and returning false if it is not a
/// complete valid number.
bool parse_f64(std::string_view s, double& out);

/// Parse a whole u32 token, accepting + and rejecting negatives, decimals and overflow.
bool parse_u32(std::string_view s, std::uint32_t& out);

/// Parse a size_t token with the same rules as parse_u32.
bool parse_usize(std::string_view s, std::size_t& out);

/// Parse a whole int64 token, accepting either sign and rejecting decimals and overflow.
bool parse_i64(std::string_view s, std::int64_t& out);

/// Convert to int64, clamping at the bounds and mapping NaN to zero.
/// Used for file repeat counts and raster indices.
std::int64_t saturating_i64(double x);

}  // namespace s2w
