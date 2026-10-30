#include "seq2waveform/scan.hpp"

#include <charconv>
#include <cmath>
#include <limits>

#if !defined(__cpp_lib_to_chars) || __cpp_lib_to_chars < 201611L
#error "seq2waveform needs std::from_chars for floating point (C++17 <charconv>): MSVC 19.24+, \
libstdc++ 11+ or libc++ 20+. Older Apple Clang lacks it -- see PORTING.md, Known gaps."
#endif

namespace s2w {
namespace {

/// The ASCII subset of Rust's `char::is_whitespace`.
constexpr bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

/// Parse the whole of `s` as `T`, Rust-style: a leading '+' is allowed, trailing characters are
/// not, and an out-of-range value fails.
template <typename T>
bool parse_whole(std::string_view s, T& out) {
    if (!s.empty() && s.front() == '+') {
        s.remove_prefix(1);  // from_chars rejects it, Rust's parse() does not
    }
    const char* end = s.data() + s.size();
    T value{};
    const std::from_chars_result r = std::from_chars(s.data(), end, value);
    if (r.ec != std::errc() || r.ptr != end) {
        return false;  // empty, malformed, out of range, or trailing characters
    }
    out = value;
    return true;
}

}  // namespace

std::string_view trim(std::string_view s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && is_space(s[b])) {
        ++b;
    }
    while (e > b && is_space(s[e - 1])) {
        --e;
    }
    return s.substr(b, e - b);
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

std::vector<std::string_view> fields(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && is_space(s[i])) {
            ++i;
        }
        if (i >= s.size()) {
            break;
        }
        const std::size_t start = i;
        while (i < s.size() && !is_space(s[i])) {
            ++i;
        }
        out.push_back(s.substr(start, i - start));
    }
    return out;
}

std::pair<std::string_view, std::string_view> split_first_space(std::string_view s) {
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (is_space(s[i])) {
            return {s.substr(0, i), trim(s.substr(i + 1))};
        }
    }
    return {s, std::string_view{}};
}

bool parse_f64(std::string_view s, double& out) {
    return parse_whole(s, out);
}

bool parse_u32(std::string_view s, std::uint32_t& out) {
    return parse_whole(s, out);
}

bool parse_usize(std::string_view s, std::size_t& out) {
    return parse_whole(s, out);
}

std::int64_t saturating_i64(double x) {
    if (std::isnan(x)) {
        return 0;  // Rust's float-to-int cast maps NaN to 0
    }
    // 2^63 is exactly representable, and is the first double above the int64 range.
    constexpr double kLimit = 9223372036854775808.0;
    if (x >= kLimit) {
        return std::numeric_limits<std::int64_t>::max();
    }
    if (x <= -kLimit) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return static_cast<std::int64_t>(x);
}

}  // namespace s2w
