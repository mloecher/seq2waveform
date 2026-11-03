// Compare fast_float and std::from_chars bit patterns and timings on shape tokens.
// Usage: bench_scan <file.seq> [repeats]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <charconv>

#include <fast_float/fast_float.h>

// Apple's libc++ ships only the integral half of P0067: there is no floating-point
// std::from_chars, and the bool overload is deleted, so a double argument lands on the deleted
// one and fails to compile. That gap is a large part of why fast_float is vendored at all, so
// where it applies this benchmark has nothing to compare against and reports only fast_float.
// __cpp_lib_to_chars is the feature-test macro for the floating-point half. Define the macro on
// the command line to force either path -- that is how the fallback gets compiled on a toolchain
// that does have std::from_chars.
#ifndef S2W_STD_F64_FROM_CHARS
#if defined(__cpp_lib_to_chars)
#define S2W_STD_F64_FROM_CHARS 1
#else
#define S2W_STD_F64_FROM_CHARS 0
#endif
#endif

#include "seq2waveform/parser.hpp"
#include "seq2waveform/scan.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// The `[SHAPES]` sample lines: the parser's dominant float workload.
std::vector<std::string> shape_sample_tokens(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string raw;
    bool in_shapes = false;
    while (std::getline(in, raw)) {
        const std::string_view t = s2w::trim(raw);
        if (t.empty() || t.front() == '#') {
            continue;
        }
        if (t.front() == '[' && t.back() == ']') {
            in_shapes = (t == "[SHAPES]");
            continue;
        }
        if (!in_shapes || s2w::starts_with(t, "shape_id") || s2w::starts_with(t, "num_samples")) {
            continue;
        }
        out.emplace_back(t);
    }
    return out;
}

constexpr bool kHaveStd = S2W_STD_F64_FROM_CHARS != 0;

#if S2W_STD_F64_FROM_CHARS
/// Parse with std::from_chars, matching parse_f64's rules (whole token, leading '+' stripped).
bool std_parse(std::string_view s, double& out) {
    if (!s.empty() && s.front() == '+') {
        s.remove_prefix(1);
    }
    const char* end = s.data() + s.size();
    const std::from_chars_result r = std::from_chars(s.data(), end, out);
    return r.ec == std::errc() && r.ptr == end;
}
#else
/// Stands in so the call sites still compile; every one of them is behind `if constexpr`.
bool std_parse(std::string_view, double&) { return false; }
#endif

/// The same, through fast_float. Identical contract: no whitespace skipping, no leading '+'.
bool ff_parse(std::string_view s, double& out) {
    if (!s.empty() && s.front() == '+') {
        s.remove_prefix(1);
    }
    const char* end = s.data() + s.size();
    const fast_float::from_chars_result r = fast_float::from_chars(s.data(), end, out);
    return r.ec == std::errc() && r.ptr == end;
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

/// Sum the tokens `reps` times, returning the best wall time in ms. The sum is returned so the
/// loop cannot be optimised away.
double time_parser(bool (*parse)(std::string_view, double&), const std::vector<std::string>& toks,
                   int reps, double& checksum) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const Clock::time_point t0 = Clock::now();
        double sum = 0.0;
        for (const std::string& tok : toks) {
            double v = 0.0;
            if (parse(tok, v)) {
                sum += v;
            }
        }
        best = std::min(best, ms_since(t0));
        checksum = sum;
    }
    return best;
}

}  // namespace

/// Report parser behavior on floating-point overflow and underflow.
void report_out_of_range() {
    const char* cases[] = {"1e999", "-1e999", "1e-999", "-1e-999"};
    std::printf("out-of-range behaviour (ec==0 means accepted):\n");
    for (const char* c : cases) {
        const std::string_view s(c);
        const char* end = s.data() + s.size();

        double b = 12345.0;
        const fast_float::from_chars_result rb = fast_float::from_chars(s.data(), end, b);

#if S2W_STD_F64_FROM_CHARS
        double a = 12345.0;
        const std::from_chars_result ra = std::from_chars(s.data(), end, a);
        std::printf("  %-8s std: ec=%-2d value=%-24.17g | fast_float: ec=%-2d value=%.17g\n", c,
                    static_cast<int>(ra.ec), a, static_cast<int>(rb.ec), b);
#else
        std::printf("  %-8s std: unavailable                      | fast_float: ec=%-2d "
                    "value=%.17g\n",
                    c, static_cast<int>(rb.ec), b);
#endif
    }
    std::printf("  (errc::result_out_of_range == %d)\n\n",
                static_cast<int>(std::errc::result_out_of_range));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: bench_scan <file.seq> [reps]\n");
        return 2;
    }
    if (std::string_view(argv[1]) == "--out-of-range") {
        report_out_of_range();
        return 0;
    }
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::printf("cannot open %s\n", argv[1]);
        return 2;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const std::vector<std::string> toks = shape_sample_tokens(text);
    if (toks.empty()) {
        std::printf("%-12s no shape samples\n", argv[1]);
        return 0;
    }

    // 1. Bit-identity over every token, both directions of agreement.
    std::size_t mismatches = 0;
    std::size_t disagree_ok = 0;
    if constexpr (kHaveStd) {
        for (const std::string& tok : toks) {
            double a = 0.0;
            double b = 0.0;
            const bool oka = std_parse(tok, a);
            const bool okb = ff_parse(tok, b);
            if (oka != okb) {
                ++disagree_ok;
            } else if (oka && !same_bits(a, b)) {
                if (mismatches < 3) {
                    std::printf("  MISMATCH on \"%s\": %.17g vs %.17g\n", tok.c_str(), a, b);
                }
                ++mismatches;
            }
        }
    }

    // 2. Throughput on those tokens, and the whole-file parse for scale.
    double sum_std = 0.0;
    double sum_ff = 0.0;
    double ms_std = 0.0;
    if constexpr (kHaveStd) {
        ms_std = time_parser(std_parse, toks, reps, sum_std);
    }
    const double ms_ff = time_parser(ff_parse, toks, reps, sum_ff);

    // parse_text, not parse(istream): the file is already in memory, so this times the parser
    // rather than the read.
    double ms_parse = 1e300;
    std::size_t blocks = 0;
    for (int r = 0; r < reps; ++r) {
        const Clock::time_point t0 = Clock::now();
        const s2w::Sequence seq = s2w::parse_text(text);
        ms_parse = std::min(ms_parse, ms_since(t0));
        blocks = seq.blocks.size();
    }

    const double n = static_cast<double>(toks.size());
    char std_col[64];
    if constexpr (kHaveStd) {
        std::snprintf(std_col, sizeof(std_col), "%8.2f ms (%5.1f ns/tok)  %5.2fx", ms_std,
                      ms_std * 1e6 / n, ms_std / ms_ff);
    } else {
        std::snprintf(std_col, sizeof(std_col), "%34s", "unavailable in this libc++");
    }
    std::printf(
        "%-12s %9zu tok  %6zu blk | from_chars %s  fast_float %8.2f ms "
        "(%5.1f ns/tok) | parse %8.2f ms, floats %4.1f%% | %s\n",
        argv[1], toks.size(), blocks, std_col, ms_ff, ms_ff * 1e6 / n, ms_parse,
        // Share of parse time spent scanning floats -- measured with fast_float, since that is
        // what the parser now uses. (It is a close estimate, not a profile: the micro-benchmark
        // covers the [SHAPES] tokens, ~95% of the floats in a large sequence.)
        100.0 * ms_ff / ms_parse,
        !kHaveStd ? "fast_float only"
                  : (mismatches == 0 && disagree_ok == 0) ? "bit-identical" : "DIFFERS");

    if (mismatches != 0 || disagree_ok != 0) {
        std::printf("  %zu value mismatch(es), %zu accept/reject disagreement(s)\n", mismatches,
                    disagree_ok);
        return 1;
    }
    if constexpr (kHaveStd) {
        if (!same_bits(sum_std, sum_ff)) {
            std::printf("  checksums differ: %.17g vs %.17g\n", sum_std, sum_ff);
            return 1;
        }
    }
    return 0;
}
