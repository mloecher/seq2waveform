// Core tests for the Siemens `.dsv` reader, ported one-for-one from the Rust unit tests in
// crates/seq2waveform/src/dsv.rs, with the minimal file included verbatim.
//
// The property worth stating: the `[VALUES]` body is a run-length-encoded *derivative*, decoded
// straight to piecewise-linear corners without ever building the dense array. Sampling those
// corners at `(k+0.5)*raster` must reproduce `cumsum(deltas)/VERTFACTOR` exactly -- linear
// interpolation of a line is the line -- and that round-trip is the decoder's own test.
//
// No test framework on purpose: one translation unit, plain checks, non-zero exit on failure.

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "seq2waveform/detail/dsv_detail.hpp"
#include "seq2waveform/dsv.hpp"
#include "seq2waveform/error.hpp"
#include "seq2waveform/time.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void close(double a, double b, const char* what, double tol = 1e-12) {
    if (!(std::fabs(a - b) < tol)) {
        std::printf("FAIL: %s (%.17g vs %.17g)\n", what, a, b);
        ++failures;
    }
}

void rle_decodes_to_corners_and_dense() {
    // [0,0,3] -> 0 repeated 5 times; then 7 once. deltas = [0,0,0,0,0,7].
    const auto decoded = s2w::detail::rle_to_piecewise({0, 0, 3, 7}, 1.0, 1.0);
    const s2w::PiecewiseLinear& w = decoded.first;
    check(decoded.second == 6, "six dense samples");
    // the collinear zeros collapse: corners at samples 0, 4, 5
    check(w.size() == 3, "three corners after the collinear drop");
    // the dense reconstruction matches cumsum(deltas)
    std::vector<double> dense;
    for (std::size_t k = 0; k < decoded.second; ++k) {
        dense.push_back(w.value_at(s2w::Ps::from_secs(static_cast<double>(k) + 0.5)));
    }
    check(dense == std::vector<double>({0.0, 0.0, 0.0, 0.0, 0.0, 7.0}),
          "dense round-trips to cumsum(deltas)");
}

const char* kMiniDsv =
    "; a comment\n"
    "[FILETYPE]\n"
    "FORMAT=DSV_V0100\n"
    "\n"
    "[DEFINITIONS]\n"
    "HORIDELTA=10.000000\n"
    "VERTFACTOR=1000000.000000\n"
    "VERTUNITNAME=mT/m\n"
    "SAMPLES=6\n"
    "\n"
    "[FRAME]\n"
    "\n"
    "[VALUES]\n"
    "0\n"
    "0\n"
    "3\n"
    "1000000\n"
    ";  end of graphic data file\n";

s2w::DsvChannel parse_dsv(const char* text) {
    std::istringstream in(text);
    return s2w::read_dsv_channel(in);
}

void parses_a_minimal_dsv() {
    const s2w::DsvChannel ch = parse_dsv(kMiniDsv);
    check(ch.size() == 6, "six samples");
    check(ch.unit == "mT/m", "unit name");
    close(ch.raster_time, 10e-6, "raster time is HORIDELTA in seconds");

    // dense round-trips to cumsum(deltas)/VERTFACTOR
    const std::vector<double> dense = ch.dense();
    check(dense.size() == 6, "six dense values");
    if (dense.size() == 6) {
        bool zeros = true;
        for (std::size_t i = 0; i < 5; ++i) {
            zeros = zeros && dense[i] == 0.0;
        }
        check(zeros, "the first five dense values are zero");
        close(dense[5], 1.0, "the last dense value is VERTFACTOR scaled to 1.0", 1e-9);
    }
    close(ch.duration(), 60e-6, "duration is samples * raster");

    // the flat zeros collapsed to a few corners, not six dense samples
    check(ch.waveform.size() < ch.num_samples, "the decode is sparser than the dense form");
}

void a_comment_before_any_section_is_not_a_terminator() {
    // The leading `; a comment` sits outside [VALUES], where it must be ignored rather than
    // ending the stream -- the terminator only counts inside the value section.
    const s2w::DsvChannel ch = parse_dsv(kMiniDsv);
    check(ch.size() == 6, "the leading comment did not truncate the channel");
}

void a_bad_value_line_raises() {
    const char* bad =
        "[DEFINITIONS]\n"
        "HORIDELTA=10.0\n"
        "VERTFACTOR=1.0\n"
        "[VALUES]\n"
        "0\n"
        "not-a-number\n";
    try {
        parse_dsv(bad);
        check(false, "a non-integer value line raises");
    } catch (const s2w::DsvError& e) {
        // The full text is what reaches Python, and what the parity suite compares.
        check(std::string(e.what()) == "DSV format error: non-integer value line: \"not-a-number\"",
              "the error quotes the offending line");
    }
}

void missing_definitions_raise() {
    try {
        parse_dsv("[DEFINITIONS]\nHORIDELTA=10.0\n[VALUES]\n0\n");
        check(false, "a missing VERTFACTOR raises");
    } catch (const s2w::DsvError& e) {
        check(std::string(e.message()) == "missing or invalid VERTFACTOR",
              "the error names the missing key");
    }
    try {
        parse_dsv("[DEFINITIONS]\nVERTFACTOR=1.0\n[VALUES]\n0\n");
        check(false, "a missing HORIDELTA raises");
    } catch (const s2w::DsvError& e) {
        check(std::string(e.message()) == "missing or invalid HORIDELTA",
              "the error names the missing key");
    }
}

void an_empty_value_stream_is_not_an_error() {
    // A channel header with no samples: legal, and the result is simply empty.
    const s2w::DsvChannel ch = parse_dsv("[DEFINITIONS]\nHORIDELTA=10.0\nVERTFACTOR=1.0\n[VALUES]\n");
    check(ch.size() == 0, "no samples");
    check(ch.empty(), "reports empty");
    check(ch.waveform.empty(), "no corners");
    close(ch.duration(), 0.0, "zero duration");
    check(ch.dense().empty(), "no dense values");
}

void a_single_sample_channel_keeps_its_one_corner() {
    // The first sample and the last sample are the same sample here; the final-sample guard must
    // not emit it twice.
    const s2w::DsvChannel ch =
        parse_dsv("[DEFINITIONS]\nHORIDELTA=10.0\nVERTFACTOR=1.0\n[VALUES]\n5\n");
    check(ch.size() == 1, "one sample");
    check(ch.waveform.size() == 1, "exactly one corner, not two");
    const std::vector<double> dense = ch.dense();
    check(dense.size() == 1 && dense[0] == 5.0, "the single value round-trips");
}

void a_long_run_stays_sparse() {
    // 10,000 equal deltas is one constant-slope segment: two corners, not 10,000. This is the
    // whole point of decoding to corners rather than to the dense array.
    const auto decoded = s2w::detail::rle_to_piecewise({3, 3, 9998}, 1.0, 1.0);
    check(decoded.second == 10000, "ten thousand samples");
    check(decoded.first.size() == 2, "a constant-slope run is two corners");
    // and it still reconstructs exactly
    const s2w::PiecewiseLinear& w = decoded.first;
    close(w.value_at(s2w::Ps::from_secs(0.5)), 3.0, "first sample");
    close(w.value_at(s2w::Ps::from_secs(9999.5)), 30000.0, "last sample", 1e-9);
    close(w.value_at(s2w::Ps::from_secs(5000.5)), 15003.0, "a sample inside the run", 1e-9);
}

}  // namespace

int main() {
    rle_decodes_to_corners_and_dense();
    parses_a_minimal_dsv();
    a_comment_before_any_section_is_not_a_terminator();
    a_bad_value_line_raises();
    missing_definitions_raise();
    an_empty_value_stream_is_not_an_error();
    a_single_sample_channel_keeps_its_one_corner();
    a_long_run_stays_sparse();

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
