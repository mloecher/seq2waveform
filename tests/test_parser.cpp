// Core tests for the model and the .seq parser, ported one-for-one from the Rust unit tests in
// crates/seq2waveform/src/{model.rs,parser/mod.rs}, MINI_SEQ included verbatim. Keeping them in
// step is the cheapest evidence that the port preserved the reader.
//
// The scanning tests at the end have no Rust counterpart: they pin the behaviour that Rust gets
// from `str::parse` and that C++ had to reproduce by hand (whole-token match, a leading '+',
// locale independence, and a saturating float-to-int cast).
//
// No test framework on purpose: one translation unit, plain checks, non-zero exit on failure.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "seq2waveform/error.hpp"
#include "seq2waveform/model.hpp"
#include "seq2waveform/parser.hpp"
#include "seq2waveform/scan.hpp"

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

void equal(const std::string& a, const std::string& b, const char* what) {
    if (a != b) {
        std::printf("FAIL: %s (\"%s\" vs \"%s\")\n", what, a.c_str(), b.c_str());
        ++failures;
    }
}

const char* kMiniSeq =
    "# comment\n"
    "[VERSION]\n"
    "major 1\n"
    "minor 5\n"
    "revision 1\n"
    "\n"
    "[DEFINITIONS]\n"
    "GradientRasterTime 1e-05\n"
    "BlockDurationRaster 1e-05\n"
    "Name mini\n"
    "\n"
    "[BLOCKS]\n"
    " 1 640   1   0   0   1  0  0\n"
    " 2  20   0   0   0   0  1  1\n"
    "\n"
    "[RF]\n"
    "1 130.843 1 2 0 3000 200 0 0 -32000 0 e\n"
    "\n"
    "[TRAP]\n"
    "1 666667 200 6000 200 0\n"
    "\n"
    "[ADC]\n"
    "1 20000 4000 40 0 0 0 0 0\n"
    "\n"
    "[EXTENSIONS]\n"
    "1 1 1 0\n"
    "\n"
    "extension TRIGGERS 1\n"
    "1 1 3 0 10\n"
    "\n"
    "[SHAPES]\n"
    "\n"
    "shape_id 1\n"
    "num_samples 5\n"
    "1\n"
    "0\n"
    "0\n"
    "2\n";

s2w::Sequence parse_str(const std::string& s) {
    std::istringstream in(s);
    return s2w::parse(in);
}

// --- model ---------------------------------------------------------------

void shape_uncompressed_passthrough() {
    s2w::Shape s;
    s.num_samples = 3;
    s.data = {0.1, 0.2, 0.3};
    const std::vector<double> got = s.decompressed();  // len == num_samples -> no cumsum
    check(got == std::vector<double>({0.1, 0.2, 0.3}), "uncompressed shape passes through");
}

void shape_rle_derivative_then_cumsum() {
    // data [1, 0,0,2] = derivative "1, then 0 repeated (2+2)=4 times" -> [1,0,0,0,0]
    // (data.len() 4 < num_samples 5 -> compressed path); cumsum -> [1,1,1,1,1]
    s2w::Shape s;
    s.num_samples = 5;
    s.data = {1.0, 0.0, 0.0, 2.0};
    check(s.decompressed() == std::vector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
          "RLE derivative expands then accumulates");
}

// --- parser --------------------------------------------------------------

void parses_version_and_definitions() {
    const s2w::Sequence seq = parse_str(kMiniSeq);
    check(seq.version == s2w::Version{1, 5, 1}, "version is 1.5.1");
    const auto bdr = seq.definition_f64("BlockDurationRaster");
    check(bdr.has_value() && *bdr == 1e-5, "BlockDurationRaster reads as 1e-5");
    const auto name = seq.definitions.find("Name");
    check(name != seq.definitions.end() && name->second == "mini", "a text definition is kept raw");
    check(seq.definition_f64("GradientRasterTime").has_value(), "a numeric definition parses");
    check(!seq.definition_f64("Name").has_value(), "a text definition does not parse as a number");
    check(!seq.definition_f64("NoSuchKey").has_value(), "a missing definition is absent");
}

void parses_blocks_with_durations_and_ids() {
    const s2w::Sequence seq = parse_str(kMiniSeq);
    check(seq.blocks.size() == 2, "two blocks");
    if (seq.blocks.size() != 2) {
        return;
    }
    const s2w::Block& b1 = seq.blocks[0];
    check(b1.num == 1 && b1.rf == 1 && b1.gz == 1 && b1.adc == 0, "block 1 event ids");
    close(b1.duration, 640.0 * 1e-5, "block 1 duration is 6.4 ms");
    const s2w::Block& b2 = seq.blocks[1];
    check(b2.adc == 1 && b2.ext == 1, "block 2 carries the ADC and the extension");
}

void parses_events_with_unit_scaling() {
    const s2w::Sequence seq = parse_str(kMiniSeq);

    const s2w::Rf& rf = seq.rf.at(1);
    close(rf.amplitude, 130.843, "rf amplitude is Hz, unscaled", 1e-6);
    close(rf.center, 3000e-6, "rf center is us -> s");
    close(rf.delay, 200e-6, "rf delay is us -> s");
    check(rf.use == s2w::RfUse::Excitation, "rf use 'e' is excitation");

    const s2w::Trap* g = std::get_if<s2w::Trap>(&seq.gradients.at(1));
    check(g != nullptr, "gradient 1 is a trapezoid");
    if (g != nullptr) {
        close(g->amplitude, 666667.0, "trap amplitude is Hz/m", 1e-6);
        close(g->rise, 200e-6, "trap rise is us -> s");
        close(g->flat, 6000e-6, "trap flat is us -> s");
    }

    const s2w::Adc& adc = seq.adc.at(1);
    check(adc.num == 20000, "adc sample count");
    close(adc.dwell, 4000e-9, "adc dwell is ns -> s (4 us)", 1e-15);
    close(adc.delay, 40e-6, "adc delay is us -> s");

    const s2w::Trigger& trg = seq.triggers.at(1);
    check(trg.channel == 3, "trigger channel");
    close(trg.duration, 10e-6, "trigger duration is us -> s");
    check(seq.ext_type_names.at(1) == "TRIGGERS", "extension type name is recorded");
}

void parses_and_decompresses_shape() {
    const s2w::Sequence seq = parse_str(kMiniSeq);
    const s2w::Shape& s = seq.shapes.at(1);
    check(s.num_samples == 5, "shape num_samples");
    check(s.decompressed() == std::vector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
          "shape decompresses to five ones");
}

void parse_error_carries_the_source_line() {
    // line 4 has a non-numeric trap `rise`
    const std::string bad = "[VERSION]\nmajor 1\n[TRAP]\n1 100 x 6000 200 0\n";
    try {
        parse_str(bad);
        check(false, "a malformed trap row raises");
    } catch (const s2w::ParseError& e) {
        check(e.line() == 4, "the error names line 4");
        equal(e.message(), "missing/invalid trap rise", "the error names the field");
        // The full text is what reaches Python, so it is pinned here.
        equal(e.what(), "parse error at line 4: missing/invalid trap rise", "formatted message");
    }
}

void blank_and_comment_lines_still_count_toward_the_line_number() {
    // Rust enumerates lines before skipping blanks and comments, so the reported number is the
    // real file line. Easy to get wrong by counting only content lines.
    const std::string bad = "# a comment\n\n[TRAP]\n\n# another\n1 100 x 6000 200 0\n";
    try {
        parse_str(bad);
        check(false, "a malformed trap row raises");
    } catch (const s2w::ParseError& e) {
        check(e.line() == 6, "blank and comment lines are counted");
    }
}

void a_missing_use_field_is_undefined() {
    // v1.4 files have no `use` column; that is the common case, not an error.
    const std::string v14 = "[RF]\n1 130.843 1 2 0 3000 200 0 0 -32000 0\n";
    const s2w::Sequence seq = parse_str(v14);
    check(seq.rf.at(1).use == s2w::RfUse::Undefined, "no use column -> undefined");
    equal(s2w::rf_use_name(seq.rf.at(1).use), "undefined", "undefined has a name");
}

void block_durations_resolve_against_a_later_definition() {
    // [DEFINITIONS] after [BLOCKS] still has to scale the durations, which is why the raw rows are
    // held back until the whole file is read.
    const std::string late = "[BLOCKS]\n1 640 0 0 0 0 0 0\n[DEFINITIONS]\nBlockDurationRaster 1e-05\n";
    const s2w::Sequence seq = parse_str(late);
    check(seq.blocks.size() == 1, "one block");
    if (!seq.blocks.empty()) {
        close(seq.blocks[0].duration, 640.0 * 1e-5, "duration uses the later raster");
    }
}

void block_durations_default_to_a_ten_microsecond_raster() {
    const s2w::Sequence seq = parse_str("[BLOCKS]\n1 640 0 0 0 0 0 0\n");
    check(seq.blocks.size() == 1, "one block");
    if (!seq.blocks.empty()) {
        close(seq.blocks[0].duration, 640.0 * 10e-6, "default raster is 10 us");
    }
}

void an_unknown_section_is_skipped() {
    const s2w::Sequence seq = parse_str("[SOMETHING_NEW]\nwhatever 1 2 3\n[BLOCKS]\n1 1 0 0 0 0 0 0\n");
    check(seq.blocks.size() == 1, "rows of an unknown section are ignored, not fatal");
}

// --- scanning ------------------------------------------------------------

void scanning_matches_rust_parse() {
    double d = 0.0;
    check(s2w::parse_f64("1e-05", d) && d == 1e-5, "exponent form");
    check(s2w::parse_f64("+1.5", d) && d == 1.5, "a leading + is accepted, as Rust does");
    check(s2w::parse_f64("-32000", d) && d == -32000.0, "negative");
    check(!s2w::parse_f64("1.5x", d), "trailing characters are a failure, not a partial parse");
    check(!s2w::parse_f64("", d), "an empty token fails");
    check(!s2w::parse_f64("x", d), "a non-number fails");
    check(!s2w::parse_f64(" 1.5", d), "untrimmed input fails (callers trim first)");

    // Out of range. Rust saturates: "1e999" parses to +inf, "1e-999" to 0.0. The three standard
    // libraries all disagree here (WG21 P4168 exists because of it), which is part of why doubles
    // go through fast_float instead.
    check(s2w::parse_f64("1e999", d) && std::isinf(d) && d > 0, "overflow saturates to +inf");
    check(s2w::parse_f64("-1e999", d) && std::isinf(d) && d < 0, "overflow saturates to -inf");
    check(s2w::parse_f64("1e-999", d) && d == 0.0, "underflow saturates to 0");

    std::uint32_t u = 0;
    check(s2w::parse_u32("20000", u) && u == 20000, "plain integer");
    check(s2w::parse_u32("+7", u) && u == 7, "a leading + is accepted");
    check(!s2w::parse_u32("-1", u), "a negative value is not a u32");
    check(!s2w::parse_u32("1.0", u), "a decimal point is not a u32");
    check(!s2w::parse_u32("4294967296", u), "overflow fails");
}

void saturating_cast_matches_rust() {
    check(s2w::saturating_i64(3.7) == 3, "truncates toward zero");
    check(s2w::saturating_i64(-3.7) == -3, "truncates toward zero when negative");
    check(s2w::saturating_i64(std::nan("")) == 0, "NaN becomes 0, as Rust's `as i64` does");
    check(s2w::saturating_i64(1e300) == std::numeric_limits<std::int64_t>::max(), "saturates high");
    check(s2w::saturating_i64(-1e300) == std::numeric_limits<std::int64_t>::min(), "saturates low");
}

}  // namespace

int main() {
    shape_uncompressed_passthrough();
    shape_rle_derivative_then_cumsum();
    parses_version_and_definitions();
    parses_blocks_with_durations_and_ids();
    parses_events_with_unit_scaling();
    parses_and_decompresses_shape();
    parse_error_carries_the_source_line();
    blank_and_comment_lines_still_count_toward_the_line_number();
    a_missing_use_field_is_undefined();
    block_durations_resolve_against_a_later_definition();
    block_durations_default_to_a_ten_microsecond_raster();
    an_unknown_section_is_skipped();
    scanning_matches_rust_parse();
    saturating_cast_matches_rust();

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
