// Core tests for waveform expansion, ported one-for-one from the Rust unit tests in
// crates/seq2waveform/src/waveform.rs, SEQ included verbatim.
//
// The two heaviest checks are worth knowing about:
//
// * `resample_jump_is_exact_vs_bruteforce` re-derives the DSV resampling the slow, obvious way --
//   value_at at every raster center, then the same collinear drop -- and demands the fast path's
//   segment-skipping jump produce the identical result. It is the only thing standing between the
//   jump and a silently wrong waveform on long flats.
// * `dsv_mode_reads_zero_outside_the_channel` is a regression test for a bug fixed on the Rust
//   side while this port was underway: the kept center nearest each end sits half a raster inside
//   the first/last ramp, so without an explicit zero center beyond them the channel clamped to a
//   non-zero value before its first event and after its last.
//
// No test framework on purpose: one translation unit, plain checks, non-zero exit on failure.

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "seq2waveform/detail/waveform_detail.hpp"
#include "seq2waveform/model.hpp"
#include "seq2waveform/parser.hpp"
#include "seq2waveform/pwl.hpp"
#include "seq2waveform/time.hpp"
#include "seq2waveform/waveform.hpp"

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

const char* kSeq =
    "[VERSION]\n"
    "major 1\n"
    "minor 5\n"
    "revision 1\n"
    "[DEFINITIONS]\n"
    "GradientRasterTime 1e-05\n"
    "BlockDurationRaster 1e-05\n"
    "RadiofrequencyRasterTime 1e-06\n"
    "[BLOCKS]\n"
    " 1 100 1 0 0 1 0 0\n"
    " 2  50 0 0 0 0 1 1\n"
    "[RF]\n"
    "1 100 1 0 0 500 200 0 0 0 0 e\n"
    "[TRAP]\n"
    "1 100 200 600 200 0\n"
    "[ADC]\n"
    "1 4 1000 40 0 0 0 0 0\n"
    "[EXTENSIONS]\n"
    "1 1 1 0\n"
    "extension TRIGGERS 1\n"
    "1 1 3 0 10\n"
    "[SHAPES]\n"
    "shape_id 1\n"
    "num_samples 2\n"
    "0.5\n"
    "1.0\n";

s2w::Sequence parse_seq() {
    std::istringstream in(kSeq);
    return s2w::parse(in);
}

s2w::Waveforms expand() { return s2w::waveforms(parse_seq()); }

void trapezoid_gradient_corners() {
    const s2w::Waveforms w = expand();
    // block 1 GZ = trap 1: amp 100, rise 200us, flat 600us, fall 200us, at curr_dur 0
    const double expect_t[] = {0.0, 200e-6, 800e-6, 1000e-6};
    check(w.gz.t.size() == 4, "four trapezoid corners");
    if (w.gz.t.size() == 4) {
        for (std::size_t i = 0; i < 4; ++i) {
            close(w.gz.t[i].secs(), expect_t[i], "trapezoid corner time");
        }
    }
    check(w.gz.v == std::vector<double>({0.0, 100.0, 100.0, 0.0}), "trapezoid corner values");
}

void rf_adc_triggers_still_work() {
    const s2w::Waveforms w = expand();
    check(w.rf.size() == 1, "one RF pulse");
    if (!w.rf.empty()) {
        close(w.rf[0].center.secs(), 700e-6, "RF center is delay + center");
        check(w.rf[0].use == s2w::RfUse::Excitation, "RF use survives expansion");
    }
    check(w.adc.size() == 1, "one ADC event");
    if (!w.adc.empty()) {
        check(!w.adc[0].t.empty(), "the ADC event has samples");
        if (!w.adc[0].t.empty()) {
            close(w.adc[0].t[0].secs(), 1e-3 + 40e-6 + 0.5e-6, "first ADC sample is at a dwell center");
        }
    }
    check(w.triggers.size() == 1, "one trigger");
    if (!w.triggers.empty()) {
        close(w.triggers[0].time.secs(), 1e-3, "trigger time is the block start");
    }
}

void restore_recovers_a_triangle() {
    // a triangle: edges 0 -> 2 -> 0 over 2 raster intervals; centers are the midpoints 1, 1
    const double raster = 1.0;
    const std::vector<double> waveform = {1.0, 1.0};  // centers of edges [0, 2, 0]
    const std::vector<double> tt = {0.5, 1.5};
    const auto rec = s2w::detail::restore_additional_shape_samples(tt, waveform, 0.0, 0.0, raster);
    // the corners should be the apex triangle: (0,0), (1,2), (2,0)
    check(rec.first == std::vector<double>({0.0, 1.0, 2.0}), "restored triangle times");
    check(rec.second == std::vector<double>({0.0, 2.0, 0.0}), "restored triangle values");
}

/// The brute-force DSV resampling: value_at at every center plus the same collinear drop, with no
/// segment-skipping jump. The range matches resample_at_centers -- one center past each end, where
/// value_at clamps.
s2w::PiecewiseLinear resample_bruteforce(const s2w::PiecewiseLinear& pwl, double raster) {
    const std::int64_t raster_ps = s2w::Ps::from_secs(raster).raw();
    const std::int64_t half = raster_ps / 2;
    const auto center = [&](std::int64_t k) { return s2w::Ps(raster_ps * k + half); };
    const std::size_t n = pwl.t.size();
    const s2w::Ps t0 = pwl.t[0];
    const s2w::Ps t1 = pwl.t[n - 1];
    const auto k_lo = static_cast<std::int64_t>(std::ceil(t0.secs() / raster - 0.5)) - 1;
    const auto k_hi = static_cast<std::int64_t>(std::floor(t1.secs() / raster - 0.5)) + 1;

    s2w::PiecewiseLinear out;
    for (std::int64_t k = k_lo; k <= k_hi; ++k) {
        const s2w::Ps tc = center(k);
        out.t.push_back(tc);
        out.v.push_back(pwl.value_at(tc));
        while (out.t.size() >= 3) {
            const std::size_t m = out.t.size();
            const std::size_t a = m - 3;
            const std::size_t b = m - 2;
            const std::size_t c = m - 1;
            const double cross = (out.t[b] - out.t[a]).secs() * (out.v[c] - out.v[b]) -
                                 (out.t[c] - out.t[b]).secs() * (out.v[b] - out.v[a]);
            const double scale =
                raster * (std::fmax(std::fmax(std::fabs(out.v[a]), std::fabs(out.v[b])),
                                    std::fabs(out.v[c])) +
                          1.0);
            if (std::fabs(cross) <= 1e-9 * scale) {
                out.t.erase(out.t.begin() + static_cast<std::ptrdiff_t>(b));
                out.v.erase(out.v.begin() + static_cast<std::ptrdiff_t>(b));
            } else {
                break;
            }
        }
    }
    return out;
}

void resample_jump_is_exact_vs_bruteforce() {
    // long flat gaps (which trigger the jump), a ramp, a curve of small segments (no jump), traps
    const double secs[] = {0.0,     1000e-6, 1010e-6, 1020e-6, 1030e-6,
                           1040e-6, 2000e-6, 2050e-6, 2060e-6};
    s2w::PiecewiseLinear pwl;
    for (const double x : secs) {
        pwl.t.push_back(s2w::Ps::from_secs(x));
    }
    pwl.v = {0.0, 0.0, 100.0, 150.0, 130.0, 0.0, 0.0, 500.0, 0.0};
    const double raster = 10e-6;

    const s2w::PiecewiseLinear got = s2w::detail::resample_at_centers(pwl, raster);
    const s2w::PiecewiseLinear want = resample_bruteforce(pwl, raster);

    check(got.t == want.t, "jump vs brute-force: times differ");
    check(got.v == want.v, "jump vs brute-force: values differ");
}

void dsv_mode_rounds_corners_and_stays_sparse() {
    s2w::Sequence seq = parse_seq();
    seq.grad_mode = s2w::GradMode::Dsv;
    // block-1 Z trap: amp 100, rise/flat/fall 200/600/200 us
    const s2w::PiecewiseLinear g = s2w::gradient(seq, 'z');
    // stays piecewise-linear and sparse -- ramps and flats collapse, not one corner per 10 us cell
    check(g.size() < 20, "DSV mode stays sparse");
    close(g.value_at(s2w::Ps::from_secs(400e-6)), 100.0, "DSV plateau is preserved", 1e-6);
    // the sharp edge corner at 200 us is rounded, so it lands just below the plateau
    const double vc = g.value_at(s2w::Ps::from_secs(200e-6));
    check(vc >= 95.0 && vc < 100.0, "DSV rounds the trapezoid corner below the plateau");
}

void dsv_mode_reads_zero_outside_the_channel() {
    s2w::Sequence seq = parse_seq();
    seq.grad_mode = s2w::GradMode::Dsv;
    for (const char axis : {'x', 'y', 'z'}) {
        const s2w::PiecewiseLinear g = s2w::gradient(seq, axis);
        if (g.empty()) {
            continue;
        }
        check(g.v.front() == 0.0, "DSV channel starts at zero");
        check(g.v.back() == 0.0, "DSV channel ends at zero");
        // far outside the corner range, where value_at clamps
        check(g.value_at(g.t.front() - s2w::Ps::from_secs(1.0)) == 0.0,
              "DSV channel reads zero before its first event");
        check(g.value_at(g.t.back() + s2w::Ps::from_secs(1.0)) == 0.0,
              "DSV channel reads zero after its last event");
    }
}

void gradient_axis_selection_matches_the_block_table() {
    // Only GZ carries the trapezoid in this sequence; x and y must come back empty rather than
    // picking up another axis's events.
    const s2w::Sequence seq = parse_seq();
    check(s2w::gradient(seq, 'z').size() == 4, "z has the trapezoid");
    check(s2w::gradient(seq, 'x').empty(), "x is empty");
    check(s2w::gradient(seq, 'y').empty(), "y is empty");
    check(s2w::gradient(seq, 'q').empty(), "an unknown axis is empty, not an error");
}

void adc_times_matches_the_per_event_split() {
    // adc_times is a focused walk and adc_events is another; they must not disagree.
    const s2w::Sequence seq = parse_seq();
    const std::vector<s2w::Ps> flat = s2w::adc_times(seq);
    std::vector<s2w::Ps> joined;
    for (const s2w::AdcEvent& e : s2w::adc_events(seq)) {
        joined.insert(joined.end(), e.t.begin(), e.t.end());
    }
    check(flat == joined, "adc_times equals the concatenated per-event times");
}

}  // namespace

int main() {
    trapezoid_gradient_corners();
    rf_adc_triggers_still_work();
    restore_recovers_a_triangle();
    resample_jump_is_exact_vs_bruteforce();
    dsv_mode_rounds_corners_and_stays_sparse();
    dsv_mode_reads_zero_outside_the_channel();
    gradient_axis_selection_matches_the_block_table();
    adc_times_matches_the_per_event_split();

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
