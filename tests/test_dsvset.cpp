// Tests for DsvSet units, event metadata and the shared sampling interface.

#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include "seq2waveform/constants.hpp"
#include "seq2waveform/dsvset.hpp"

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

/// A channel with `n` samples at `raster`, and the given corners (times in seconds).
s2w::DsvChannel channel(std::size_t n, double raster, const std::vector<double>& t,
                        const std::vector<double>& v) {
    s2w::DsvChannel ch;
    ch.raster_time = raster;
    ch.unit = "mT/m";
    ch.num_samples = n;
    for (const double x : t) {
        ch.waveform.t.push_back(s2w::Ps::from_secs(x));
    }
    ch.waveform.v = v;
    return ch;
}

/// An INF with one ADC segment, one trigger and one RF event, built directly.
s2w::Inf synthetic_inf() {
    s2w::Inf inf;

    s2w::AdcSegment seg;
    seg.start = 1e-3;
    seg.duration = 8e-6;
    seg.num_samples = 4;
    inf.adc.push_back(seg);

    inf.columns = {"Rel.Time", "TXFlip/Dur", "Freq/Phase", "SYNC/Dur"};
    s2w::EventBlock b;
    b.start = 2e-3;
    s2w::EventRow trig;
    trig.rel_time = 0.0;
    trig.cells = {"0", "", "", "[undef]: EXTRIG0/ 10"};
    b.rows.push_back(trig);
    s2w::EventRow rf;
    rf.rel_time = 100e-6;
    // flip 90 deg over 9 us; the Freq/Phase cell gives -32000 Hz and 180 degrees of phase
    rf.cells = {"100", "[sX]: 90/9", "[sX zSet]: -32000/ 180", ""};
    b.rows.push_back(rf);
    inf.blocks.push_back(b);

    return inf;
}

s2w::DsvSet make_set(double gamma) {
    // x spans 10 samples, y 20, z 5 -> the set's duration is y's
    return s2w::DsvSet(channel(10, 10e-6, {0.5e-5, 9.5e-5}, {0.0, 1.0}),
                       channel(20, 10e-6, {0.5e-5, 19.5e-5}, {0.0, 2.0}),
                       channel(5, 10e-6, {0.5e-5, 4.5e-5}, {0.0, 3.0}),
                       channel(9, 1e-6, {0.5e-6, 4.5e-6, 8.5e-6}, {0.0, 5.0, 0.0}),
                       synthetic_inf(), gamma);
}

void gradients_convert_mt_per_m_to_hz_per_m() {
    const s2w::DsvSet set = make_set(s2w::kScanGamma);
    // Hz/m = mT/m * 1e-3 * gamma. Nothing else may touch the value, and the times are already in
    // the Siemens frame, so they pass through unchanged.
    const double factor = s2w::kScanGamma * 1e-3;
    const s2w::PiecewiseLinear gx = set.gradient('x');
    check(gx.size() == 2, "two corners on x");
    if (gx.size() == 2) {
        close(gx.v[0], 0.0, "x starts at zero");
        close(gx.v[1], 1.0 * factor, "x is scaled by gamma * 1e-3", 1e-6);
        close(gx.t[1].secs(), 9.5e-5, "the corner times are not shifted");
    }
    const s2w::PiecewiseLinear gz = set.gradient('z');
    if (gz.size() == 2) {
        close(gz.v[1], 3.0 * factor, "z is scaled the same way", 1e-6);
    }
    close(set.gamma(), s2w::kScanGamma, "gamma is reported back");
}

void a_custom_gamma_scales_the_gradients() {
    const s2w::DsvSet set = make_set(1.0e6);
    const s2w::PiecewiseLinear gx = set.gradient('x');
    if (gx.size() == 2) {
        close(gx.v[1], 1.0 * 1.0e6 * 1e-3, "a caller-supplied gamma is used", 1e-9);
    }
}

void an_unknown_axis_is_empty() {
    const s2w::DsvSet set = make_set(s2w::kScanGamma);
    check(set.gradient('q').empty(), "an unknown axis gives an empty channel, not an error");
}

void duration_is_the_longest_channel() {
    const s2w::DsvSet set = make_set(s2w::kScanGamma);
    // x 10 samples, y 20, z 5, all at 10 us -> 200 us. Taking any single channel would be wrong;
    // the gradients span the whole run only collectively.
    close(set.duration(), 20 * 10e-6, "duration is the span of the longest channel");
}

void adc_events_carry_no_freq_or_phase() {
    const s2w::DsvSet set = make_set(s2w::kScanGamma);
    const std::vector<s2w::AdcEvent> ev = set.adc_events();
    check(ev.size() == 1, "one ADC event");
    if (!ev.empty()) {
        check(ev[0].t.size() == 4, "four samples");
        // Deliberate zeros: the INF records no per-sample freq/phase, and inventing one would be
        // worse than reporting none.
        check(ev[0].freq == 0.0 && ev[0].phase == 0.0, "freq and phase are zero, not guessed");
        close(ev[0].t[0].secs(), 1e-3 + 1e-6, "first sample is at a dwell center");
    }
}

void triggers_have_no_channel() {
    const s2w::DsvSet set = make_set(s2w::kScanGamma);
    const std::vector<s2w::TriggerEvent> tr = set.trigger_events();
    check(tr.size() == 1, "one trigger");
    if (!tr.empty()) {
        check(tr[0].channel == 0 && tr[0].trig_type == 0,
              "channel and type are zero: the SYNC column carries neither");
        close(tr[0].time.secs(), 2e-3, "time is block start + Rel.Time");
        close(tr[0].duration.secs(), 10e-6, "duration is us -> s");
    }
}

void excitation_phase_is_converted_to_radians() {
    const s2w::DsvSet set = make_set(s2w::kScanGamma);
    const auto exc = set.excitations();
    check(exc.size() == 1, "one excitation");
    if (!exc.empty()) {
        // The INF stores phase in DEGREES; the interface is radians. 180 deg -> pi.
        close(std::get<2>(exc[0]), 3.14159265358979323846, "phase is degrees -> radians", 1e-12);
        close(std::get<1>(exc[0]), -32000.0, "freq passes through in Hz", 1e-9);
        // The INF has no excite/refocus type, so every RF event counts as an excitation.
        check(std::get<0>(exc[0]).raw() > 0, "the center is in the Siemens frame");
    }
}

void the_sampler_pipeline_runs_on_a_dsv_set() {
    // Exercise shared sampling helpers on a DSV source.
    const s2w::DsvSet set = make_set(s2w::kScanGamma);
    const s2w::Sampler s = set.sampler();
    const s2w::Trace g = s.gradients({s2w::Ps::from_secs(5e-5)});
    check(g.x.size() == 1 && std::isfinite(g.x[0]), "the sampler evaluates gradients");

    const std::vector<s2w::SampleSpec> specs = set.adc_specs();
    check(specs.size() == 1, "one ADC spec");
    const s2w::FlatTrace flat = set.adc_kspace_flat();
    check(flat.t.size() == 4, "four flat k-space points");
    check(flat.offsets.size() == 2 && flat.offsets.front() == 0 && flat.offsets.back() == 4,
          "the event offsets bracket the whole array");
}

}  // namespace

int main() {
    gradients_convert_mt_per_m_to_hz_per_m();
    a_custom_gamma_scales_the_gradients();
    an_unknown_axis_is_empty();
    duration_is_the_longest_channel();
    adc_events_carry_no_freq_or_phase();
    triggers_have_no_channel();
    excitation_phase_is_converted_to_radians();
    the_sampler_pipeline_runs_on_a_dsv_set();

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
