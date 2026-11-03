// Tests for sampling and k-space. The test trapezoid has area 0.09 1/m:
// 100 Hz/m * (800 us flat + 100 us from the ramps).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "seq2waveform/parser.hpp"
#include "seq2waveform/sample.hpp"
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

// excitation at t=0; a Z trapezoid (amp 100 Hz/m, area 0.09 1/m); a 3-sample ADC in block 2.
const char* kAdcSeq =
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
    " 2 100 0 0 0 0 1 0\n"
    "[RF]\n"
    "1 0 0 0 0 0 0 0 0 0 0 e\n"
    "[TRAP]\n"
    "1 100 100 800 100 0\n"
    "[ADC]\n"
    "1 3 1000 0 0 0 0 0 0\n";

// a trigger at t=0 (block 1), then a Z trapezoid in block 2 (area 0.09).
const char* kTrigSeq =
    "[VERSION]\n"
    "major 1\n"
    "minor 5\n"
    "revision 1\n"
    "[DEFINITIONS]\n"
    "GradientRasterTime 1e-05\n"
    "BlockDurationRaster 1e-05\n"
    "[BLOCKS]\n"
    " 1  20 0 0 0 0 0 1\n"
    " 2 100 0 0 0 1 0 0\n"
    "[TRAP]\n"
    "1 100 100 800 100 0\n"
    "[EXTENSIONS]\n"
    "1 1 1 0\n"
    "extension TRIGGERS 1\n"
    "1 1 3 0 10\n";

s2w::SequenceSource source(const char* text) {
    std::istringstream in(text);
    return s2w::SequenceSource(s2w::parse(in));
}

void adc_gradients_and_kspace_from_excitation() {
    const s2w::SequenceSource seq = source(kAdcSeq);
    // the ADC sits after the trap, so the gradient there is 0
    const std::vector<s2w::Trace> g = seq.adc_gradient_samples();
    check(g.size() == 1, "one ADC gradient trace");
    if (!g.empty()) {
        bool all_zero = true;
        for (const double v : g[0].z) {
            all_zero = all_zero && std::fabs(v) < 1e-9;
        }
        check(all_zero, "gradient is zero at the ADC, which is past the trap");
    }
    // k-space z = the full trap area (0.09 1/m), integrated from the excitation at t=0
    const std::vector<s2w::Trace> k = seq.adc_kspace();
    check(k.size() == 1, "one ADC k-space trace");
    if (!k.empty()) {
        check(k[0].t.size() == 3, "three ADC samples");
        bool kz_ok = !k[0].z.empty();
        for (const double v : k[0].z) {
            kz_ok = kz_ok && std::fabs(v - 0.09) < 1e-9;
        }
        check(kz_ok, "kz is the full trapezoid area");
        bool kx_zero = true;
        for (const double v : k[0].x) {
            kx_zero = kx_zero && std::fabs(v) < 1e-12;
        }
        check(kx_zero, "kx is zero -- no x gradient");
    }
}

void flat_kspace_matches_per_event() {
    const s2w::SequenceSource seq = source(kAdcSeq);
    const std::vector<s2w::Trace> per = seq.adc_kspace();
    const s2w::FlatTrace flat = seq.adc_kspace_flat();
    check(flat.offsets.size() == per.size() + 1, "offsets has events + 1 entries");
    check(!flat.offsets.empty() && flat.offsets.front() == 0, "offsets starts at 0");
    check(!flat.offsets.empty() && flat.offsets.back() == flat.t.size(), "offsets ends at len(t)");
    for (std::size_t e = 0; e < per.size() && e + 1 < flat.offsets.size(); ++e) {
        const std::size_t a = flat.offsets[e];
        const std::size_t b = flat.offsets[e + 1];
        check(b - a == per[e].t.size(), "event length matches");
        bool same = true;
        for (std::size_t i = 0; a + i < b && i < per[e].t.size(); ++i) {
            same = same && flat.t[a + i] == per[e].t[i] && flat.kz[a + i] == per[e].z[i];
        }
        check(same, "the flat path equals the per-event path exactly");
    }
}

void trigger_kspace_integrates_from_the_trigger() {
    const s2w::SequenceSource seq = source(kTrigSeq);
    // measure well after the trap; integrate from the trigger at t=0 -> the full area
    const std::vector<s2w::Trace> k = seq.trigger_kspace(1.5e-3, 10e-6, 2);
    check(k.size() == 1, "one trigger trace");
    if (!k.empty() && k[0].t.size() == 2) {
        close(k[0].t[0].secs(), 1.5e-3, "first sample is trigger + delay");
        close((k[0].t[1] - k[0].t[0]).secs(), 10e-6, "sample spacing is dt");
        bool kz_ok = true;
        for (const double v : k[0].z) {
            kz_ok = kz_ok && std::fabs(v - 0.09) < 1e-9;
        }
        check(kz_ok, "kz is the full area, integrated from the trigger");
    } else {
        check(false, "expected two trigger samples");
    }
}

void dense_gradient_covers_the_sequence() {
    const s2w::SequenceSource seq = source(kAdcSeq);
    close(seq.duration(), 2e-3, "duration is two 1 ms blocks", 1e-9);
    const s2w::Trace tr = seq.gradients_dense(1e-4);  // 0.1 ms raster
    check(tr.t.size() == tr.z.size(), "dense trace times and values agree in length");
    check(!tr.t.empty() && std::fabs(tr.t.front().secs()) < 1e-12, "dense grid starts at 0");
    check(!tr.t.empty() && tr.t.back().secs() >= 2e-3 - 1e-12, "dense grid covers the duration");
    // the Z trap plateau (amp 100) covers 0.1..0.9 ms in block 1
    const auto tv = seq.gradient_axis_dense('z', 1e-4);
    check(tv.first.size() == tv.second.size(), "axis-dense times and values agree in length");
    if (tv.second.size() > 5) {
        close(tv.second[5], 100.0, "g at 0.5 ms is the plateau amplitude", 1e-6);
    }
}

void arbitrary_sampling_via_the_sampler() {
    const s2w::SequenceSource seq = source(kAdcSeq);
    const s2w::Sampler s = seq.sampler();
    // the gradient at the flat top of the trap (0.1..0.9 ms) is the amplitude
    const s2w::Trace g = s.gradients({s2w::Ps::from_secs(0.5e-3)});
    check(!g.z.empty() && std::fabs(g.z[0] - 100.0) < 1e-9, "sampled plateau amplitude");
    // k from t=0 to 0.5 ms = the rise area + the flat from 0.1 to 0.5 ms
    //                      = 5e-3 + 100*0.4e-3 = 0.045
    const s2w::Trace k = s.kspace({s2w::Ps::from_secs(0.5e-3)}, s2w::Ps::zero());
    check(!k.z.empty() && std::fabs(k.z[0] - 0.045) < 1e-9, "partial k-space integral");
}

void unsorted_times_match_the_sorted_sweep() {
    // Sorted and unsorted queries must agree at the same times.
    const s2w::SequenceSource seq = source(kAdcSeq);
    const s2w::Sampler s = seq.sampler();
    std::vector<s2w::Ps> rising;
    for (int i = 0; i < 40; ++i) {
        rising.push_back(s2w::Ps::from_secs(static_cast<double>(i) * 50e-6));
    }
    std::vector<s2w::Ps> shuffled = rising;
    // a deterministic scramble: reverse, which is strictly decreasing, so the sweep is rejected
    std::reverse(shuffled.begin(), shuffled.end());

    const s2w::Trace a = s.gradients(rising);
    const s2w::Trace b = s.gradients(shuffled);
    bool same = true;
    for (std::size_t i = 0; i < rising.size(); ++i) {
        same = same && a.z[i] == b.z[rising.size() - 1 - i];
    }
    check(same, "sorted and unsorted gradient paths agree");

    const s2w::Trace ka = s.kspace(rising, s2w::Ps::zero());
    const s2w::Trace kb = s.kspace(shuffled, s2w::Ps::zero());
    same = true;
    for (std::size_t i = 0; i < rising.size(); ++i) {
        same = same && ka.z[i] == kb.z[rising.size() - 1 - i];
    }
    check(same, "sorted and unsorted k-space paths agree");
}

void adc_with_no_preceding_excitation() {
    // An ADC without preceding RF reports NaN metadata and integrates from the start.
    const char* no_rf =
        "[DEFINITIONS]\n"
        "GradientRasterTime 1e-05\n"
        "BlockDurationRaster 1e-05\n"
        "[BLOCKS]\n"
        " 1 100 0 0 0 1 1 0\n"
        "[TRAP]\n"
        "1 100 100 800 100 0\n"
        "[ADC]\n"
        "1 3 1000 0 0 0 0 0 0\n";
    const s2w::SequenceSource seq = source(no_rf);

    const auto exc = seq.adc_excitations();
    check(exc.size() == 1, "one ADC event");
    if (!exc.empty()) {
        check(std::isnan(std::get<0>(exc[0])) && std::isnan(std::get<1>(exc[0])) &&
                  std::isnan(std::get<2>(exc[0])),
              "an unreferenced ADC reports NaN rather than a plausible wrong number");
    }
    const auto specs = seq.adc_specs();
    check(specs.size() == 1, "one ADC spec");
    if (!specs.empty()) {
        check(specs[0].reference.raw() == std::numeric_limits<std::int64_t>::min(),
              "the reference is the far-past sentinel");
    }
    // The sentinel is before every corner, where moment_to is 0, so k-space is the plain integral
    // from the start -- here the whole trapezoid area by the time the ADC samples.
    const std::vector<s2w::Trace> k = seq.adc_kspace();
    check(k.size() == 1 && !k[0].z.empty(), "one k-space trace with samples");
    if (!k.empty() && !k[0].z.empty()) {
        check(std::isfinite(k[0].z[0]), "k-space is finite despite the sentinel reference");
    }
}

void uniform_grid_edges() {
    // Degenerate uniform grids return the zero instant.
    check(s2w::uniform_grid(0.0, 1e-5).size() == 1, "zero duration gives one point");
    check(s2w::uniform_grid(1e-3, 0.0).size() == 1, "zero raster gives one point");
    check(s2w::uniform_grid(1e-3, -1.0).size() == 1, "negative raster gives one point");
    const std::vector<s2w::Ps> g = s2w::uniform_grid(1e-3, 1e-4);
    check(g.size() == 11, "an exact multiple still covers the end");
    check(!g.empty() && g.back().secs() >= 1e-3 - 1e-12, "the last point reaches the duration");
}

}  // namespace

int main() {
    adc_gradients_and_kspace_from_excitation();
    flat_kspace_matches_per_event();
    trigger_kspace_integrates_from_the_trigger();
    dense_gradient_covers_the_sequence();
    arbitrary_sampling_via_the_sampler();
    unsorted_times_match_the_sorted_sweep();
    adc_with_no_preceding_excitation();
    uniform_grid_edges();

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
