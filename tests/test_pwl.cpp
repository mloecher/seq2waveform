// Tests for picosecond arithmetic, interpolation and integration.

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include "seq2waveform/pwl.hpp"
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

s2w::Ps s(double x) { return s2w::Ps::from_secs(x); }

s2w::PiecewiseLinear pwl(const std::vector<double>& t, std::vector<double> v) {
    s2w::PiecewiseLinear w;
    w.t.reserve(t.size());
    for (const double x : t) {
        w.t.push_back(s(x));
    }
    w.v = std::move(v);
    return w;
}

void ps_roundtrip_and_exactness() {
    check(s2w::Ps::from_secs(10e-6) == s2w::Ps(10'000'000), "10 us is exact in ps");
    check(s2w::Ps::from_secs(1e-9) == s2w::Ps(1000), "1 ns is exact in ps");
    check(s2w::Ps::from_secs(0.5e-9) == s2w::Ps(500), "half-dwell 0.5 ns is exact in ps");
    close(s2w::Ps(10'000'000).secs(), 10e-6, "ps -> seconds roundtrip", 1e-18);
}

void ps_difference_is_exact_at_large_offset() {
    // 24 s + 0.3 us, minus 24 s -> exactly 0.3 us, regardless of the huge absolute value.
    const s2w::Ps base = s2w::Ps::from_secs(24.0);
    const s2w::Ps a = base + s2w::Ps::from_secs(0.3e-6);
    check((a - base) == s2w::Ps(300'000), "difference exact at 24 s offset");
    close((a - base).secs(), 0.3e-6, "difference in seconds exact at 24 s offset", 1e-21);
}

void interpolates_and_clamps() {
    const auto w = pwl({0.0, 1.0, 2.0}, {0.0, 2.0, 2.0});
    close(w.value_at(s(-1.0)), 0.0, "clamp left");
    close(w.value_at(s(0.5)), 1.0, "lerp on the ramp");
    close(w.value_at(s(1.0)), 2.0, "exact corner");
    close(w.value_at(s(1.5)), 2.0, "flat");
    close(w.value_at(s(3.0)), 2.0, "clamp right");
}

void integrates_exactly() {
    // A ramp 0->2 over [0,1], then flat 2 over [1,2].
    const auto w = pwl({0.0, 1.0, 2.0}, {0.0, 2.0, 2.0});
    close(w.integral(s(0.0), s(1.0)), 1.0, "triangle area");
    close(w.integral(s(1.0), s(2.0)), 2.0, "rectangle area");
    close(w.integral(s(0.0), s(2.0)), 3.0, "both segments");
    close(w.integral(s(0.5), s(1.0)), 0.75, "partial segment");
    close(w.integral(s(1.0), s(0.0)), -1.0, "reversed bounds negate");
}

void cumulative_moment_evaluates_fast() {
    // Constant 2 over [0,3]: moment(t) = 2t, saturating at 6 after the end.
    const auto w = pwl({0.0, 3.0}, {2.0, 2.0});
    const auto cum = w.cumulative_moment();
    close(w.moment_to(s(0.0), cum), 0.0, "moment at the first corner");
    close(w.moment_to(s(1.5), cum), 3.0, "moment mid-waveform");
    close(w.moment_to(s(3.0), cum), 6.0, "moment at the last corner");
    close(w.moment_to(s(5.0), cum), 6.0, "moment saturates past the end");
    close(w.moment_to(s(2.0), cum), w.integral(s(0.0), s(2.0)), "moment matches integral");

    // Ramp 0->2 over [0,1]: moment(t) = t^2.
    const auto r = pwl({0.0, 1.0}, {0.0, 2.0});
    const auto rc = r.cumulative_moment();
    close(r.moment_to(s(0.5), rc), 0.25, "quadratic moment on a ramp");
}

void sorted_sweeps_match_per_point() {
    // Segments including a zero-width corner, queried before the start, on corners, between,
    // and past the end.
    const auto w = pwl({0.0, 1.0, 1.0, 2.0, 4.0}, {0.0, 2.0, -1.0, -1.0, 3.0});
    const auto cum = w.cumulative_moment();
    std::vector<s2w::Ps> times;
    for (const double x : {-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}) {
        times.push_back(s(x));
    }

    const auto swept = w.sample_at_sorted(times);
    const auto per_point = w.sample_at(times);
    for (std::size_t i = 0; i < times.size(); ++i) {
        close(swept[i], per_point[i], "sorted sweep matches per-point value");
    }

    const auto moments = w.moments_at_sorted(times, cum);
    for (std::size_t i = 0; i < times.size(); ++i) {
        close(moments[i], w.moment_to(times[i], cum), "sorted sweep matches per-point moment");
    }
}

void handles_empty_waveform() {
    const s2w::PiecewiseLinear empty;
    close(empty.value_at(s(1.0)), 0.0, "empty waveform samples as zero");
    close(empty.integral(s(0.0), s(1.0)), 0.0, "empty waveform integrates to zero");
    const std::vector<s2w::Ps> times{s(0.0), s(1.0)};
    check(empty.sample_at_sorted(times).size() == 2, "empty waveform returns one value per time");
    check(empty.cumulative_moment().empty(), "empty waveform has no corner moments");
}

}  // namespace

int main() {
    ps_roundtrip_and_exactness();
    ps_difference_is_exact_at_large_offset();
    interpolates_and_clamps();
    integrates_exactly();
    cumulative_moment_evaluates_fast();
    sorted_sweeps_match_per_point();
    handles_empty_waveform();

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
