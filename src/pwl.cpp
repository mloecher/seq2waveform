#include "seq2waveform/pwl.hpp"

#include <algorithm>

namespace s2w {
namespace {

/// Index of the first corner strictly past `at` -- the C++ spelling of Rust's
/// `partition_point(|x| x <= at)`. Keeping these identical is what makes the ports agree.
std::size_t first_past(const std::vector<Ps>& t, Ps at) {
    return static_cast<std::size_t>(std::upper_bound(t.begin(), t.end(), at) - t.begin());
}

/// Lerp within the segment [j-1, j], guarding the zero-width (duplicate corner) case.
double lerp_in_segment(const PiecewiseLinear& w, std::size_t j, Ps at) {
    const Ps t0 = w.t[j - 1];
    const Ps t1 = w.t[j];
    const double v0 = w.v[j - 1];
    const double v1 = w.v[j];
    if (t1 == t0) {
        return v0;
    }
    return v0 + (v1 - v0) * at.frac_between(t0, t1);
}

}  // namespace

double PiecewiseLinear::value_at(Ps at) const {
    const std::size_t n = t.size();
    if (n == 0) {
        return 0.0;
    }
    if (at <= t[0]) {
        return v[0];
    }
    if (at >= t[n - 1]) {
        return v[n - 1];
    }
    return lerp_in_segment(*this, first_past(t, at), at);
}

std::vector<double> PiecewiseLinear::sample_at(const std::vector<Ps>& times) const {
    std::vector<double> out;
    out.reserve(times.size());
    for (const Ps at : times) {
        out.push_back(value_at(at));
    }
    return out;
}

std::vector<double> PiecewiseLinear::sample_at_sorted(const std::vector<Ps>& times) const {
    return sample_at_sorted(times.data(), times.size());
}

std::vector<double> PiecewiseLinear::sample_at_sorted(const Ps* times, std::size_t count) const {
    const std::size_t n = t.size();
    if (n == 0 || count == 0) {
        return std::vector<double>(count, 0.0);
    }
    std::vector<double> out;
    out.reserve(count);
    // Cursor: first corner past times[0], never below 1 so [j-1, j] is always a real segment.
    std::size_t j = std::max<std::size_t>(first_past(t, times[0]), 1);
    for (std::size_t i = 0; i < count; ++i) {
        const Ps at = times[i];
        if (at <= t[0]) {
            out.push_back(v[0]);
            continue;
        }
        if (at >= t[n - 1]) {
            out.push_back(v[n - 1]);
            continue;
        }
        while (j < n && t[j] <= at) {
            ++j;
        }
        out.push_back(lerp_in_segment(*this, j, at));
    }
    return out;
}

double PiecewiseLinear::integral(Ps t0, Ps t1) const {
    if (t1 < t0) {
        return -integral(t1, t0);
    }
    const std::size_t n = t.size();
    if (n == 0 || t1 == t0) {
        return 0.0;
    }
    // Walk corner to corner adding each trapezoid area, with (a, va) the running left edge.
    double area = 0.0;
    Ps a = t0;
    double va = value_at(t0);
    for (std::size_t i = 0; i < n; ++i) {
        const Ps ti = t[i];
        if (ti <= t0) {
            continue;
        }
        if (ti >= t1) {
            break;
        }
        const double vi = v[i];
        area += 0.5 * (va + vi) * (ti - a).secs();
        a = ti;
        va = vi;
    }
    const double vb = value_at(t1);
    area += 0.5 * (va + vb) * (t1 - a).secs();
    return area;
}

std::vector<double> PiecewiseLinear::cumulative_moment() const {
    const std::size_t n = t.size();
    std::vector<double> cum(n, 0.0);
    for (std::size_t i = 1; i < n; ++i) {
        cum[i] = cum[i - 1] + 0.5 * (v[i - 1] + v[i]) * (t[i] - t[i - 1]).secs();
    }
    return cum;
}

double PiecewiseLinear::moment_to(Ps at, const std::vector<double>& cum) const {
    const std::size_t n = t.size();
    if (n == 0 || at <= t[0]) {
        return 0.0;
    }
    if (at >= t[n - 1]) {
        return cum[n - 1];
    }
    // Full moments up to the segment start, plus the partial trapezoid [t0, at].
    const std::size_t j = first_past(t, at);
    const double vt = lerp_in_segment(*this, j, at);
    return cum[j - 1] + 0.5 * (v[j - 1] + vt) * (at - t[j - 1]).secs();
}

std::vector<double> PiecewiseLinear::moments_at_sorted(const std::vector<Ps>& times,
                                                       const std::vector<double>& cum) const {
    return moments_at_sorted(times.data(), times.size(), cum);
}

std::vector<double> PiecewiseLinear::moments_at_sorted(const Ps* times, std::size_t count,
                                                       const std::vector<double>& cum) const {
    std::vector<double> out(count, 0.0);
    moments_at_sorted_into(times, count, cum, out.data());
    return out;
}

void PiecewiseLinear::moments_at_sorted_into(const Ps* times, std::size_t count,
                                             const std::vector<double>& cum, double* out) const {
    const std::size_t n = t.size();
    if (n == 0 || count == 0) {
        std::fill(out, out + count, 0.0);
        return;
    }
    std::size_t j = std::max<std::size_t>(first_past(t, times[0]), 1);
    for (std::size_t i = 0; i < count; ++i) {
        const Ps at = times[i];
        if (at <= t[0]) {
            out[i] = 0.0;
            continue;
        }
        if (at >= t[n - 1]) {
            out[i] = cum[n - 1];
            continue;
        }
        while (j < n && t[j] <= at) {
            ++j;
        }
        const double vt = lerp_in_segment(*this, j, at);
        out[i] = cum[j - 1] + 0.5 * (v[j - 1] + vt) * (at - t[j - 1]).secs();
    }
}

}  // namespace s2w
