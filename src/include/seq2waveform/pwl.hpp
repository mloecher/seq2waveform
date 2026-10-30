// Piecewise-linear waveform: value is linear between (t, v) corners.
//
// The canonical gradient form -- what Pulseq emits (trapezoid = 4 corners) and what the DSV
// collapses to. Sparse for trapezoids, exact to integrate (k-space is its running integral).
// Corner times are Ps (integer picoseconds); values double.

#pragma once

#include <cstddef>
#include <vector>

#include "seq2waveform/time.hpp"

namespace s2w {

/// A piecewise-linear waveform. `value_at` interpolates linearly between corners and clamps to
/// the endpoints outside [t.front(), t.back()].
struct PiecewiseLinear {
    /// Strictly increasing corner times (picoseconds).
    std::vector<Ps> t;
    /// Corner values (canonical units; Hz/m for gradients).
    std::vector<double> v;

    std::size_t size() const { return t.size(); }
    bool empty() const { return t.empty(); }

    /// Linearly interpolated value at `t`; clamps to the endpoints outside the corner range.
    double value_at(Ps at) const;

    /// Sample the waveform at the given times (binary search per point).
    std::vector<double> sample_at(const std::vector<Ps>& times) const;

    /// `value_at` over non-decreasing `times`, via one forward cursor over the corners instead of
    /// a binary search per point -- O(corners + times). Hot path for ADC readouts.
    std::vector<double> sample_at_sorted(const std::vector<Ps>& times) const;

    /// As above over a contiguous range, so a batched caller can work on a slice of a larger times
    /// array without copying it (`Sampler::kspace_flat` does, per event). The vector overload
    /// delegates here.
    std::vector<double> sample_at_sorted(const Ps* times, std::size_t count) const;

    /// As above, writing `count` values into `out` rather than returning a vector. `out` must have
    /// room for `count`. The counterpart of `moments_at_sorted_into`: it lets a caller sweep a long
    /// grid in fixed-size chunks without materialising either the times or the values twice
    /// (`DsvChannel::dense` does, over tens of millions of samples).
    void sample_at_sorted_into(const Ps* times, std::size_t count, double* out) const;

    /// Exact integral of v*dt from `t0` to `t1` -- sum of trapezoids over the linear segments.
    /// The k-space building block: k(t) = integral from the excitation to t.
    double integral(Ps t0, Ps t1) const;

    /// Running integral at each corner: cum[i] = integral from t[0] to t[i]. Precompute once,
    /// then `moment_to` evaluates any time in O(1) -- the k-space fast path.
    std::vector<double> cumulative_moment() const;

    /// Integral from t[0] to `at` using precomputed corner moments `cum`. Zero before the corner
    /// range, saturates after it. A k-space value referenced to t_ref is
    /// moment_to(t) - moment_to(t_ref).
    double moment_to(Ps at, const std::vector<double>& cum) const;

    /// `moment_to` over non-decreasing `times`, via one forward cursor -- O(corners + times).
    /// The hot path for k-space over an ADC readout.
    std::vector<double> moments_at_sorted(const std::vector<Ps>& times,
                                          const std::vector<double>& cum) const;

    /// As above over a contiguous range; see `sample_at_sorted(const Ps*, std::size_t)`.
    std::vector<double> moments_at_sorted(const Ps* times, std::size_t count,
                                          const std::vector<double>& cum) const;

    /// As above, writing `count` values into `out` rather than returning a vector. `out` must have
    /// room for `count`. This is what the batched k-space path uses: it fills its slice of the
    /// result in place, so a whole-run sampling neither allocates per event nor copies the values
    /// a second time.
    void moments_at_sorted_into(const Ps* times, std::size_t count,
                                const std::vector<double>& cum, double* out) const;
};

}  // namespace s2w
