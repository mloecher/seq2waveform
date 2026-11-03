// Waveform reconstruction helpers shared with the core tests.

#pragma once

#include <utility>
#include <vector>

#include "seq2waveform/pwl.hpp"

namespace s2w {
namespace detail {

/// Reconstruct raster-edge corner points from a centers-sampled arbitrary gradient.
/// Port of `restoreAdditionalShapeSamples.m`. Returns `(times_s, values)`, times relative to the
/// start of the shape. See waveform.cpp for the algorithm.
std::pair<std::vector<double>, std::vector<double>> restore_additional_shape_samples(
    const std::vector<double>& tt, const std::vector<double>& waveform, double first, double last,
    double grad_raster);

/// Resample a rendered gradient channel onto the raster centers and keep only the corners -- the
/// Siemens DSV / commanded waveform.
PiecewiseLinear resample_at_centers(const PiecewiseLinear& pwl, double raster);

}  // namespace detail
}  // namespace s2w
