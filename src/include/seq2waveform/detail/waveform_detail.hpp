// Internals of waveform.cpp, exposed only so the tests can reach them.
//
// Rust's unit tests live in the same module as the private functions they exercise and need no
// equivalent of this header; C++ has no such arrangement, and these two steps are where a port
// drifts, so testing them directly rather than through `gradient()` is worth a header. Nothing
// outside waveform.cpp and tests/test_waveform.cpp should include it.

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
