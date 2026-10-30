// Internals of dsv.cpp, exposed only so the tests can reach them.
//
// `rle_to_piecewise` is the whole of the DSV decode, and the Rust unit test calls it directly with
// a hand-written delta stream -- far clearer than driving it through a synthetic file. Nothing
// outside dsv.cpp and tests/test_dsv.cpp should include this.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "seq2waveform/pwl.hpp"

namespace s2w {
namespace detail {

/// Decode the RLE delta stream directly into piecewise-linear corners, returning
/// `(waveform, num_samples)`.
///
/// A sample is a corner exactly when the slope changes there, i.e. `d[k] != d[k+1]`. The first
/// sample, the last sample and every interior slope change are emitted, streaming the RLE so the
/// full dense array is never held.
std::pair<PiecewiseLinear, std::size_t> rle_to_piecewise(const std::vector<std::int64_t>& rle,
                                                         double raster, double vertfactor);

}  // namespace detail
}  // namespace s2w
