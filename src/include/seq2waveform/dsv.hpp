// Reader for Siemens DSP-Simulator `.dsv` waveform files.
//
// Layout (per channel):
//
//     [DEFINITIONS]
//     HORIDELTA=10.000000      <- sample spacing, microseconds
//     VERTFACTOR=1000000.0     <- divide the cumulative sum by this -> physical units
//     VERTUNITNAME=mT/m
//     [VALUES]
//     0                        <- integer DELTAS, run-length encoded
//     ...
//
// The `[VALUES]` body is the waveform's derivative, run-length encoded (`d, d, k` = "`d` repeated
// `k+2` times"). It is decoded straight to piecewise-linear corners -- a run of equal deltas is a
// constant-slope segment, so only slope changes are kept (plus the first and last sample), and the
// dense array is never materialized. `DsvChannel::dense` reconstructs per-sample values on demand.
//
// Ported from the Rust `dsv.rs`. `DsvSet` (the whole measurement set as a WaveformSource) needs
// the INF reader and arrives at port step 7.

#pragma once

#include <cstddef>
#include <istream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "seq2waveform/pwl.hpp"

namespace s2w {

/// A single decoded DSV channel.
struct DsvChannel {
    /// Raw key/value pairs from the `[DEFINITIONS]` header.
    std::map<std::string, std::string> definitions;
    /// Sample spacing in seconds (from `HORIDELTA`).
    double raster_time = 0.0;
    /// Physical unit name from `VERTUNITNAME`, e.g. "mT/m".
    std::string unit;
    /// The decoded waveform as piecewise-linear corners. Sample times are interval-centered at
    /// `(k+0.5)*raster_time`; values are in physical units, with `VERTFACTOR` applied.
    PiecewiseLinear waveform;
    /// The original (dense) sample count.
    std::size_t num_samples = 0;

    std::size_t size() const { return num_samples; }
    bool empty() const { return num_samples == 0; }

    /// Total duration in seconds (`num_samples * raster_time`).
    double duration() const { return static_cast<double>(num_samples) * raster_time; }

    /// Reconstruct the dense per-sample values (the original DSV samples).
    ///
    /// Exact: sampling the piecewise-linear waveform at `(k+0.5)*raster` reproduces the original
    /// `cumsum(deltas)/VERTFACTOR`, because linear interpolation of a line is the line.
    std::vector<double> dense() const;
};

/// Decode a `.dsv` channel from text already in memory. Throws `DsvError` on a malformed value
/// line or a missing `HORIDELTA`/`VERTFACTOR`.
DsvChannel read_dsv_text(std::string_view text);

/// Read and decode a `.dsv` channel from any stream: slurps it, then calls `read_dsv_text`.
DsvChannel read_dsv_channel(std::istream& in);

/// Read and decode a `.dsv` file from disk.
DsvChannel read_dsv_file(const std::string& path);

}  // namespace s2w
