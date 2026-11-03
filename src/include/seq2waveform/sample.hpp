// Sample gradients (Hz/m) or their time integral, k-space (1/m).
// Sampler holds the waveforms and cumulative moments; SampleSpec supplies query times
// and an integration reference. Gamma is already included in the gradient values.
// SequenceSource and DsvSet share the WaveformSource sampling helpers.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "seq2waveform/model.hpp"
#include "seq2waveform/pwl.hpp"
#include "seq2waveform/time.hpp"
#include "seq2waveform/waveform.hpp"

namespace s2w {

/// Where to sample, and the reference time for k-space integration.
struct SampleSpec {
    std::vector<Ps> times;
    /// k-space integration start (an excitation center, or a trigger time). Ignored when sampling
    /// gradient values.
    Ps reference;
};

/// A sampled trace: times and per-axis values (gradient Hz/m, or k-space 1/m).
struct Trace {
    std::vector<Ps> t;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
};

/// Flat batched k-space: the three concatenated axis arrays.
struct FlatK {
    std::vector<double> kx;
    std::vector<double> ky;
    std::vector<double> kz;
};

/// Flat batched k-space with its times and event boundaries.
struct FlatTrace {
    std::vector<Ps> t;
    std::vector<double> kx;
    std::vector<double> ky;
    std::vector<double> kz;
    /// Cumulative per-event boundaries, length events + 1: `offsets[0] == 0`,
    /// `offsets[events] == t.size()`, and event `e` is `[offsets[e], offsets[e+1])`.
    std::vector<std::size_t> offsets;
};

/// Precomputed gradient channels and their cumulative moments, for fast repeated sampling.
class Sampler {
public:
    Sampler(std::array<PiecewiseLinear, 3> g, std::array<std::vector<double>, 3> cum)
        : g_(std::move(g)), cum_(std::move(cum)) {}

    /// Gradient values (Hz/m) at `times`, linearly interpolated between corners.
    ///
    /// When `times` is non-decreasing -- the usual case, ADC readouts and trigger rasters -- this
    /// is a single forward sweep per axis; otherwise it falls back to a binary search per point.
    Trace gradients(const std::vector<Ps>& times) const;

    /// k-space (1/m) at `times`, integrating each gradient from `reference`.
    Trace kspace(const std::vector<Ps>& times, Ps reference) const;

    /// Batched k-space over many events in one call, returning flat concatenated axes.
    ///
    /// `offsets` are cumulative event boundaries into `times` (length `references.size() + 1`);
    /// event `e` is `times[offsets[e] .. offsets[e+1])` integrated from `references[e]`. Reuses one
    /// prebuilt sampler and produces three big arrays instead of a per-event list, which is the
    /// fast path for whole-sequence k-space and for delay sweeps.
    FlatK kspace_flat(const std::vector<Ps>& times, const std::vector<std::size_t>& offsets,
                      const std::vector<Ps>& references) const;

private:
    std::array<PiecewiseLinear, 3> g_;
    std::array<std::vector<double>, 3> cum_;
};

/// A source of gradient waveforms and measurement events for the Sampler pipeline.
///
/// Subclasses supply the five primitives (gradients in Hz/m, in their own time frame); every
/// k-space and sampling helper below is shared, so the same pipeline runs on a `.seq` or on a
/// DSV+INF set.
class WaveformSource {
public:
    virtual ~WaveformSource() = default;

    // --- primitives ------------------------------------------------------

    /// Gradient channel for `axis` ('x', 'y' or 'z') as piecewise-linear corners, Hz/m.
    virtual PiecewiseLinear gradient(char axis) const = 0;
    /// ADC events, one per acquisition, each with its sample times (plus freq/phase).
    virtual std::vector<AdcEvent> adc_events() const = 0;
    /// Excitations as `(center, freq_hz, phase_rad)`, sorted by center ascending.
    virtual std::vector<std::tuple<Ps, double, double>> excitations() const = 0;
    /// Trigger / external-sync events.
    virtual std::vector<TriggerEvent> trigger_events() const = 0;
    /// Total time span (s): a Sequence's summed block durations, or a DsvSet's channel duration
    /// (Siemens frame, which includes the ~1020 us lead-in).
    virtual double duration() const = 0;

    /// Event starts for excitation matching; overrides avoid building every ADC sample.
    virtual std::vector<Ps> adc_event_starts() const;

    // --- shared helpers --------------------------------------------------

    /// Build a Sampler, precomputing gx/gy/gz and their cumulative moments.
    Sampler sampler() const;

    /// Excitation center times, sorted ascending -- the k-space references for ADC sampling.
    std::vector<Ps> excitation_centers() const;

    /// Per ADC event, the matched excitation `(center_s, freq_hz, phase_rad)`: the most recent
    /// excitation before the event, or `(nan, nan, nan)` if there is none. Aligned 1:1 with
    /// `adc_kspace` and `adc_specs`, so it zips directly against them.
    std::vector<std::tuple<double, double, double>> adc_excitations() const;

    /// ADC sample grids, referenced to the preceding excitation. Warns on count mismatches.
    std::vector<SampleSpec> adc_specs() const;

    /// Sample specs for trigger-based (Skope) measurements: per trigger, `n` samples at `dt` after
    /// `delay`. Reference = the trigger time itself.
    std::vector<SampleSpec> trigger_specs(double delay, double dt, std::size_t n) const;

    std::vector<Trace> adc_gradient_samples() const;
    std::vector<Trace> adc_kspace() const;
    std::vector<Trace> trigger_gradient_samples(double delay, double dt, std::size_t n) const;
    std::vector<Trace> trigger_kspace(double delay, double dt, std::size_t n) const;

    /// Whole-sequence ADC k-space as flat arrays. One big array per axis instead of a per-event
    /// list: the fast path for large sequences.
    FlatTrace adc_kspace_flat() const;

    /// Trigger-based k-space as flat arrays. See `adc_kspace_flat`.
    FlatTrace trigger_kspace_flat(double delay, double dt, std::size_t n) const;

    /// All three gradient axes on a uniform `raster` over `[0, duration]`.
    Trace gradients_dense(double raster) const;

    /// One gradient axis on a uniform `raster` over `[0, duration]`.
    std::pair<std::vector<Ps>, std::vector<double>> gradient_axis_dense(char axis,
                                                                       double raster) const;
};

/// Adapt parsed sequence data to the shared sampling interface.
class SequenceSource : public WaveformSource {
public:
    explicit SequenceSource(Sequence seq) : seq_(std::move(seq)) {}

    PiecewiseLinear gradient(char axis) const override;
    std::vector<AdcEvent> adc_events() const override;
    /// RF pulses with `use == Excitation` (undefined is NOT folded in) as `(center, freq, phase)`,
    /// sorted by center.
    std::vector<std::tuple<Ps, double, double>> excitations() const override;
    std::vector<TriggerEvent> trigger_events() const override;
    double duration() const override;
    std::vector<Ps> adc_event_starts() const override;

    Sequence& sequence() { return seq_; }
    const Sequence& sequence() const { return seq_; }

private:
    Sequence seq_;
};

/// Uniform sample grid `0, raster, 2*raster, ...` covering `[0, duration]` (the last point is at
/// or past `duration`). Exposed for the tests.
std::vector<Ps> uniform_grid(double duration, double raster);

/// Concatenate specs into flat times, cumulative offsets (length events + 1) and per-event
/// references -- the input shape for `Sampler::kspace_flat`. Exposed for the tests.
std::tuple<std::vector<Ps>, std::vector<std::size_t>, std::vector<Ps>> flatten_specs(
    const std::vector<SampleSpec>& specs);

}  // namespace s2w
