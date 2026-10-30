// Waveform expansion: turn a parsed Sequence into explicit waveforms (Pulseq's
// `waveforms_and_times`).
//
// Walk the blocks accumulating time, resolve each block's events, and emit gradient corners, RF
// pulses, ADC times and triggers. The gradient path covers trapezoid corners,
// `restoreAdditionalShapeSamples` for centers-raster arbitrary shapes, and the cross-block stitch
// (a forced ramp to and from zero across gaps).
//
// Divergence from Pulseq: RF keeps its real `use` -- undefined pulses are NOT folded into
// excitation. Pulseq's own h5 routes both into /excitation, so an h5 comparison has to union them.
//
// Ported from the Rust `waveform.rs`. Rust puts these on `Sequence` with an `impl` block in this
// module; C++ cannot, so they are free functions taking `const Sequence&` and `model.hpp` stays
// free of waveform concerns.

#pragma once

#include <cstdint>
#include <vector>

#include "seq2waveform/model.hpp"
#include "seq2waveform/pwl.hpp"
#include "seq2waveform/time.hpp"

namespace s2w {

/// A dense complex RF pulse (one per RF event). Times absolute (ps), B1 in Hz.
struct RfPulse {
    std::vector<Ps> t;
    std::vector<double> re;
    std::vector<double> im;
    /// Effective excitation time: block start + delay + center.
    Ps center;
    double freq = 0.0;
    double phase = 0.0;
    RfUse use = RfUse::Undefined;
};

/// RF event metadata without the complex waveform -- Rust's `(Ps, f64, f64, RfUse)` tuple.
struct RfMeta {
    Ps center;
    double freq = 0.0;
    double phase = 0.0;
    RfUse use = RfUse::Undefined;
};

/// One ADC event: its sample times plus per-event freq/phase offsets.
struct AdcEvent {
    std::vector<Ps> t;
    double freq = 0.0;
    double phase = 0.0;
};

/// A resolved trigger event (absolute time).
struct TriggerEvent {
    std::uint32_t channel = 0;
    Ps time;
    Ps duration;
    std::uint32_t trig_type = 0;
};

/// Expanded waveforms for a whole sequence. Gradients are piecewise-linear (Hz/m).
struct Waveforms {
    PiecewiseLinear gx;
    PiecewiseLinear gy;
    PiecewiseLinear gz;
    std::vector<RfPulse> rf;
    std::vector<AdcEvent> adc;
    std::vector<TriggerEvent> triggers;
};

/// Expand the sequence into explicit waveforms.
Waveforms waveforms(const Sequence& seq);

// --- focused single-output walks (skip building everything else) ---------

/// One gradient channel (Hz/m piecewise-linear). `axis` is 'x', 'y' or 'z'; anything else gives an
/// empty channel, as in Rust.
PiecewiseLinear gradient(const Sequence& seq, char axis);

/// ADC sample times only, flattened across events. Avoids building RF and gradients.
std::vector<Ps> adc_times(const Sequence& seq);

/// ADC events split as in the `.seq`: one entry per ADC event rather than one flat time vector.
std::vector<AdcEvent> adc_events(const Sequence& seq);

/// RF event metadata only -- skips the complex waveform.
std::vector<RfMeta> rf_events_meta(const Sequence& seq);

/// Resolved trigger events only.
std::vector<TriggerEvent> trigger_events(const Sequence& seq);

}  // namespace s2w
