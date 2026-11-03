// Expand sequence blocks into gradient corners, RF pulses, ADC times and triggers.
// RF use is preserved: undefined pulses are not relabelled as excitations.

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

/// RF metadata without the complex waveform.
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

/// One gradient channel in Hz/m. Axes are 'x', 'y', 'z'; other values return an empty channel.
PiecewiseLinear gradient(const Sequence& seq, char axis);

/// ADC sample times only, flattened across events. Avoids building RF and gradients.
std::vector<Ps> adc_times(const Sequence& seq);

/// ADC events split as in the `.seq`: one entry per ADC event rather than one flat time vector.
std::vector<AdcEvent> adc_events(const Sequence& seq);

/// The first sample time of each ADC event, without building the samples. Matches
/// `adc_events()[i].t.front()` exactly (and `Ps::zero()` for an empty event) at a fraction of the
/// cost -- one number per event rather than one per sample.
std::vector<Ps> adc_event_starts(const Sequence& seq);

/// RF event metadata only -- skips the complex waveform.
std::vector<RfMeta> rf_events_meta(const Sequence& seq);

/// Resolved trigger events only.
std::vector<TriggerEvent> trigger_events(const Sequence& seq);

}  // namespace s2w
