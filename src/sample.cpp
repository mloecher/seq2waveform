#include "seq2waveform/sample.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

namespace s2w {
namespace {

/// Whether `times` is non-decreasing -- a cheap single pass that selects the fast sweep path.
/// `std::is_sorted`'s default predicate is `<`, so it accepts equal neighbours, matching Rust's
/// `windows(2).all(|w| w[0] <= w[1])` (and its `true` for fewer than two elements).
bool is_sorted_asc(const Ps* times, std::size_t count) {
    return std::is_sorted(times, times + count);
}

/// Most recent excitation at or before `t`, or a sentinel far before the start -- which makes
/// k-space integrate from the beginning of the waveform, since `moment_to` returns 0 there.
Ps excite_before(const std::vector<Ps>& excites, Ps t) {
    const std::size_t i = static_cast<std::size_t>(
        std::upper_bound(excites.begin(), excites.end(), t) - excites.begin());
    return i == 0 ? Ps(std::numeric_limits<std::int64_t>::min()) : excites[i - 1];
}

}  // namespace

// --- Sampler -------------------------------------------------------------

Trace Sampler::gradients(const std::vector<Ps>& times) const {
    const bool sorted = is_sorted_asc(times.data(), times.size());
    const auto sample = [&](const PiecewiseLinear& g) {
        return sorted ? g.sample_at_sorted(times) : g.sample_at(times);
    };
    Trace tr;
    tr.x = sample(g_[0]);
    tr.y = sample(g_[1]);
    tr.z = sample(g_[2]);
    tr.t = times;
    return tr;
}

Trace Sampler::kspace(const std::vector<Ps>& times, Ps reference) const {
    const bool sorted = is_sorted_asc(times.data(), times.size());
    const auto axis = [&](std::size_t i) {
        // The reference moment is subtracted from every point, which is what makes k-space
        // relative to the excitation (or trigger) rather than to the start of the waveform.
        const double m0 = g_[i].moment_to(reference, cum_[i]);
        std::vector<double> col;
        if (sorted) {
            col = g_[i].moments_at_sorted(times, cum_[i]);
        } else {
            col.reserve(times.size());
            for (const Ps t : times) {
                col.push_back(g_[i].moment_to(t, cum_[i]));
            }
        }
        for (double& x : col) {
            x -= m0;
        }
        return col;
    };
    Trace tr;
    tr.t = times;
    tr.x = axis(0);
    tr.y = axis(1);
    tr.z = axis(2);
    return tr;
}

FlatK Sampler::kspace_flat(const std::vector<Ps>& times, const std::vector<std::size_t>& offsets,
                           const std::vector<Ps>& references) const {
    const std::size_t n = times.size();
    std::array<std::vector<double>, 3> out = {std::vector<double>(n, 0.0),
                                              std::vector<double>(n, 0.0),
                                              std::vector<double>(n, 0.0)};
    for (std::size_t e = 0; e < references.size(); ++e) {
        const std::size_t a = offsets[e];
        const std::size_t b = offsets[e + 1];
        const std::size_t len = b - a;
        // The segment is addressed in place; the pointer-range overloads exist so a batched run
        // does not copy every event's times out of the flat array.
        const Ps* seg = times.data() + a;
        const bool sorted = is_sorted_asc(seg, len);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double m0 = g_[axis].moment_to(references[e], cum_[axis]);
            // Straight into this event's slice of the result. Going via a per-event temporary
            // instead cost one 40 KB allocation per event per axis and a second pass to copy the
            // values out -- on `cones`, 12,648 allocations and 64 M redundant element copies.
            double* dst = out[axis].data() + a;
            if (sorted) {
                g_[axis].moments_at_sorted_into(seg, len, cum_[axis], dst);
            } else {
                for (std::size_t i = 0; i < len; ++i) {
                    dst[i] = g_[axis].moment_to(seg[i], cum_[axis]);
                }
            }
            for (std::size_t i = 0; i < len; ++i) {
                dst[i] -= m0;
            }
        }
    }
    FlatK k;
    k.kx = std::move(out[0]);
    k.ky = std::move(out[1]);
    k.kz = std::move(out[2]);
    return k;
}

// --- WaveformSource shared helpers --------------------------------------

Sampler WaveformSource::sampler() const {
    std::array<PiecewiseLinear, 3> g = {gradient('x'), gradient('y'), gradient('z')};
    std::array<std::vector<double>, 3> cum = {g[0].cumulative_moment(), g[1].cumulative_moment(),
                                              g[2].cumulative_moment()};
    return Sampler(std::move(g), std::move(cum));
}

std::vector<Ps> WaveformSource::excitation_centers() const {
    const std::vector<std::tuple<Ps, double, double>> exc = excitations();
    std::vector<Ps> out;
    out.reserve(exc.size());
    for (const auto& e : exc) {
        out.push_back(std::get<0>(e));
    }
    return out;
}

std::vector<Ps> WaveformSource::adc_event_starts() const {
    const std::vector<AdcEvent> events = adc_events();
    std::vector<Ps> out;
    out.reserve(events.size());
    for (const AdcEvent& e : events) {
        out.push_back(e.t.empty() ? Ps::zero() : e.t.front());
    }
    return out;
}

std::vector<std::tuple<double, double, double>> WaveformSource::adc_excitations() const {
    const std::vector<std::tuple<Ps, double, double>> exc = excitations();
    std::vector<Ps> centers;
    centers.reserve(exc.size());
    for (const auto& e : exc) {
        centers.push_back(std::get<0>(e));
    }

    // Only the event *starts*: this used to call `adc_events()` and read the first time of each,
    // building every sample time in the run to use 4,216 of them.
    const std::vector<Ps> starts = adc_event_starts();
    std::vector<std::tuple<double, double, double>> out;
    out.reserve(starts.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const Ps t : starts) {
        const std::size_t i = static_cast<std::size_t>(
            std::upper_bound(centers.begin(), centers.end(), t) - centers.begin());
        if (i == 0) {
            // No excitation precedes this readout: NaN rather than a silently wrong offset.
            out.emplace_back(nan, nan, nan);
        } else {
            out.emplace_back(std::get<0>(exc[i - 1]).secs(), std::get<1>(exc[i - 1]),
                             std::get<2>(exc[i - 1]));
        }
    }
    return out;
}

std::vector<SampleSpec> WaveformSource::adc_specs() const {
    const std::vector<Ps> excites = excitation_centers();
    std::vector<AdcEvent> events = adc_events();
    std::vector<SampleSpec> specs;
    specs.reserve(events.size());
    for (AdcEvent& ev : events) {
        SampleSpec sp;
        sp.reference = excite_before(excites, ev.t.empty() ? Ps::zero() : ev.t.front());
        sp.times = std::move(ev.t);
        specs.push_back(std::move(sp));
    }
    if (excites.size() != specs.size()) {
        std::fprintf(stderr,
                     "seq2waveform: warning: %zu excitation(s) but %zu ADC event(s); using the "
                     "most recent excitation before each ADC as the k-space reference\n",
                     excites.size(), specs.size());
    }
    return specs;
}

std::vector<SampleSpec> WaveformSource::trigger_specs(double delay, double dt,
                                                      std::size_t n) const {
    const std::vector<TriggerEvent> trgs = trigger_events();
    std::vector<SampleSpec> out;
    out.reserve(trgs.size());
    for (const TriggerEvent& trg : trgs) {
        SampleSpec sp;
        sp.times.reserve(n);
        for (std::size_t k = 0; k < n; ++k) {
            sp.times.push_back(trg.time + Ps::from_secs(delay + static_cast<double>(k) * dt));
        }
        sp.reference = trg.time;
        out.push_back(std::move(sp));
    }
    return out;
}

std::vector<Trace> WaveformSource::adc_gradient_samples() const {
    const Sampler s = sampler();
    const std::vector<SampleSpec> specs = adc_specs();
    std::vector<Trace> out;
    out.reserve(specs.size());
    for (const SampleSpec& sp : specs) {
        out.push_back(s.gradients(sp.times));
    }
    return out;
}

std::vector<Trace> WaveformSource::adc_kspace() const {
    const Sampler s = sampler();
    const std::vector<SampleSpec> specs = adc_specs();
    std::vector<Trace> out;
    out.reserve(specs.size());
    for (const SampleSpec& sp : specs) {
        out.push_back(s.kspace(sp.times, sp.reference));
    }
    return out;
}

std::vector<Trace> WaveformSource::trigger_gradient_samples(double delay, double dt,
                                                            std::size_t n) const {
    const Sampler s = sampler();
    const std::vector<SampleSpec> specs = trigger_specs(delay, dt, n);
    std::vector<Trace> out;
    out.reserve(specs.size());
    for (const SampleSpec& sp : specs) {
        out.push_back(s.gradients(sp.times));
    }
    return out;
}

std::vector<Trace> WaveformSource::trigger_kspace(double delay, double dt, std::size_t n) const {
    const Sampler s = sampler();
    const std::vector<SampleSpec> specs = trigger_specs(delay, dt, n);
    std::vector<Trace> out;
    out.reserve(specs.size());
    for (const SampleSpec& sp : specs) {
        out.push_back(s.kspace(sp.times, sp.reference));
    }
    return out;
}

FlatTrace WaveformSource::adc_kspace_flat() const {
    auto flat = flatten_specs(adc_specs());
    const FlatK k = sampler().kspace_flat(std::get<0>(flat), std::get<1>(flat), std::get<2>(flat));
    FlatTrace out;
    out.t = std::move(std::get<0>(flat));
    out.offsets = std::move(std::get<1>(flat));
    out.kx = std::move(k.kx);
    out.ky = std::move(k.ky);
    out.kz = std::move(k.kz);
    return out;
}

FlatTrace WaveformSource::trigger_kspace_flat(double delay, double dt, std::size_t n) const {
    auto flat = flatten_specs(trigger_specs(delay, dt, n));
    const FlatK k = sampler().kspace_flat(std::get<0>(flat), std::get<1>(flat), std::get<2>(flat));
    FlatTrace out;
    out.t = std::move(std::get<0>(flat));
    out.offsets = std::move(std::get<1>(flat));
    out.kx = std::move(k.kx);
    out.ky = std::move(k.ky);
    out.kz = std::move(k.kz);
    return out;
}

Trace WaveformSource::gradients_dense(double raster) const {
    std::vector<Ps> times = uniform_grid(duration(), raster);
    Trace tr;
    tr.x = gradient('x').sample_at_sorted(times);
    tr.y = gradient('y').sample_at_sorted(times);
    tr.z = gradient('z').sample_at_sorted(times);
    tr.t = std::move(times);  // a local, dead after this -- copying it duplicated the whole grid
    return tr;
}

std::pair<std::vector<Ps>, std::vector<double>> WaveformSource::gradient_axis_dense(
    char axis, double raster) const {
    std::vector<Ps> times = uniform_grid(duration(), raster);
    std::vector<double> v = gradient(axis).sample_at_sorted(times);
    return {std::move(times), std::move(v)};
}

// --- SequenceSource ------------------------------------------------------

PiecewiseLinear SequenceSource::gradient(char axis) const {
    return s2w::gradient(seq_, axis);  // the focused walk in waveform.cpp
}

std::vector<AdcEvent> SequenceSource::adc_events() const { return s2w::adc_events(seq_); }

std::vector<Ps> SequenceSource::adc_event_starts() const { return s2w::adc_event_starts(seq_); }

std::vector<TriggerEvent> SequenceSource::trigger_events() const {
    return s2w::trigger_events(seq_);
}

double SequenceSource::duration() const {
    // The summed block durations, in block order, so the f64 accumulation matches Rust's `sum()`.
    double total = 0.0;
    for (const Block& b : seq_.blocks) {
        total += b.duration;
    }
    return total;
}

std::vector<std::tuple<Ps, double, double>> SequenceSource::excitations() const {
    std::vector<std::tuple<Ps, double, double>> out;
    for (const RfMeta& r : s2w::rf_events_meta(seq_)) {
        if (r.use == RfUse::Excitation) {
            out.emplace_back(r.center, r.freq, r.phase);
        }
    }
    // Sort by center only, and stably, so pulses sharing a center keep block order -- Rust's
    // `sort_by_key(|x| x.0)` is a stable sort on the same key.
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });
    return out;
}

// --- free helpers --------------------------------------------------------

std::vector<Ps> uniform_grid(double duration, double raster) {
    if (raster <= 0.0 || duration <= 0.0) {
        return {Ps::zero()};
    }
    const std::size_t n = static_cast<std::size_t>(std::ceil(duration / raster)) + 1;
    std::vector<Ps> out;
    out.reserve(n);
    for (std::size_t k = 0; k < n; ++k) {
        out.push_back(Ps::from_secs(static_cast<double>(k) * raster));
    }
    return out;
}

std::tuple<std::vector<Ps>, std::vector<std::size_t>, std::vector<Ps>> flatten_specs(
    const std::vector<SampleSpec>& specs) {
    std::vector<Ps> times;
    std::vector<std::size_t> offsets = {0};
    std::vector<Ps> refs;
    refs.reserve(specs.size());
    offsets.reserve(specs.size() + 1);
    // Size the flat array from the exact total: appending 4000-odd events into a growing vector
    // recopies the whole run every 1.5x, which on `cones` is 21.5 M elements moved repeatedly.
    std::size_t total = 0;
    for (const SampleSpec& sp : specs) {
        total += sp.times.size();
    }
    times.reserve(total);
    for (const SampleSpec& sp : specs) {
        times.insert(times.end(), sp.times.begin(), sp.times.end());
        offsets.push_back(times.size());
        refs.push_back(sp.reference);
    }
    return {std::move(times), std::move(offsets), std::move(refs)};
}

}  // namespace s2w
