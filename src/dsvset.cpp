#include "seq2waveform/dsvset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

namespace s2w {
namespace {

constexpr double kPi = 3.14159265358979323846;

double to_radians(double degrees) { return degrees * (kPi / 180.0); }

/// An empty channel, returned for an unknown axis so the behaviour matches Rust's
/// `_ => return PiecewiseLinear::default()`.
const DsvChannel& empty_channel() {
    static const DsvChannel empty;
    return empty;
}

}  // namespace

DsvSet::DsvSet(DsvChannel grx, DsvChannel gry, DsvChannel grz, DsvChannel rfd, Inf inf,
               double gamma)
    : grx_(std::move(grx)),
      gry_(std::move(gry)),
      grz_(std::move(grz)),
      rfd_(std::move(rfd)),
      inf_(std::move(inf)),
      gamma_(gamma) {}

DsvSet DsvSet::from_paths(const std::string& grx, const std::string& gry, const std::string& grz,
                          const std::string& rfd, const std::string& inf, double gamma) {
    return DsvSet(read_dsv_file(grx), read_dsv_file(gry), read_dsv_file(grz), read_dsv_file(rfd),
                  read_inf_file(inf), gamma);
}

const DsvChannel& DsvSet::channel(char axis) const {
    switch (axis) {
        case 'x':
            return grx_;
        case 'y':
            return gry_;
        case 'z':
            return grz_;
        default:
            return empty_channel();
    }
}

PiecewiseLinear DsvSet::gradient(char axis) const {
    if (axis != 'x' && axis != 'y' && axis != 'z') {
        return PiecewiseLinear{};
    }
    const DsvChannel& ch = channel(axis);
    // Hz/m = mT/m * 1e-3 (T/m) * gamma (Hz/T). Formed as one factor and applied per value, in
    // that order, so the rounding matches Rust's.
    const double factor = gamma_ * 1e-3;
    PiecewiseLinear out;
    out.t = ch.waveform.t;
    out.v.reserve(ch.waveform.v.size());
    for (const double v : ch.waveform.v) {
        out.v.push_back(v * factor);
    }
    return out;
}

std::vector<AdcEvent> DsvSet::adc_events() const {
    const std::vector<AdcSegment> segs = inf_.adc_events();
    std::vector<AdcEvent> out;
    out.reserve(segs.size());
    for (const AdcSegment& seg : segs) {
        AdcEvent e;
        e.t = seg.sample_times();
        e.freq = 0.0;  // the INF carries no per-sample freq/phase
        e.phase = 0.0;
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<Ps> DsvSet::adc_event_starts() const {
    const std::vector<AdcSegment> segs = inf_.adc_events();
    std::vector<Ps> out;
    out.reserve(segs.size());
    for (const AdcSegment& seg : segs) {
        // `AdcSegment::sample_times()`'s k == 0 term, and `Ps::zero()` for an empty segment, so
        // this agrees with `adc_events()` exactly.
        out.push_back(seg.num_samples == 0
                          ? Ps::zero()
                          : Ps::from_secs(seg.start + seg.dwell() * 0.5));
    }
    return out;
}

std::vector<std::tuple<Ps, double, double>> DsvSet::excitations() const {
    const double dt = rfd_.raster_time;
    std::size_t n_off = 0;
    double max_off = 0.0;

    const std::vector<RfEvent> events = inf_.rf_events();
    std::vector<std::tuple<Ps, double, double>> out;
    out.reserve(events.size());
    for (const RfEvent& e : events) {
        const std::pair<double, double> co = e.center_and_offset(rfd_);
        if (std::fabs(co.second) > dt) {
            ++n_off;
            max_off = std::fmax(max_off, std::fabs(co.second));
        }
        out.emplace_back(Ps::from_secs(co.first), e.freq, to_radians(e.phase_deg));
    }
    if (n_off > 0) {
        std::fprintf(stderr,
                     "seq2waveform: warning: %zu RF pulse(s) have their peak >%.0f us off the "
                     "window middle (max %.0f us) - using the sub-raster RFD peak as the "
                     "excitation reference (asymmetric pulse, e.g. min-phase). That center is "
                     "limited by the %.0f us RFD raster, unlike the exact middle used for "
                     "symmetric pulses.\n",
                     n_off, dt * 1e6, max_off * 1e6, dt * 1e6);
    }
    // Sort by center only, and stably, so pulses sharing a center keep INF order -- Rust's
    // `sort_by_key(|x| x.0)` is a stable sort on the same key.
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });
    return out;
}

std::vector<TriggerEvent> DsvSet::trigger_events() const {
    const std::vector<SyncEvent> syncs = inf_.triggers();
    std::vector<TriggerEvent> out;
    out.reserve(syncs.size());
    for (const SyncEvent& e : syncs) {
        TriggerEvent t;
        t.channel = 0;  // the SYNC column has no channel
        t.time = Ps::from_secs(e.time);
        t.duration = Ps::from_secs(e.duration);
        t.trig_type = 0;
        out.push_back(t);
    }
    return out;
}

double DsvSet::duration() const {
    return std::fmax(std::fmax(grx_.duration(), gry_.duration()), grz_.duration());
}

}  // namespace s2w
