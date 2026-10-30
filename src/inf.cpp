#include "seq2waveform/inf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "seq2waveform/detail/inf_detail.hpp"
#include "seq2waveform/scan.hpp"

namespace s2w {
namespace detail {

std::vector<AdcSegment> merge_contiguous(const std::vector<AdcSegment>& segs) {
    std::vector<AdcSegment> out;
    for (const AdcSegment& seg : segs) {
        if (!out.empty()) {
            AdcSegment& last = out.back();
            // `last.dwell()` is read before the merge mutates it, as in Rust.
            const double dwell = last.dwell();
            const bool same_dwell =
                std::fabs(dwell - seg.dwell()) <= 1e-6 * std::fmax(dwell, 1e-12);
            const bool contiguous =
                std::fabs(seg.start - (last.start + last.duration)) <= 0.5 * seg.dwell();
            if (same_dwell && contiguous) {
                last.duration += seg.duration;
                last.num_samples += seg.num_samples;
                continue;
            }
        }
        out.push_back(seg);
    }
    return out;
}

std::optional<std::pair<std::string_view, std::string_view>> split_cell(std::string_view s) {
    // `[label]: a/ b` -> the trimmed (a, b) after the `[label]:` prefix.
    const std::size_t marker = s.find("]:");
    const std::string_view body = marker == std::string_view::npos ? s : s.substr(marker + 2);
    const std::size_t slash = body.find('/');
    if (slash == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string_view a = trim(body.substr(0, slash));
    const std::string_view b = trim(body.substr(slash + 1));
    if (a.empty()) {
        return std::nullopt;
    }
    return std::make_pair(a, b);
}

}  // namespace detail

namespace {

/// The first whitespace-separated token of `s` parsed as a double, if it parses.
std::optional<double> first_f64(std::string_view s) {
    const std::vector<std::string_view> f = fields(s);
    double v = 0.0;
    if (f.empty() || !parse_f64(f[0], v)) {
        return std::nullopt;
    }
    return v;
}

}  // namespace

// --- AdcSegment ----------------------------------------------------------

void AdcSegment::append_sample_times(std::vector<Ps>& out) const {
    const double d = dwell();
    for (std::size_t k = 0; k < num_samples; ++k) {
        out.push_back(Ps::from_secs(start + d * (static_cast<double>(k) + 0.5)));
    }
}

std::vector<Ps> AdcSegment::sample_times() const {
    std::vector<Ps> out;
    out.reserve(num_samples);
    append_sample_times(out);
    return out;
}

// --- RfEvent -------------------------------------------------------------

std::optional<double> RfEvent::rfd_peak(const DsvChannel& rfd) const {
    const PiecewiseLinear& pwl = rfd.waveform;
    const double dt = rfd.raster_time;
    if (pwl.t.empty() || dt <= 0.0) {
        return std::nullopt;
    }
    const double end = start + duration;

    // Bracket the pulse window by binary search instead of testing every corner. The corners are
    // sorted and `secs()` is monotonic in the raw count, so these bounds select exactly the
    // indices `t >= start && t <= end` picked out before -- the same set, bit for bit. Scanning the
    // whole channel twice per event instead made this quadratic: on `cones` that is 4,236 RF
    // events over 309,230 RFD corners, 1.3 G iterations and ~1.1 s, paid by every DSV k-space call
    // because `excitations()` goes through here.
    const std::size_t lo = static_cast<std::size_t>(
        std::lower_bound(pwl.t.begin(), pwl.t.end(), start,
                         [](Ps corner, double s) { return corner.secs() < s; }) -
        pwl.t.begin());
    const std::size_t hi = static_cast<std::size_t>(
        std::upper_bound(pwl.t.begin(), pwl.t.end(), end,
                         [](double e, Ps corner) { return e < corner.secs(); }) -
        pwl.t.begin());

    // Rust starts from f64::MIN (the most negative finite), not -infinity.
    double vmax = std::numeric_limits<double>::lowest();
    for (std::size_t i = lo; i < hi; ++i) {
        vmax = std::fmax(vmax, pwl.v[i]);
    }
    if (vmax <= 0.0) {
        return std::nullopt;
    }

    // Plateau extent (corners at or above 0.99999 * max), plus a max-sample time for refinement.
    const double thr = vmax * 0.99999;
    double first = 0.0;
    double last = 0.0;
    double tp = 0.0;
    bool seen = false;
    for (std::size_t i = lo; i < hi; ++i) {
        const double t = pwl.t[i].secs();
        if (pwl.v[i] >= thr) {
            if (!seen) {
                first = t;
                seen = true;
            }
            last = t;
            tp = t;
        }
    }
    if (!seen) {
        return std::nullopt;
    }
    if (last - first > dt * 0.5) {
        return 0.5 * (first + last);  // a real plateau -> its midpoint
    }

    // A sharp peak: parabolic refinement from the neighbouring raster samples.
    const double vl = pwl.value_at(Ps::from_secs(tp - dt));
    const double vr = pwl.value_at(Ps::from_secs(tp + dt));
    const double denom = vl - 2.0 * vmax + vr;
    if (denom < 0.0) {
        const double p = std::fmin(std::fmax(0.5 * (vl - vr) / denom, -0.5), 0.5);
        return tp + p * dt;
    }
    return tp;
}

std::pair<double, double> RfEvent::center_and_offset(const DsvChannel& rfd) const {
    const double middle = start + duration * 0.5;
    const std::optional<double> peak = rfd_peak(rfd);
    const double p = peak.value_or(middle);
    const double off = p - middle;
    const double c = std::fabs(off) > rfd.raster_time ? p : middle;
    return {c, off};
}

double RfEvent::center(const DsvChannel& rfd) const { return center_and_offset(rfd).first; }

// --- Inf -----------------------------------------------------------------

std::optional<std::size_t> Inf::col(const std::string& prefix) const {
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (starts_with(columns[i], prefix)) {
            return i;
        }
    }
    return std::nullopt;
}

std::vector<Ps> Inf::adc_sample_times() const {
    // Count first, reserve once, then append in place -- rather than a vector per segment copied
    // into a growing output, which allocated per segment and touched every sample twice.
    std::size_t total = 0;
    for (const AdcSegment& s : adc) {
        total += s.num_samples;
    }
    std::vector<Ps> out;
    out.reserve(total);
    for (const AdcSegment& s : adc) {
        s.append_sample_times(out);
    }
    return out;
}

std::vector<AdcSegment> Inf::adc_events() const { return detail::merge_contiguous(adc); }

double Inf::lead_in() const { return blocks.size() > 1 ? blocks[1].start : 0.0; }

std::vector<SyncEvent> Inf::triggers() const {
    const std::optional<std::size_t> si = col("SYNC");
    if (!si.has_value()) {
        return {};
    }
    std::vector<SyncEvent> out;
    for (const EventBlock& block : blocks) {
        for (const EventRow& r : block.rows) {
            if (*si >= r.cells.size()) {
                continue;
            }
            const auto parts = detail::split_cell(r.cells[*si]);
            if (!parts.has_value()) {
                continue;
            }
            double dur_us = 0.0;
            if (!parse_f64(parts->second, dur_us)) {
                continue;
            }
            SyncEvent e;
            e.name = std::string(parts->first);
            e.time = block.start + r.rel_time;
            e.duration = dur_us * 1e-6;
            out.push_back(std::move(e));
        }
    }
    return out;
}

std::vector<RfEvent> Inf::rf_events() const {
    const std::optional<std::size_t> txi = col("TXFlip");
    if (!txi.has_value()) {
        return {};
    }
    const std::optional<std::size_t> fpi = col("Freq/Phase");
    std::vector<RfEvent> out;
    for (const EventBlock& block : blocks) {
        // (rel_time, freq, phase) for each Freq/Phase entry in this block, in row order.
        std::vector<std::array<double, 3>> fps;
        if (fpi.has_value()) {
            for (const EventRow& r : block.rows) {
                if (*fpi >= r.cells.size()) {
                    continue;
                }
                const auto parts = detail::split_cell(r.cells[*fpi]);
                if (!parts.has_value()) {
                    continue;
                }
                double a = 0.0;
                double b = 0.0;
                if (parse_f64(parts->first, a) && parse_f64(parts->second, b)) {
                    fps.push_back({r.rel_time, a, b});
                }
            }
        }

        for (const EventRow& r : block.rows) {
            if (*txi >= r.cells.size()) {
                continue;
            }
            const auto parts = detail::split_cell(r.cells[*txi]);
            if (!parts.has_value()) {
                continue;
            }
            double flip = 0.0;
            double dur_us = 0.0;
            if (!parse_f64(parts->first, flip) || !parse_f64(parts->second, dur_us)) {
                continue;
            }
            // The LAST Freq/Phase entry at or before this row -- Rust's `filter(..).last()`, so it
            // relies on row order, not on sorting.
            double freq = 0.0;
            double phase = 0.0;
            for (const std::array<double, 3>& e : fps) {
                if (e[0] <= r.rel_time + 1e-12) {
                    freq = e[1];
                    phase = e[2];
                }
            }
            RfEvent ev;
            ev.start = block.start + r.rel_time;
            ev.duration = dur_us * 1e-6;
            ev.flip_deg = flip;
            ev.freq = freq;
            ev.phase_deg = phase;
            out.push_back(ev);
        }
    }
    return out;
}

// --- parsing -------------------------------------------------------------

Inf read_inf_stream(std::istream& in) {
    Inf inf;
    std::optional<std::pair<double, double>> pending_adc;  // (start_us, dur_us)
    std::optional<EventBlock> cur;

    // The INF files run to 142 MB, so the line loop matters: read in 64 KiB blocks and split the
    // buffer in place rather than allocating a std::string per line (see PORTING.md's note on
    // parse_text).
    std::string text;
    {
        char buf[1 << 16];
        while (in.read(buf, sizeof buf) || in.gcount() > 0) {
            text.append(buf, static_cast<std::size_t>(in.gcount()));
        }
    }

    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t nl = text.find('\n', pos);
        const std::size_t stop = nl == std::string::npos ? text.size() : nl;
        const std::string_view t = trim(std::string_view(text).substr(pos, stop - pos));
        pos = nl == std::string::npos ? text.size() : nl + 1;

        if (starts_with(t, "* EventBlock")) {
            if (cur.has_value()) {
                inf.blocks.push_back(std::move(*cur));
                cur.reset();
            }
            pending_adc.reset();
            EventBlock b;
            b.start = first_f64(t.substr(12)).value_or(0.0) * 1e-6;
            cur = std::move(b);
        } else if (starts_with(t, "adc MeasHeader")) {
            if (cur.has_value()) {
                inf.blocks.push_back(std::move(*cur));
                cur.reset();
            }
            const std::vector<std::string_view> f = fields(t.substr(14));
            double s = 0.0;
            double d = 0.0;
            // Only armed when BOTH parse, as in Rust.
            pending_adc.reset();
            if (f.size() >= 2 && parse_f64(f[0], s) && parse_f64(f[1], d)) {
                pending_adc = std::make_pair(s, d);
            }
        } else if (starts_with(t, "ushSamplesInScan")) {
            if (pending_adc.has_value()) {
                const std::pair<double, double> sd = *pending_adc;
                pending_adc.reset();
                std::size_t n = 0;
                const std::size_t colon = t.find(':');
                if (colon != std::string_view::npos) {
                    const std::vector<std::string_view> v = fields(t.substr(colon + 1));
                    if (!v.empty() && !parse_usize(v[0], n)) {
                        n = 0;
                    }
                }
                if (n > 0) {
                    AdcSegment seg;
                    seg.start = sd.first * 1e-6;
                    seg.duration = sd.second * 1e-6;
                    seg.num_samples = n;
                    inf.adc.push_back(seg);
                }
            }
        } else if (!t.empty() && t.front() == '|') {
            // Split on '|'; a box border has too few fields to be a table row.
            std::vector<std::string_view> parts;
            std::size_t p = 0;
            while (true) {
                const std::size_t bar = t.find('|', p);
                if (bar == std::string_view::npos) {
                    parts.push_back(t.substr(p));
                    break;
                }
                parts.push_back(t.substr(p, bar - p));
                p = bar + 1;
            }
            if (parts.size() < 3 || !cur.has_value()) {
                continue;
            }
            std::vector<std::string> cells;
            cells.reserve(parts.size() - 2);
            for (std::size_t i = 1; i + 1 < parts.size(); ++i) {
                cells.emplace_back(trim(parts[i]));
            }
            const std::string_view first = cells.empty() ? std::string_view{} : cells.front();
            if (first == "Rel.Time") {
                if (inf.columns.empty()) {
                    inf.columns = std::move(cells);  // record the header once
                }
            } else {
                double rel_us = 0.0;
                if (parse_f64(first, rel_us)) {
                    EventRow row;
                    row.rel_time = rel_us * 1e-6;
                    row.cells = std::move(cells);
                    cur->rows.push_back(std::move(row));
                }
                // else: a separator, the rotation matrix or RTAdjData -> ignore
            }
        }
    }
    if (cur.has_value()) {
        inf.blocks.push_back(std::move(*cur));
    }
    return inf;
}

Inf read_inf_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("I/O error: cannot open " + path);
    }
    return read_inf_stream(in);
}

}  // namespace s2w
