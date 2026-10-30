#include "seq2waveform/waveform.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "seq2waveform/detail/waveform_detail.hpp"
#include "seq2waveform/scan.hpp"

namespace s2w {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// Walk the blocks, calling `fn(start, block)` with each block's absolute start time.
///
/// The single place block durations accumulate -- every expansion below goes through it, which is
/// what keeps their time bases identical. Rust spells it as an iterator; the accumulation is the
/// same, including that `Ps::from_secs` rounds each duration before it is added.
template <typename F>
void walk(const Sequence& seq, F&& fn) {
    Ps t = Ps::zero();
    for (const Block& b : seq.blocks) {
        const Ps start = t;
        t += Ps::from_secs(b.duration);
        fn(start, b);
    }
}

/// How many blocks carry a non-zero id for `pick`. A cheap pre-pass so an event list can be
/// reserved once; an upper bound, since a block whose id is missing from the library emits nothing.
template <typename F>
std::size_t count_events(const Sequence& seq, F&& pick) {
    std::size_t n = 0;
    for (const Block& b : seq.blocks) {
        if (pick(b) != 0) {
            ++n;
        }
    }
    return n;
}

/// The decompressed samples of shape `id`, or empty when absent -- Rust's
/// `shapes.get(&id).map(decompressed).unwrap_or_default()`.
std::vector<double> shape_or_empty(const Sequence& seq, std::uint32_t id) {
    const auto it = seq.shapes.find(id);
    return it == seq.shapes.end() ? std::vector<double>{} : it->second.decompressed();
}

}  // namespace

namespace detail {

/// Reconstruct raster-edge corner points from a centers-sampled arbitrary gradient.
///
/// Port of `restoreAdditionalShapeSamples.m`: the samples sit at raster centers, with
/// `center[k] = 0.5*(edge[k] + edge[k+1])`. Reconstruct the edges (an alternating-sign cumulative
/// sum from `first`), blend with the smooth average-of-neighbours estimate where the two agree,
/// interleave edges and centers on the half-raster, then keep only the corners (non-zero second
/// difference). Falls back to the raw centers (with `first`/`last`) when the reconstruction's last
/// edge disagrees with the recorded `last` -- the "spiral" case.
std::pair<std::vector<double>, std::vector<double>> restore_additional_shape_samples(
    const std::vector<double>& tt, const std::vector<double>& waveform, double first, double last,
    double grad_raster) {
    const std::size_t n = waveform.size();
    double max_abs = 0.0;
    for (const double x : waveform) {
        // std::fmax, like Rust's f64::max, returns the non-NaN operand.
        max_abs = std::fmax(max_abs, std::fabs(x));
    }

    // edge_rest[i]: edge value at raster boundary i (i = 0..n), from the alternating recurrence.
    std::vector<double> edge_rest(n + 1, 0.0);
    double c = 0.0;
    for (std::size_t i = 0; i <= n; ++i) {
        const double val = i == 0 ? first : 2.0 * waveform[i - 1];
        const double s = i % 2 == 0 ? 1.0 : -1.0;
        c += val * s;
        edge_rest[i] = c * s;
    }

    // Smooth edge estimate: the average of the neighbouring centers.
    std::vector<double> edge_interp(n + 1, 0.0);
    edge_interp[0] = first;
    edge_interp[n] = last;
    for (std::size_t k = 1; k < n; ++k) {
        edge_interp[k] = 0.5 * (waveform[k - 1] + waveform[k]);
    }

    // Spiral fallback: the recurrence did not land on the recorded `last`, so trust the centers.
    if (std::fabs(edge_rest[n] - last) > 2e-5 * max_abs) {
        std::vector<double> t;
        std::vector<double> v;
        t.reserve(n + 2);
        v.reserve(n + 2);
        t.push_back(0.0);
        v.push_back(first);
        for (std::size_t k = 0; k < n; ++k) {
            t.push_back(tt[k]);
            v.push_back(waveform[k]);
        }
        t.push_back(tt[n - 1] + grad_raster / 2.0);
        v.push_back(last);
        return {std::move(t), std::move(v)};
    }

    // Blend: smooth where the two estimates agree, exact where they differ (a genuine kink).
    const double tol = std::numeric_limits<double>::epsilon() + 2e-5 * max_abs;
    std::vector<double> edge(n + 1, 0.0);
    for (std::size_t i = 0; i <= n; ++i) {
        edge[i] = std::fabs(edge_rest[i] - edge_interp[i]) <= tol ? edge_interp[i] : edge_rest[i];
    }

    // Interleave on the half-raster: [edge0, w0, edge1, w1, ..., w_{n-1}, edge_n].
    std::vector<double> os;
    os.reserve(2 * n + 1);
    for (std::size_t k = 0; k < n; ++k) {
        os.push_back(edge[k]);
        os.push_back(waveform[k]);
    }
    os.push_back(edge[n]);

    // Keep the corners: the endpoints, plus anything with a non-zero second difference. The
    // endpoint tests come first so os[i-1] is never evaluated at i == 0.
    const std::size_t m = os.size();
    std::vector<double> t;
    std::vector<double> v;
    for (std::size_t i = 0; i < m; ++i) {
        const bool keep = i == 0 || i == m - 1 ||
                          std::fabs(os[i + 1] - 2.0 * os[i] + os[i - 1]) > 1e-8;
        if (keep) {
            t.push_back(static_cast<double>(i) * grad_raster * 0.5);
            v.push_back(os[i]);
        }
    }
    return {std::move(t), std::move(v)};
}

/// Resample a rendered gradient channel onto the gradient-raster CENTERS (`(k+0.5)*raster`) and
/// keep only the corners -- the Siemens DSV / commanded waveform. Trapezoid corners get rounded
/// (the corner sits between centers), arbitrary shapes come out smooth (centers lie on the
/// rendered ramps), and straight runs collapse back to their endpoints via a streaming collinear
/// drop, so the result stays sparse and is dense only where genuinely curved.
PiecewiseLinear resample_at_centers(const PiecewiseLinear& pwl, double raster) {
    const std::size_t n = pwl.t.size();
    if (n < 2 || raster <= 0.0) {
        return pwl;
    }
    const std::int64_t raster_ps = Ps::from_secs(raster).raw();
    const std::int64_t half = raster_ps / 2;
    const auto center = [raster_ps, half](std::int64_t k) { return Ps(raster_ps * k + half); };
    const Ps t0 = pwl.t[0];
    const Ps t1 = pwl.t[n - 1];
    if (std::ceil(t0.secs() / raster - 0.5) > std::floor(t1.secs() / raster - 0.5)) {
        return pwl;  // no center falls inside the span; resampling would lose it
    }
    // One center BEYOND each end, carrying the source's clamped value there (0 for a gradient that
    // starts and ends at rest). Without them the outermost kept center sits half a raster inside
    // the first/last ramp, and `value_at` clamps to that non-zero value for every time outside the
    // channel -- a gradient reading non-zero before its first event and after its last.
    const std::int64_t k_lo = saturating_i64(std::ceil(t0.secs() / raster - 0.5)) - 1;
    const std::int64_t k_hi = saturating_i64(std::floor(t1.secs() / raster - 0.5)) + 1;

    std::vector<Ps> t;
    std::vector<double> v;
    // A moving cursor over the corners (centers are monotonic): O(centers + corners), not a binary
    // search per center.
    std::size_t j = std::max<std::size_t>(
        static_cast<std::size_t>(std::upper_bound(pwl.t.begin(), pwl.t.end(), center(k_lo)) -
                                 pwl.t.begin()),
        1);
    std::int64_t k = k_lo;
    while (k <= k_hi) {
        const Ps tc = center(k);
        while (j < n && pwl.t[j] <= tc) {
            ++j;
        }
        // Clamp outside [t0, t1] exactly as value_at does; never extrapolate off the end segments.
        double vc = 0.0;
        if (tc <= t0) {
            vc = pwl.v[0];
        } else if (j >= n) {
            vc = pwl.v[n - 1];
        } else {
            const Ps ta = pwl.t[j - 1];
            const Ps tb = pwl.t[j];
            const double va = pwl.v[j - 1];
            const double vb = pwl.v[j];
            vc = tb == ta ? va : va + (vb - va) * tc.frac_between(ta, tb);
        }
        t.push_back(tc);
        v.push_back(vc);

        // Drop trailing collinear middle points, so flats and ramps stay two corners rather than
        // one per raster cell.
        while (t.size() >= 3) {
            const std::size_t m = t.size();
            const std::size_t a = m - 3;
            const std::size_t b = m - 2;
            const std::size_t cc = m - 1;
            const double cross = (t[b] - t[a]).secs() * (v[cc] - v[b]) -
                                 (t[cc] - t[b]).secs() * (v[b] - v[a]);
            const double scale =
                raster * (std::fmax(std::fmax(std::fabs(v[a]), std::fabs(v[b])), std::fabs(v[cc])) +
                          1.0);
            if (std::fabs(cross) <= 1e-9 * scale) {
                t.erase(t.begin() + static_cast<std::ptrdiff_t>(b));
                v.erase(v.begin() + static_cast<std::ptrdiff_t>(b));
            } else {
                break;
            }
        }

        // Skip the collinear interior of the current linear segment: the next corner is pwl.t[j],
        // and centers strictly before it lie on the same line and would be dropped anyway. Long
        // flats, gaps and ramps become O(1) instead of one iteration per raster cell.
        std::int64_t next_k = 0;
        if (tc <= t0) {
            next_k = k;  // the lead-in point is off the segment [t[j-1], t[j]]: no interior to skip
        } else if (j < n) {
            next_k = saturating_i64(std::floor(pwl.t[j].secs() / raster - 0.5));
        } else {
            next_k = k_hi;
        }
        k = next_k > k + 1 ? next_k : k + 1;
    }
    PiecewiseLinear out;
    out.t = std::move(t);
    out.v = std::move(v);
    return out;
}

}  // namespace detail

namespace {

using detail::resample_at_centers;
using detail::restore_additional_shape_samples;

/// Concatenate per-block gradient pieces into one continuous channel, porting the
/// `waveforms_and_times` stitch: insert a forced ramp to and from zero across gaps, and drop
/// duplicate samples at shared boundaries.
PiecewiseLinear stitch(std::vector<PiecewiseLinear> pieces, double grad_raster) {
    const Ps grad = Ps::from_secs(grad_raster);
    const Ps half = Ps(grad.raw() / 2);
    std::vector<Ps> t;
    std::vector<double> v;

    // One allocation each, from the exact total. Growing instead would recopy the whole output
    // every 1.5x (MSVC's growth factor) -- tens of MB of memcpy on a long sequence. The slack is
    // for the forced ramps, which add at most two corners per piece boundary.
    std::size_t total = 2 * pieces.size();
    for (const PiecewiseLinear& piece : pieces) {
        total += piece.t.size();
    }
    t.reserve(total);
    v.reserve(total);

    for (PiecewiseLinear& piece : pieces) {
        std::vector<Ps> lt = std::move(piece.t);
        std::vector<double> lv = std::move(piece.v);
        if (lt.empty()) {
            continue;
        }

        std::size_t cnt = t.size();
        if (cnt != 0 && t[cnt - 1] + grad < lt[0]) {
            // A gap between the previous piece's end and this piece's start. A value more than
            // 1e-6 from zero gets an explicit half-raster ramp; anything smaller is snapped,
            // because inserting a corner for it would be noise.
            if (v[cnt - 1] != 0.0) {
                if (std::fabs(v[cnt - 1]) > 1e-6) {
                    t.push_back(t[cnt - 1] + half);
                    v.push_back(0.0);
                } else {
                    v.back() = 0.0;
                }
            }
            if (lv[0] != 0.0) {
                if (std::fabs(lv[0]) > 1e-6) {
                    lt.insert(lt.begin(), lt[0] - half);
                    lv.insert(lv.begin(), 0.0);
                } else {
                    lv[0] = 0.0;
                }
            }
        }

        cnt = t.size();  // the gap handling may have appended, and may have shifted lt[0]
        if (cnt == 0 || t[cnt - 1] < lt[0]) {
            t.insert(t.end(), lt.begin(), lt.end());
            v.insert(v.end(), lv.begin(), lv.end());
        } else {
            // Overlap: append from the first local sample strictly after the current end.
            const std::size_t d = static_cast<std::size_t>(
                std::find_if(lt.begin(), lt.end(), [&](Ps x) { return x > t[cnt - 1]; }) -
                lt.begin());
            t.insert(t.end(), lt.begin() + static_cast<std::ptrdiff_t>(d), lt.end());
            v.insert(v.end(), lv.begin() + static_cast<std::ptrdiff_t>(d), lv.end());
        }
    }

    PiecewiseLinear out;
    out.t = std::move(t);
    out.v = std::move(v);
    return out;
}

/// Build one block's gradient piece (corner points), porting `waveforms_and_times`. Empty when the
/// event contributes nothing -- a zero-length trapezoid, or a missing/empty shape.
bool build_grad_piece(const Sequence& seq, const Gradient& g, Ps curr_dur, double grad_raster,
                      PiecewiseLinear& out) {
    if (const Trap* tr = std::get_if<Trap>(&g)) {
        const Ps off = curr_dur + Ps::from_secs(tr->delay);
        // The sums are formed in f64 before conversion, exactly as in Rust: from_secs(rise + flat)
        // is not from_secs(rise) + from_secs(flat).
        if (std::fabs(tr->flat) > std::numeric_limits<double>::epsilon()) {
            out.t = {off, off + Ps::from_secs(tr->rise), off + Ps::from_secs(tr->rise + tr->flat),
                     off + Ps::from_secs(tr->rise + tr->flat + tr->fall)};
            out.v = {0.0, tr->amplitude, tr->amplitude, 0.0};
            return true;
        }
        if (std::fabs(tr->rise) > std::numeric_limits<double>::epsilon() &&
            std::fabs(tr->fall) > std::numeric_limits<double>::epsilon()) {
            out.t = {off, off + Ps::from_secs(tr->rise), off + Ps::from_secs(tr->rise + tr->fall)};
            out.v = {0.0, tr->amplitude, 0.0};
            return true;
        }
        return false;  // empty gradient
    }

    const ArbitraryGrad& a = std::get<ArbitraryGrad>(g);
    const auto amp_it = seq.shapes.find(a.amp_shape);
    if (amp_it == seq.shapes.end()) {
        return false;  // Rust's `?` on the shape lookup
    }
    const std::vector<double> shape = amp_it->second.decompressed();
    const std::size_t n = shape.size();
    if (n == 0) {
        return false;
    }
    std::vector<double> waveform(n);
    for (std::size_t k = 0; k < n; ++k) {
        waveform[k] = a.amplitude * shape[k];
    }

    // Build the time vector and the shape duration (port of getBlock).
    std::vector<double> tt;
    double shape_dur = 0.0;
    if (a.time_shape == 0) {
        tt.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            tt[k] = (static_cast<double>(k) + 0.5) * grad_raster;
        }
        shape_dur = static_cast<double>(n) * grad_raster;
    } else {
        const std::vector<double> ts = shape_or_empty(seq, a.time_shape);
        tt.resize(ts.size());
        for (std::size_t k = 0; k < ts.size(); ++k) {
            tt[k] = ts[k] * grad_raster;
        }
        shape_dur = tt.empty() ? 0.0 : tt.back();
        // Rust indexes tt[k] for k < n below and would panic on a short time shape. Throwing is
        // the same failure, with a message instead of undefined behaviour here.
        if (tt.size() != n) {
            throw std::runtime_error("gradient time shape " + std::to_string(a.time_shape) +
                                     " has " + std::to_string(tt.size()) +
                                     " samples but amplitude shape " +
                                     std::to_string(a.amp_shape) + " has " + std::to_string(n));
        }
    }

    const Ps off = curr_dur + Ps::from_secs(a.delay);
    bool is_centers = true;
    for (std::size_t k = 0; k < tt.size(); ++k) {
        if (std::fabs(tt[k] / grad_raster - (static_cast<double>(k) + 0.5)) >= 1e-6) {
            is_centers = false;
            break;
        }
    }

    if (is_centers && seq.grad_mode == GradMode::Centers) {
        // Connect the center samples directly (matching the Siemens DSV / played waveform), with
        // first/last at the block edges. No edge reconstruction.
        out.t.clear();
        out.v.clear();
        out.t.reserve(n + 2);
        out.v.reserve(n + 2);
        out.t.push_back(off);
        out.v.push_back(a.first);
        for (std::size_t k = 0; k < n; ++k) {
            out.t.push_back(off + Ps::from_secs(tt[k]));
            out.v.push_back(waveform[k]);
        }
        out.t.push_back(off + Ps::from_secs(tt[n - 1] + grad_raster / 2.0));
        out.v.push_back(a.last);
        return true;
    }
    if (is_centers) {
        std::pair<std::vector<double>, std::vector<double>> rec =
            restore_additional_shape_samples(tt, waveform, a.first, a.last, grad_raster);
        out.t.clear();
        out.t.reserve(rec.first.size());
        for (const double x : rec.first) {
            out.t.push_back(off + Ps::from_secs(x));
        }
        out.v = std::move(rec.second);  // `rec` is a local, dead after this: copying it duplicated
        return true;                    // every reconstructed corner, once per block per axis
    }
    if (std::fabs(tt[0] / grad_raster - 0.5) < 1e-6) {
        // Oversampled or rasterized: pad with first/last at the edges.
        out.t.clear();
        out.v.clear();
        out.t.reserve(n + 2);
        out.v.reserve(n + 2);
        out.t.push_back(off);
        out.v.push_back(a.first);
        for (std::size_t k = 0; k < n; ++k) {
            out.t.push_back(off + Ps::from_secs(tt[k]));
            out.v.push_back(waveform[k]);
        }
        out.t.push_back(off + Ps::from_secs(shape_dur));
        out.v.push_back(a.last);
        return true;
    }
    // Extended trapezoid: the edges are already on the raster.
    out.t.clear();
    out.t.reserve(n);
    for (std::size_t k = 0; k < n; ++k) {
        out.t.push_back(off + Ps::from_secs(tt[k]));
    }
    out.v = std::move(waveform);
    return true;
}

void push_grad(const Sequence& seq, std::uint32_t id, Ps curr_dur, double grad_raster,
               std::vector<PiecewiseLinear>& out) {
    if (id == 0) {
        return;
    }
    const auto it = seq.gradients.find(id);
    if (it == seq.gradients.end()) {
        return;
    }
    PiecewiseLinear piece;
    if (build_grad_piece(seq, it->second, curr_dur, grad_raster, piece)) {
        out.push_back(std::move(piece));
    }
}

/// Stitch the per-block pieces into one channel, then apply the gradient mode's final pass.
PiecewiseLinear finish_channel(const Sequence& seq, std::vector<PiecewiseLinear> pieces,
                               double grad_raster) {
    // Deliberately not `const`: a const local cannot be implicitly moved on return, so `return
    // pwl` would deep-copy both corner arrays -- 43 MB per call on `cones`.
    PiecewiseLinear pwl = stitch(std::move(pieces), grad_raster);
    if (seq.grad_mode == GradMode::Dsv) {
        return resample_at_centers(pwl, grad_raster);
    }
    return pwl;
}

AdcEvent build_adc(const Adc& adc, Ps curr_dur) {
    AdcEvent e;
    e.t.reserve(adc.num);
    for (std::uint32_t k = 0; k < adc.num; ++k) {
        e.t.push_back(curr_dur + Ps::from_secs(adc.delay + adc.dwell * (static_cast<double>(k) + 0.5)));
    }
    e.freq = adc.freq;
    e.phase = adc.phase;
    return e;
}

RfPulse build_rf(const Sequence& seq, const Rf& rf, Ps curr_dur, double rf_raster) {
    const std::vector<double> mag = shape_or_empty(seq, rf.mag_shape);
    const std::vector<double> phase = shape_or_empty(seq, rf.phase_shape);
    const std::size_t n = mag.size();
    RfPulse p;
    p.t.reserve(n);
    p.re.reserve(n);
    p.im.reserve(n);
    for (std::size_t k = 0; k < n; ++k) {
        p.t.push_back(curr_dur +
                      Ps::from_secs(rf.delay + (static_cast<double>(k) + 0.5) * rf_raster));
        const double amp = rf.amplitude * mag[k];
        const double ph = 2.0 * kPi * (k < phase.size() ? phase[k] : 0.0);  // shape is in cycles
        p.re.push_back(amp * std::cos(ph));
        p.im.push_back(amp * std::sin(ph));
    }
    p.center = curr_dur + Ps::from_secs(rf.delay + rf.center);
    p.freq = rf.freq;
    p.phase = rf.phase;
    p.use = rf.use;
    return p;
}

/// Walk one block's extension list, collecting TRIGGERS entries. The node count is capped, as in
/// Rust, so a self-referential `next_id` cannot loop forever.
void resolve_triggers(const Sequence& seq, std::uint32_t ext_start, Ps curr_dur,
                      std::vector<TriggerEvent>& out) {
    std::uint32_t node_id = ext_start;
    for (int guard = 0; guard < 10000; ++guard) {
        if (node_id == 0) {
            break;
        }
        const auto node_it = seq.ext_nodes.find(node_id);
        if (node_it == seq.ext_nodes.end()) {
            break;
        }
        const ExtNode& node = node_it->second;
        const auto name_it = seq.ext_type_names.find(node.type_id);
        const bool is_trig = name_it != seq.ext_type_names.end() && name_it->second == "TRIGGERS";
        if (is_trig) {
            const auto trg_it = seq.triggers.find(node.ref_id);
            if (trg_it != seq.triggers.end()) {
                const Trigger& trg = trg_it->second;
                TriggerEvent e;
                e.channel = trg.channel;
                e.time = curr_dur + Ps::from_secs(trg.delay);
                e.duration = Ps::from_secs(trg.duration);
                e.trig_type = trg.trig_type;
                out.push_back(e);
            }
        }
        node_id = node.next_id;
    }
}

double grad_raster_of(const Sequence& seq) {
    return seq.definition_f64("GradientRasterTime").value_or(10e-6);
}

}  // namespace

Waveforms waveforms(const Sequence& seq) {
    const double grad_raster = grad_raster_of(seq);
    const double rf_raster = seq.definition_f64("RadiofrequencyRasterTime").value_or(1e-6);

    Waveforms w;
    // At most one piece per block on each axis; sizing up front keeps the piece lists from
    // regrowing (and moving every piece) as the walk proceeds.
    std::vector<PiecewiseLinear> gx;
    std::vector<PiecewiseLinear> gy;
    std::vector<PiecewiseLinear> gz;
    gx.reserve(seq.blocks.size());
    gy.reserve(seq.blocks.size());
    gz.reserve(seq.blocks.size());

    walk(seq, [&](Ps start, const Block& block) {
        if (block.rf != 0) {
            const auto it = seq.rf.find(block.rf);
            if (it != seq.rf.end()) {
                w.rf.push_back(build_rf(seq, it->second, start, rf_raster));
            }
        }
        if (block.adc != 0) {
            const auto it = seq.adc.find(block.adc);
            if (it != seq.adc.end()) {
                w.adc.push_back(build_adc(it->second, start));
            }
        }
        if (block.ext != 0) {
            resolve_triggers(seq, block.ext, start, w.triggers);
        }
        push_grad(seq, block.gx, start, grad_raster, gx);
        push_grad(seq, block.gy, start, grad_raster, gy);
        push_grad(seq, block.gz, start, grad_raster, gz);
    });

    w.gx = finish_channel(seq, std::move(gx), grad_raster);
    w.gy = finish_channel(seq, std::move(gy), grad_raster);
    w.gz = finish_channel(seq, std::move(gz), grad_raster);
    return w;
}

PiecewiseLinear gradient(const Sequence& seq, char axis) {
    const double grad_raster = grad_raster_of(seq);
    std::vector<PiecewiseLinear> pieces;
    pieces.reserve(seq.blocks.size());  // at most one piece per block
    walk(seq, [&](Ps start, const Block& block) {
        std::uint32_t id = 0;
        switch (axis) {
            case 'x':
                id = block.gx;
                break;
            case 'y':
                id = block.gy;
                break;
            case 'z':
                id = block.gz;
                break;
            default:
                id = 0;
        }
        push_grad(seq, id, start, grad_raster, pieces);
    });
    return finish_channel(seq, std::move(pieces), grad_raster);
}

std::vector<Ps> adc_times(const Sequence& seq) {
    // Count first, then reserve once. Reserving per block instead -- reserve(size() + num) inside
    // the walk -- asks for the exact new size every time, which defeats the geometric growth
    // push_back gives for free and recopies the whole array on every ADC event. On `cones` (4216
    // events, 21.5 M samples) that was 47 s of pure memcpy; this is ~60 ms.
    std::size_t total = 0;
    for (const Block& b : seq.blocks) {
        if (b.adc == 0) {
            continue;
        }
        const auto it = seq.adc.find(b.adc);
        if (it != seq.adc.end()) {
            total += it->second.num;
        }
    }

    std::vector<Ps> out;
    out.reserve(total);
    walk(seq, [&](Ps start, const Block& block) {
        if (block.adc == 0) {
            return;
        }
        const auto it = seq.adc.find(block.adc);
        if (it == seq.adc.end()) {
            return;
        }
        const Adc& adc = it->second;
        for (std::uint32_t k = 0; k < adc.num; ++k) {
            out.push_back(start +
                          Ps::from_secs(adc.delay + adc.dwell * (static_cast<double>(k) + 0.5)));
        }
    });
    return out;
}

std::vector<AdcEvent> adc_events(const Sequence& seq) {
    std::vector<AdcEvent> out;
    out.reserve(count_events(seq, [](const Block& b) { return b.adc; }));
    walk(seq, [&](Ps start, const Block& block) {
        if (block.adc == 0) {
            return;
        }
        const auto it = seq.adc.find(block.adc);
        if (it != seq.adc.end()) {
            out.push_back(build_adc(it->second, start));
        }
    });
    return out;
}

std::vector<Ps> adc_event_starts(const Sequence& seq) {
    std::vector<Ps> out;
    out.reserve(count_events(seq, [](const Block& b) { return b.adc; }));
    walk(seq, [&](Ps start, const Block& block) {
        if (block.adc == 0) {
            return;
        }
        const auto it = seq.adc.find(block.adc);
        if (it == seq.adc.end()) {
            return;
        }
        const Adc& adc = it->second;
        // `build_adc`'s k == 0 term, and `Ps::zero()` for an empty event -- the same value
        // `adc_events()` would yield, so the excitation matching cannot drift from the full path.
        out.push_back(adc.num == 0 ? Ps::zero()
                                   : start + Ps::from_secs(adc.delay + adc.dwell * 0.5));
    });
    return out;
}

std::vector<RfMeta> rf_events_meta(const Sequence& seq) {
    std::vector<RfMeta> out;
    out.reserve(count_events(seq, [](const Block& b) { return b.rf; }));
    walk(seq, [&](Ps start, const Block& block) {
        if (block.rf == 0) {
            return;
        }
        const auto it = seq.rf.find(block.rf);
        if (it == seq.rf.end()) {
            return;
        }
        const Rf& rf = it->second;
        RfMeta m;
        m.center = start + Ps::from_secs(rf.delay + rf.center);
        m.freq = rf.freq;
        m.phase = rf.phase;
        m.use = rf.use;
        out.push_back(m);
    });
    return out;
}

std::vector<TriggerEvent> trigger_events(const Sequence& seq) {
    std::vector<TriggerEvent> out;
    walk(seq, [&](Ps start, const Block& block) {
        if (block.ext != 0) {
            resolve_triggers(seq, block.ext, start, out);
        }
    });
    return out;
}

}  // namespace s2w
