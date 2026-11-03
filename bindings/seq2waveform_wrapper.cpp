// nanobind bindings: call the core and return NumPy arrays owned by capsules.
// Times cross the boundary as float64 seconds or exact int64 picoseconds.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "seq2waveform/constants.hpp"
#include "seq2waveform/dsv.hpp"
#include "seq2waveform/dsvset.hpp"
#include "seq2waveform/error.hpp"
#include "seq2waveform/inf.hpp"
#include "seq2waveform/model.hpp"
#include "seq2waveform/parser.hpp"
#include "seq2waveform/pwl.hpp"
#include "seq2waveform/sample.hpp"
#include "seq2waveform/time.hpp"
#include "seq2waveform/waveform.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using Out = nb::ndarray<nb::numpy, T, nb::ndim<1>>;
template <typename T>
using In = nb::ndarray<const T, nb::ndim<1>, nb::c_contig>;

/// Move a vector into a capsule; NumPy shares its storage until the last view is freed.
template <typename T>
Out<T> to_numpy(std::vector<T>&& vec) {
    auto* owned = new std::vector<T>(std::move(vec));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return Out<T>(owned->data(), {owned->size()}, owner);
}

std::vector<s2w::Ps> ps_from_seconds(In<double> times) {
    std::vector<s2w::Ps> out;
    out.reserve(times.shape(0));
    for (std::size_t i = 0; i < times.shape(0); ++i) {
        out.push_back(s2w::Ps::from_secs(times(i)));
    }
    return out;
}

std::vector<s2w::Ps> ps_verbatim(In<std::int64_t> times) {
    std::vector<s2w::Ps> out;
    out.reserve(times.shape(0));
    for (std::size_t i = 0; i < times.shape(0); ++i) {
        out.push_back(s2w::Ps(times(i)));
    }
    return out;
}

std::vector<double> to_vector(In<double> a) {
    if (a.shape(0) == 0) {
        return {};
    }
    return std::vector<double>(a.data(), a.data() + a.shape(0));
}

s2w::PiecewiseLinear make_waveform(std::vector<s2w::Ps> times, std::vector<double> values) {
    if (times.size() != values.size()) {
        throw nb::value_error("corner times and values must have the same length");
    }
    if (std::adjacent_find(times.begin(), times.end(),
                           [](s2w::Ps a, s2w::Ps b) { return a >= b; }) != times.end()) {
        throw nb::value_error("corner times must be strictly increasing in picoseconds");
    }
    return {std::move(times), std::move(values)};
}

In<double> checked_moments(const s2w::PiecewiseLinear& w, In<double> cum) {
    if (cum.shape(0) != w.size()) {
        throw nb::value_error("cum must have one value per waveform corner");
    }
    return cum;
}

/// Corner or sample times as float64 seconds.
///
/// Seconds are lossless here: the raw int64 is multiplied by 1e-12, and sequence times stay below
/// 1e15 ps, well inside the 2^53 where that map is injective. Callers who need the exact grid back
/// -- a delay sweep, say -- can recover it with `ps_from_secs`, which `test_invariants.py` pins.
std::vector<double> secs_vec(const std::vector<s2w::Ps>& t) {
    std::vector<double> out;
    out.reserve(t.size());
    for (const s2w::Ps x : t) {
        out.push_back(x.secs());
    }
    return out;
}

Out<double> secs_array(const std::vector<s2w::Ps>& t) { return to_numpy(secs_vec(t)); }

s2w::GradMode grad_mode_from_str(const std::string& mode) {
    if (mode == "pulseq") {
        return s2w::GradMode::Pulseq;
    }
    if (mode == "centers") {
        return s2w::GradMode::Centers;
    }
    if (mode == "dsv") {
        return s2w::GradMode::Dsv;
    }
    throw nb::value_error("gradient_mode must be \"pulseq\", \"centers\", or \"dsv\"");
}

const char* grad_mode_to_str(s2w::GradMode m) {
    switch (m) {
        case s2w::GradMode::Centers:
            return "centers";
        case s2w::GradMode::Dsv:
            return "dsv";
        case s2w::GradMode::Pulseq:
            break;
    }
    return "pulseq";
}

/// Any-rank, C-contiguous input array -- the Sampler accepts an N-D time grid.
template <typename T>
using InAny = nb::ndarray<const T, nb::c_contig>;

/// Take ownership of a trace and move its axis arrays into NumPy.
nb::object trace_to_py(s2w::Trace tr) {
    return nb::make_tuple(secs_array(tr.t), to_numpy(std::move(tr.x)), to_numpy(std::move(tr.y)),
                          to_numpy(std::move(tr.z)));
}

/// Flatten query times to picoseconds, retaining the shape for the result.
template <typename T, typename Conv>
std::pair<std::vector<s2w::Ps>, std::vector<std::size_t>> flatten_times(const InAny<T>& a,
                                                                       Conv conv) {
    std::vector<std::size_t> shape;
    std::size_t total = 1;
    for (std::size_t d = 0; d < a.ndim(); ++d) {
        shape.push_back(a.shape(d));
        total *= a.shape(d);
    }
    std::vector<s2w::Ps> out;
    out.reserve(total);
    const T* p = a.data();
    for (std::size_t i = 0; i < total; ++i) {
        out.push_back(conv(p[i]));
    }
    return {std::move(out), std::move(shape)};
}

/// Restore the query shape, including scalar arrays.
nb::object reshape_like(std::vector<double>&& v, const std::vector<std::size_t>& shape) {
    nb::object arr = nb::cast(to_numpy(std::move(v)));
    if (shape.size() == 1) {
        return arr;
    }
    return arr.attr("reshape")(nb::cast(shape));
}

s2w::Ps ps_of_secs(double x) { return s2w::Ps::from_secs(x); }
s2w::Ps ps_of_raw(std::int64_t x) { return s2w::Ps(x); }

/// Convert sample specs to Python (times_s, reference_s) tuples.
std::vector<nb::object> specs_to_py(const std::vector<s2w::SampleSpec>& specs) {
    std::vector<nb::object> out;
    out.reserve(specs.size());
    for (const s2w::SampleSpec& sp : specs) {
        out.push_back(nb::make_tuple(secs_array(sp.times), sp.reference.secs()));
    }
    return out;
}

/// Move whole-run axis arrays to Python; offsets need a size_t-to-int64 conversion.
nb::object flat_to_py(s2w::FlatTrace f) {
    std::vector<std::int64_t> off;
    off.reserve(f.offsets.size());
    for (const std::size_t o : f.offsets) {
        off.push_back(static_cast<std::int64_t>(o));
    }
    return nb::make_tuple(secs_array(f.t), to_numpy(std::move(f.kx)), to_numpy(std::move(f.ky)),
                          to_numpy(std::move(f.kz)), to_numpy(std::move(off)));
}

char axis_from_str(const std::string& axis) {
    if (axis.size() != 1 || (axis[0] != 'x' && axis[0] != 'y' && axis[0] != 'z')) {
        throw nb::value_error("axis must be x, y, or z");
    }
    return axis[0];
}


/// Bind the shared sampling API once for Sequence and DsvSet.
template <typename T>
void bind_waveform_source(nb::class_<T>& cls) {
    cls.def(
           "gradient",
           [](const T& s, const std::string& axis) {
               // Keep the values movable so NumPy can take ownership.
               s2w::PiecewiseLinear g = s.gradient(axis_from_str(axis));
               return nb::make_tuple(secs_array(g.t), to_numpy(std::move(g.v)));
           },
           "axis"_a,
           "Expand one gradient channel to piecewise-linear corners.\n\n"
           "Parameters\n"
           "----------\n"
           "axis : {'x', 'y', 'z'}\n"
           "    Which gradient channel.\n\n"
           "Returns\n"
           "-------\n"
           "corner_times : ndarray of float64\n"
           "    Corner times (s), strictly increasing.\n"
           "corner_values : ndarray of float64\n"
           "    Corner amplitudes (Hz/m).")
        .def_prop_ro(
            "duration", [](const T& s) { return s.duration(); },
            "Total duration (s). For a `Sequence` the summed block durations; for a `DsvSet` the\n"
            "DSV channel span, which is the Siemens frame and includes the lead-in.")
        .def(
            "adc_events",
            [](const T& s) {
                const std::vector<s2w::AdcEvent> events = s.adc_events();
                std::vector<nb::object> out;
                out.reserve(events.size());
                for (const s2w::AdcEvent& e : events) {
                    out.push_back(nb::cast(secs_array(e.t)));
                }
                return out;
            },
            "ADC sample times split per event.\n\n"
            "Returns\n"
            "-------\n"
            "list of ndarray\n"
            "    One float64 array of sample times (s) per ADC event. For a `DsvSet` the Siemens\n"
            "    `MaxAdcSegmentLength` chunks are merged back first, so the grouping matches a\n"
            "    `.seq`'s.")
        .def(
            "adc_events_meta",
            [](const T& s) {
                const std::vector<s2w::AdcEvent> events = s.adc_events();
                std::vector<nb::object> out;
                out.reserve(events.size());
                for (const s2w::AdcEvent& e : events) {
                    out.push_back(nb::make_tuple(secs_array(e.t), e.freq, e.phase));
                }
                return out;
            },
            "ADC events with their per-event frequency and phase offsets.\n\n"
            "Returns\n"
            "-------\n"
            "list of tuple\n"
            "    `(times_s, freq_hz, phase_rad)` per ADC event. Both are 0 for a `DsvSet`: the INF\n"
            "    does not record them.")
        .def(
            "triggers",
            [](const T& s) {
                const std::vector<s2w::TriggerEvent> trgs = s.trigger_events();
                std::vector<std::tuple<std::uint32_t, double, double>> out;
                out.reserve(trgs.size());
                for (const s2w::TriggerEvent& t : trgs) {
                    out.emplace_back(t.channel, t.time.secs(), t.duration.secs());
                }
                return out;
            },
            "Trigger events as `(channel, time_s, duration_s)`.\n\n"
            "Returns\n"
            "-------\n"
            "list of tuple\n"
            "    One entry per event, in order. `channel` is 0 for a `DsvSet`: the SYNC column\n"
            "    carries no channel.")
        .def(
            "sampler", [](const T& s) { return s.sampler(); },
            "Build a `Sampler` over this source's three gradient channels.\n\n"
            "Precomputes the channels and their cumulative moments, so repeated sampling (a delay\n"
            "sweep, say) does not re-expand the waveforms.\n\n"
            "Returns\n"
            "-------\n"
            "Sampler")
        .def(
            "adc_specs", [](const T& s) { return specs_to_py(s.adc_specs()); },
            "Where to sample for each ADC event, and what to integrate k-space from.\n\n"
            "Uses the most recent excitation at or before the event, or the waveform start if\n"
            "none precedes it. Writes a warning to stderr when excitation and ADC counts differ.\n\n"
            "Returns\n"
            "-------\n"
            "list of tuple\n"
            "    `(times_s, reference_s)` per ADC event.")
        .def(
            "adc_excitations", [](const T& s) { return s.adc_excitations(); },
            "The excitation matched to each ADC event, aligned 1:1 with `adc_specs`.\n\n"
            "Returns\n"
            "-------\n"
            "list of tuple\n"
            "    `(center_s, freq_hz, phase_rad)` per ADC event, or `(nan, nan, nan)` when no\n"
            "    excitation precedes it. Use `freq` and the gradient at `center` to get the slice\n"
            "    offset of each readout.")
        .def(
            "trigger_specs",
            [](const T& s, double delay, double dt, std::size_t n) {
                return specs_to_py(s.trigger_specs(delay, dt, n));
            },
            "delay"_a, "dt"_a, "n"_a,
            "Sample specs for trigger-based (Skope) measurements.\n\n"
            "Parameters\n"
            "----------\n"
            "delay : float\n"
            "    Offset (s) from each trigger to the first sample.\n"
            "dt : float\n"
            "    Sample spacing (s).\n"
            "n : int\n"
            "    Samples per trigger.\n\n"
            "Returns\n"
            "-------\n"
            "list of tuple\n"
            "    `(times_s, reference_s)` per trigger. The reference is the trigger itself, not\n"
            "    an excitation -- that is the external measurement's own reference.")
        .def(
            "adc_gradient_samples",
            [](const T& s) {
                std::vector<s2w::Trace> traces = s.adc_gradient_samples();
                std::vector<nb::object> out;
                out.reserve(traces.size());
                for (s2w::Trace& tr : traces) {
                    out.push_back(trace_to_py(std::move(tr)));
                }
                return out;
            },
            "Gradient values (Hz/m) at every ADC sample, one trace per event.\n\n"
            "Returns\n"
            "-------\n"
            "list of tuple\n"
            "    `(t, gx, gy, gz)` per ADC event.")
        .def(
            "adc_kspace",
            [](const T& s) {
                std::vector<s2w::Trace> traces = s.adc_kspace();
                std::vector<nb::object> out;
                out.reserve(traces.size());
                for (s2w::Trace& tr : traces) {
                    out.push_back(trace_to_py(std::move(tr)));
                }
                return out;
            },
            "k-space (1/m) at every ADC sample, one trace per event.\n\n"
            "Each event integrates from its own preceding excitation. Gamma is already baked into\n"
            "the Hz/m gradients, so this is the integral of G dt directly.\n\n"
            "Returns\n"
            "-------\n"
            "list of tuple\n"
            "    `(t, kx, ky, kz)` per ADC event.")
        .def(
            "adc_kspace_flat", [](const T& s) { return flat_to_py(s.adc_kspace_flat()); },
            "Whole-run ADC k-space as flat arrays -- the fast path for large sequences.\n\n"
            "Returns\n"
            "-------\n"
            "tuple\n"
            "    `(t, kx, ky, kz, offsets)`. `offsets` are cumulative event boundaries, so\n"
            "    `offsets[0] == 0`, `offsets[-1] == len(t)`, and event `e` is\n"
            "    `offsets[e]:offsets[e+1]`. Recover the per-event split with\n"
            "    `np.split(kx, offsets[1:-1])`.")
        .def(
            "trigger_gradient_samples",
            [](const T& s, double delay, double dt, std::size_t n) {
                std::vector<s2w::Trace> traces = s.trigger_gradient_samples(delay, dt, n);
                std::vector<nb::object> out;
                out.reserve(traces.size());
                for (s2w::Trace& tr : traces) {
                    out.push_back(trace_to_py(std::move(tr)));
                }
                return out;
            },
            "delay"_a, "dt"_a, "n"_a,
            "Gradient values at trigger-based sample points, one trace per trigger.")
        .def(
            "trigger_kspace",
            [](const T& s, double delay, double dt, std::size_t n) {
                std::vector<s2w::Trace> traces = s.trigger_kspace(delay, dt, n);
                std::vector<nb::object> out;
                out.reserve(traces.size());
                for (s2w::Trace& tr : traces) {
                    out.push_back(trace_to_py(std::move(tr)));
                }
                return out;
            },
            "delay"_a, "dt"_a, "n"_a,
            "k-space at trigger-based sample points, integrated from each trigger.")
        .def(
            "trigger_kspace_flat",
            [](const T& s, double delay, double dt, std::size_t n) {
                return flat_to_py(s.trigger_kspace_flat(delay, dt, n));
            },
            "delay"_a, "dt"_a, "n"_a,
            "Trigger-based k-space as flat arrays. See `adc_kspace_flat`.")
        .def(
            "gradients_dense",
            [](const T& s, double raster) { return trace_to_py(s.gradients_dense(raster)); },
            "raster"_a,
            "All three gradient axes on a uniform raster over `[0, duration]`.\n\n"
            "Parameters\n"
            "----------\n"
            "raster : float\n"
            "    Sample spacing (s).\n\n"
            "Returns\n"
            "-------\n"
            "tuple\n"
            "    `(t, gx, gy, gz)`. The last sample is at or past `duration`.")
        .def(
            "gradient_axis_dense",
            [](const T& s, const std::string& axis, double raster) {
                auto tv = s.gradient_axis_dense(axis_from_str(axis), raster);
                return nb::make_tuple(secs_array(tv.first), to_numpy(std::move(tv.second)));
            },
            "axis"_a, "raster"_a,
            "One gradient axis on a uniform raster over `[0, duration]`.\n\n"
            "Returns\n"
            "-------\n"
            "tuple\n"
            "    `(t, values_hz_per_m)`.");
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() =
        "Compiled core of seq2waveform.\n\n"
        "Import the public API from the `seq2waveform` package rather than from here.";

    nb::class_<s2w::PiecewiseLinear>(m, "PiecewiseLinear",
                                     "A piecewise-linear waveform: the value is linear between\n"
                                     "`(t, v)` corners, clamped to the endpoints outside the corner\n"
                                     "range. The canonical gradient form -- what Pulseq emits\n"
                                     "(a trapezoid is 4 corners) and what the DSV collapses to.\n\n"
                                     "Corner times are held as int64 picoseconds internally, so\n"
                                     "differences are exact at any absolute time.")
        .def(
            "__init__",
            [](s2w::PiecewiseLinear* self, In<double> t_seconds, In<double> values) {
                new (self) s2w::PiecewiseLinear(
                    make_waveform(ps_from_seconds(t_seconds), to_vector(values)));
            },
            "t_seconds"_a, "values"_a,
            "Build from corner times in seconds.\n\n"
            "Parameters\n"
            "----------\n"
            "t_seconds : ndarray of float64\n"
            "    Strictly increasing corner times (s); rounded to the nearest picosecond.\n"
            "values : ndarray of float64\n"
            "    Corner values, same length as `t_seconds` (Hz/m for gradients).")
        .def_static(
            "from_picoseconds",
            [](In<std::int64_t> t_ps, In<double> values) {
                return make_waveform(ps_verbatim(t_ps), to_vector(values));
            },
            "t_ps"_a, "values"_a,
            "Build from corner times in int64 picoseconds, taken verbatim.\n\n"
            "The bit-exact path: use it when corner times came from another exact source and\n"
            "must not round-trip through float seconds.\n\n"
            "Parameters\n"
            "----------\n"
            "t_ps : ndarray of int64\n"
            "    Strictly increasing corner times (ps).\n"
            "values : ndarray of float64\n"
            "    Corner values, same length as `t_ps`.")
        .def("__len__", &s2w::PiecewiseLinear::size, "Number of corners.")
        .def(
            "corner_times",
            [](const s2w::PiecewiseLinear& w) {
                std::vector<double> out;
                out.reserve(w.t.size());
                for (const s2w::Ps p : w.t) {
                    out.push_back(p.secs());
                }
                return to_numpy(std::move(out));
            },
            "Corner times in seconds.")
        .def(
            "corner_times_ps",
            [](const s2w::PiecewiseLinear& w) {
                std::vector<std::int64_t> out;
                out.reserve(w.t.size());
                for (const s2w::Ps p : w.t) {
                    out.push_back(p.raw());
                }
                return to_numpy(std::move(out));
            },
            "Corner times as exact int64 picoseconds.")
        .def(
            "values",
            [](const s2w::PiecewiseLinear& w) { return to_numpy(std::vector<double>(w.v)); },
            "Corner values.")
        .def(
            "value_at",
            [](const s2w::PiecewiseLinear& w, double t_seconds) {
                return w.value_at(s2w::Ps::from_secs(t_seconds));
            },
            "t_seconds"_a,
            "Linearly interpolated value at `t_seconds`, clamped outside the corner range.")
        .def(
            "integral",
            [](const s2w::PiecewiseLinear& w, double t0, double t1) {
                return w.integral(s2w::Ps::from_secs(t0), s2w::Ps::from_secs(t1));
            },
            "t0"_a, "t1"_a,
            "Exact integral of the waveform from `t0` to `t1` (s), as a sum of trapezoids.\n\n"
            "Negative when `t1 < t0`. This is the k-space building block.")
        .def(
            "sample_at",
            [](const s2w::PiecewiseLinear& w, In<double> times) {
                return to_numpy(w.sample_at(ps_from_seconds(times)));
            },
            "times"_a,
            "Sample at arbitrary `times` (float64 seconds), in any order.\n\n"
            "Returns\n"
            "-------\n"
            "ndarray of float64\n"
            "    One value per entry of `times`.")
        .def(
            "sample_at",
            [](const s2w::PiecewiseLinear& w, In<std::int64_t> times) {
                return to_numpy(w.sample_at(ps_verbatim(times)));
            },
            "times"_a, "Sample at arbitrary `times` (int64 picoseconds, taken verbatim).")
        .def(
            "sample_at_sorted",
            [](const s2w::PiecewiseLinear& w, In<double> times) {
                return to_numpy(w.sample_at_sorted(ps_from_seconds(times)));
            },
            "times"_a,
            "Sample at non-decreasing `times` (float64 seconds) with one forward sweep.\n\n"
            "O(corners + times) instead of a binary search per point -- the hot path for an ADC\n"
            "readout. Results are undefined if `times` is not non-decreasing.")
        .def(
            "sample_at_sorted",
            [](const s2w::PiecewiseLinear& w, In<std::int64_t> times) {
                return to_numpy(w.sample_at_sorted(ps_verbatim(times)));
            },
            "times"_a,
            "Sample at non-decreasing `times` (int64 picoseconds) with one forward sweep.")
        .def(
            "cumulative_moment",
            [](const s2w::PiecewiseLinear& w) { return to_numpy(w.cumulative_moment()); },
            "Running integral at each corner: `cum[i]` is the integral from `t[0]` to `t[i]`.\n\n"
            "`moment_to` locates each query by binary search; `moments_at_sorted` uses a forward sweep.")
        .def(
            "moment_to",
            [](const s2w::PiecewiseLinear& w, double t_seconds, In<double> cum) {
                return w.moment_to(s2w::Ps::from_secs(t_seconds), checked_moments(w, cum).data());
            },
            "t_seconds"_a, "cum"_a,
            "Integral from the first corner to `t_seconds`, using precomputed `cum`.\n\n"
            "k-space referenced to `t_ref` is `moment_to(t) - moment_to(t_ref)`.")
        .def(
            "moments_at_sorted",
            [](const s2w::PiecewiseLinear& w, In<double> times, In<double> cum) {
                return to_numpy(w.moments_at_sorted(ps_from_seconds(times),
                                                    to_vector(checked_moments(w, cum))));
            },
            "times"_a, "cum"_a,
            "`moment_to` over non-decreasing `times` (float64 seconds) with one forward sweep.")
        .def(
            "moments_at_sorted",
            [](const s2w::PiecewiseLinear& w, In<std::int64_t> times, In<double> cum) {
                return to_numpy(w.moments_at_sorted(ps_verbatim(times),
                                                    to_vector(checked_moments(w, cum))));
            },
            "times"_a, "cum"_a,
            "`moment_to` over non-decreasing `times` (int64 picoseconds) with one forward sweep.");

    m.def(
        "ps_from_secs",
        [](In<double> seconds) {
            std::vector<std::int64_t> out;
            out.reserve(seconds.shape(0));
            for (std::size_t i = 0; i < seconds.shape(0); ++i) {
                out.push_back(s2w::Ps::from_secs(seconds(i)).raw());
            }
            return to_numpy(std::move(out));
        },
        "seconds"_a,
        "Convert seconds to exact int64 picoseconds (rounds half away from zero).\n\n"
        "Use this to pin a delay sweep to an exact grid before sampling.");

    m.def(
        "ps_to_secs",
        [](In<std::int64_t> picoseconds) {
            std::vector<double> out;
            out.reserve(picoseconds.shape(0));
            for (std::size_t i = 0; i < picoseconds.shape(0); ++i) {
                out.push_back(s2w::Ps(picoseconds(i)).secs());
            }
            return to_numpy(std::move(out));
        },
        "picoseconds"_a, "Convert exact int64 picoseconds back to float64 seconds.");

    m.attr("SCAN_GAMMA") = s2w::kScanGamma;

    // Preserve parser details in Python ValueError messages.
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) {
            try {
                std::rethrow_exception(p);
            } catch (const s2w::ParseError& e) {
                PyErr_SetString(PyExc_ValueError, e.what());
            } catch (const s2w::DsvError& e) {
                PyErr_SetString(PyExc_ValueError, e.what());
            }
        },
        nullptr);

    m.def(
        "parse_file",
        [](const std::string& path) { return s2w::parse_file(path).blocks.size(); }, "path"_a,
        "Parse a `.seq` file and return the number of blocks.\n\n"
        "Parameters\n"
        "----------\n"
        "path : str\n"
        "    Path to a Pulseq `.seq` file with the 1.5 event layout.\n\n"
        "Returns\n"
        "-------\n"
        "int\n"
        "    Number of rows in the `[BLOCKS]` table.\n\n"
        "Raises\n"
        "------\n"
        "ValueError\n"
        "    If a line is malformed; the message names the source line.");

    nb::class_<s2w::SequenceSource> sequence(
        m, "Sequence",
        "A parsed Pulseq sequence: parse once, then query waveforms many times without\n"
        "re-parsing.\n\n"
        "Parameters\n"
        "----------\n"
        "path : str\n"
        "    Path to a Pulseq `.seq` file with the 1.5 event layout.\n"
        "gradient_mode : {'pulseq', 'centers', 'dsv'}, optional\n"
        "    How centers-sampled arbitrary gradients render; see the `gradient_mode`\n"
        "    property. Default 'pulseq'.\n\n"
        "Raises\n"
        "------\n"
        "ValueError\n"
        "    If a line is malformed, naming the source line, or if `gradient_mode` is not\n"
        "    one of the three names.");
    sequence
        .def(
            "__init__",
            [](s2w::SequenceSource* self, const std::string& path,
               const std::string& gradient_mode) {
                // Validate the mode before parsing, so a typo fails fast rather than after
                // reading a large file.
                const s2w::GradMode mode = grad_mode_from_str(gradient_mode);
                new (self) s2w::SequenceSource(s2w::parse_file(path));
                self->sequence().grad_mode = mode;
            },
            "path"_a, "gradient_mode"_a = "pulseq")
        .def_prop_ro(
            "num_blocks", [](const s2w::SequenceSource& s) { return s.sequence().blocks.size(); },
            "Number of blocks in the parsed sequence.")
        .def_prop_rw(
            "gradient_mode",
            [](const s2w::SequenceSource& s) { return grad_mode_to_str(s.sequence().grad_mode); },
            [](s2w::SequenceSource& s, const std::string& mode) {
                s.sequence().grad_mode = grad_mode_from_str(mode);
            },
            "How arbitrary/centers-sampled gradients render:\n\n"
            "- `\"pulseq\"` (default): edge reconstruction, matches the h5 (can staircase on\n"
            "  fast waveforms);\n"
            "- `\"centers\"`: connect the center samples for *arbitrary* shapes (smooth),\n"
            "  trapezoids stay sharp;\n"
            "- `\"dsv\"`: resample the whole channel onto the raster centers -- rounds\n"
            "  trapezoid corners *and* smooths arbitrary shapes, i.e. the Siemens DSV /\n"
            "  commanded waveform, one consistent representation.")
        .def(
            "adc_times",
            [](const s2w::SequenceSource& s) { return secs_array(s2w::adc_times(s.sequence())); },
            "All ADC sample times (s), flattened across every ADC event.\n\n"
            "Returns\n"
            "-------\n"
            "ndarray of float64\n"
            "    Sample times (s), at dwell centers: `delay + dwell * (k + 0.5)`.")
        .def(
            "rf_events",
            [](const s2w::SequenceSource& s) {
                std::vector<std::tuple<double, double, double, std::string>> out;
                for (const s2w::RfMeta& r : s2w::rf_events_meta(s.sequence())) {
                    out.emplace_back(r.center.secs(), r.freq, r.phase, s2w::rf_use_name(r.use));
                }
                return out;
            },
            "RF pulses as `(center_s, freq_hz, phase_rad, use_name)`.\n\n"
            "`use_name` is the real RF use; undefined pulses are NOT relabelled as excitation,\n"
            "which is a deliberate divergence from Pulseq (its h5 routes both into\n"
            "`/excitation`, so an h5 comparison has to union the two).\n\n"
            "Returns\n"
            "-------\n"
            "list of tuple\n"
            "    One entry per RF event, in block order.");
    bind_waveform_source(sequence);

    nb::class_<s2w::DsvSet> dsvset(
        m, "DsvSet",
        "A Siemens DSP-Simulator measurement set: three gradient channels, the RFD envelope\n"
        "and the INF.\n\n"
        "Uses the same sampling API as `Sequence`. Gradients, ADC and RF events share the\n"
        "Siemens time frame, including its lead-in, so k-space needs no time shift. The INF\n"
        "does not record RF use; all RF events are treated as excitations.\n\n"
        "Parameters\n"
        "----------\n"
        "grx, gry, grz : str\n"
        "    The three gradient channel files, e.g. `SimulationProtocol_GRX.dsv`.\n"
        "rfd : str\n"
        "    The RF envelope channel, used to refine excitation centers.\n"
        "inf : str\n"
        "    The INF file: ADC events, triggers and RF events.\n"
        "gamma : float, optional\n"
        "    Gyromagnetic ratio (Hz/T) converting the DSV gradients (mT/m) to the Hz/m used\n"
        "    internally. Defaults to `SCAN_GAMMA`, the Siemens interpreter's value.");
    dsvset
        .def(
            "__init__",
            [](s2w::DsvSet* self, const std::string& grx, const std::string& gry,
               const std::string& grz, const std::string& rfd, const std::string& inf,
               double gamma) {
                new (self) s2w::DsvSet(s2w::DsvSet::from_paths(grx, gry, grz, rfd, inf, gamma));
            },
            "grx"_a, "gry"_a, "grz"_a, "rfd"_a, "inf"_a, "gamma"_a = s2w::kScanGamma)
        .def_prop_ro(
            "gamma", [](const s2w::DsvSet& s) { return s.gamma(); },
            "The gyromagnetic ratio (Hz/T) this set uses to convert mT/m to the internal Hz/m.\n\n"
            "Divide any Hz/m output by `gamma * 1e-3` to get mT/m, or by `gamma` for T/m.");
    bind_waveform_source(dsvset);

    nb::class_<s2w::Sampler>(
        m, "Sampler",
        "Precomputed gradient channels and their cumulative moments, for fast repeated\n"
        "sampling. Build one with `Sequence.sampler()` or `DsvSet.sampler()`.\n\n"
        "Query times are float64 seconds, or int64 picoseconds taken verbatim -- the exact path,\n"
        "for delay sweeps that must stay bit-exact. An N-D time array comes back with its shape\n"
        "preserved.\n\n"
        "`gradients` and `kspace` return `(t, x, y, z)`. `kspace_flat` returns `(kx, ky, kz)`.")
        .def(
            "gradients",
            [](const s2w::Sampler& s, InAny<double> times) {
                auto ts = flatten_times<double>(times, ps_of_secs);
                s2w::Trace tr = s.gradients(ts.first);
                return nb::make_tuple(reshape_like(secs_vec(tr.t), ts.second),
                                      reshape_like(std::move(tr.x), ts.second),
                                      reshape_like(std::move(tr.y), ts.second),
                                      reshape_like(std::move(tr.z), ts.second));
            },
            "times"_a,
            "Gradient values (Hz/m) at `times`, interpolated between corners.\n\n"
            "Parameters\n"
            "----------\n"
            "times : ndarray\n"
            "    Query times: float64 seconds, or int64 picoseconds taken verbatim. Any shape.\n"
            "    A non-decreasing array takes a single forward sweep per axis; otherwise it is a\n"
            "    binary search per point.\n\n"
            "Returns\n"
            "-------\n"
            "tuple of ndarray\n"
            "    `(t, gx, gy, gz)`, each shaped like `times`. `t` is the query grid in seconds,\n"
            "    after the picosecond quantisation every time in this library goes through, so it\n"
            "    is what was actually sampled rather than what was asked for.")
        .def(
            "gradients",
            [](const s2w::Sampler& s, InAny<std::int64_t> times) {
                auto ts = flatten_times<std::int64_t>(times, ps_of_raw);
                s2w::Trace tr = s.gradients(ts.first);
                return nb::make_tuple(reshape_like(secs_vec(tr.t), ts.second),
                                      reshape_like(std::move(tr.x), ts.second),
                                      reshape_like(std::move(tr.y), ts.second),
                                      reshape_like(std::move(tr.z), ts.second));
            },
            "times"_a, "As above, with query times given as exact int64 picoseconds.")
        .def(
            "kspace",
            [](const s2w::Sampler& s, InAny<double> times, double reference) {
                auto ts = flatten_times<double>(times, ps_of_secs);
                s2w::Trace tr = s.kspace(ts.first, s2w::Ps::from_secs(reference));
                return nb::make_tuple(reshape_like(secs_vec(tr.t), ts.second),
                                      reshape_like(std::move(tr.x), ts.second),
                                      reshape_like(std::move(tr.y), ts.second),
                                      reshape_like(std::move(tr.z), ts.second));
            },
            "times"_a, "reference"_a,
            "k-space (1/m) at `times`, integrating each gradient from `reference`.\n\n"
            "Parameters\n"
            "----------\n"
            "times : ndarray\n"
            "    Query times (float64 seconds or int64 picoseconds). Any shape.\n"
            "reference : float\n"
            "    Integration start (s) -- an excitation center for an ADC readout, or the\n"
            "    trigger time for an external measurement.\n\n"
            "Returns\n"
            "-------\n"
            "tuple of ndarray\n"
            "    `(t, kx, ky, kz)`, each shaped like `times`. `t` is the query grid in seconds,\n"
            "    after picosecond quantisation.")
        .def(
            "kspace",
            [](const s2w::Sampler& s, InAny<std::int64_t> times, std::int64_t reference) {
                auto ts = flatten_times<std::int64_t>(times, ps_of_raw);
                s2w::Trace tr = s.kspace(ts.first, s2w::Ps(reference));
                return nb::make_tuple(reshape_like(secs_vec(tr.t), ts.second),
                                      reshape_like(std::move(tr.x), ts.second),
                                      reshape_like(std::move(tr.y), ts.second),
                                      reshape_like(std::move(tr.z), ts.second));
            },
            "times"_a, "reference"_a,
            "As above, with times and reference given as exact int64 picoseconds.")
        .def(
            "kspace_flat",
            [](const s2w::Sampler& s, In<double> times, In<std::int64_t> offsets,
               In<double> references) {
                std::vector<s2w::Ps> t;
                t.reserve(times.shape(0));
                for (std::size_t i = 0; i < times.shape(0); ++i) {
                    t.push_back(s2w::Ps::from_secs(times(i)));
                }
                std::vector<std::size_t> off;
                off.reserve(offsets.shape(0));
                for (std::size_t i = 0; i < offsets.shape(0); ++i) {
                    off.push_back(static_cast<std::size_t>(offsets(i)));
                }
                std::vector<s2w::Ps> refs;
                refs.reserve(references.shape(0));
                for (std::size_t i = 0; i < references.shape(0); ++i) {
                    refs.push_back(s2w::Ps::from_secs(references(i)));
                }
                s2w::FlatK k = s.kspace_flat(t, off, refs);
                return nb::make_tuple(to_numpy(std::move(k.kx)), to_numpy(std::move(k.ky)),
                                      to_numpy(std::move(k.kz)));
            },
            "times"_a, "offsets"_a, "references"_a,
            "Batched k-space over many events in one call, returning flat arrays.\n\n"
            "Parameters\n"
            "----------\n"
            "times : ndarray of float64\n"
            "    All events' query times (s), concatenated.\n"
            "offsets : ndarray of int64\n"
            "    Cumulative event boundaries, length `len(references) + 1`: event `e` is\n"
            "    `times[offsets[e]:offsets[e+1]]`.\n"
            "references : ndarray of float64\n"
            "    Per-event integration start (s).\n\n"
            "Returns\n"
            "-------\n"
            "tuple of ndarray\n"
            "    `(kx, ky, kz)`, each the length of `times`. Reuses the one prebuilt sampler, so\n"
            "    this is the fast path for whole-sequence k-space and for delay sweeps.");

    m.def(
        "read_dsv",
        [](const std::string& path) {
            const s2w::DsvChannel ch = s2w::read_dsv_file(path);
            return nb::make_tuple(to_numpy(ch.dense()), ch.raster_time, ch.unit);
        },
        "path"_a,
        "Read a Siemens `.dsv` channel as dense per-sample values.\n\n"
        "Parameters\n"
        "----------\n"
        "path : str\n"
        "    Path to one `.dsv` channel file.\n\n"
        "Returns\n"
        "-------\n"
        "values : ndarray of float64\n"
        "    Per-sample values in physical units (`VERTFACTOR` applied).\n"
        "raster_time : float\n"
        "    Sample spacing (s), from `HORIDELTA`.\n"
        "unit : str\n"
        "    The `VERTUNITNAME` string, e.g. \"mT/m\".\n\n"
        "Raises\n"
        "------\n"
        "ValueError\n"
        "    If a value line is not an integer, or `HORIDELTA`/`VERTFACTOR` is missing.");

    m.def(
        "read_dsv_pwl",
        [](const std::string& path) {
            s2w::DsvChannel ch = s2w::read_dsv_file(path);  // see `gradient`: not const, so
            return nb::make_tuple(secs_array(ch.waveform.t),       // the values move out
                                  to_numpy(std::move(ch.waveform.v)), ch.raster_time, ch.unit);
        },
        "path"_a,
        "Read a Siemens `.dsv` channel as piecewise-linear corners.\n\n"
        "The `[VALUES]` body is a run-length-encoded derivative, so a run of equal deltas is\n"
        "one constant-slope segment and only slope changes are kept. The dense array is never\n"
        "built, so this is both sparser and cheaper than `read_dsv`.\n\n"
        "Parameters\n"
        "----------\n"
        "path : str\n"
        "    Path to one `.dsv` channel file.\n\n"
        "Returns\n"
        "-------\n"
        "corner_times : ndarray of float64\n"
        "    Corner times (s), interval-centered at `(k+0.5)*raster_time`.\n"
        "corner_values : ndarray of float64\n"
        "    Corner values in physical units.\n"
        "raster_time : float\n"
        "    Sample spacing (s).\n"
        "unit : str\n"
        "    The `VERTUNITNAME` string.");

    m.def(
        "read_inf_adc",
        [](const std::string& path) {
            const s2w::Inf inf = s2w::read_inf_file(path);
            return nb::make_tuple(secs_array(inf.adc_sample_times()), inf.adc.size());
        },
        "path"_a,
        "Read ADC sample times from a Siemens INF file.\n\n"
        "Parameters\n"
        "----------\n"
        "path : str\n"
        "    Path to a `*_INF.dsv` file.\n\n"
        "Returns\n"
        "-------\n"
        "sample_times : ndarray of float64\n"
        "    All ADC sample times (s, Siemens frame), concatenated in segment order.\n"
        "num_segments : int\n"
        "    Raw `adc MeasHeader` segment count, *before* the MaxAdcSegmentLength merge.");

    m.def(
        "read_inf_adc_events",
        [](const std::string& path) {
            const s2w::Inf inf = s2w::read_inf_file(path);
            std::vector<nb::object> out;
            for (const s2w::AdcSegment& seg : inf.adc_events()) {
                out.push_back(nb::cast(secs_array(seg.sample_times())));
            }
            return out;
        },
        "path"_a,
        "Read ADC sample times from a Siemens INF, grouped into logical events.\n\n"
        "Siemens splits a long acquisition by `MaxAdcSegmentLength` (5000-sample chunks, say);\n"
        "consecutive segments that are contiguous in time and share a dwell are merged back, so\n"
        "the grouping matches the `.seq`'s per-event ADC. The merge is exact.\n\n"
        "Parameters\n"
        "----------\n"
        "path : str\n"
        "    Path to a `*_INF.dsv` file.\n\n"
        "Returns\n"
        "-------\n"
        "list of ndarray\n"
        "    One float64 array of sample times (s) per logical ADC event.");

    m.def(
        "read_inf_sync",
        [](const std::string& path) {
            const s2w::Inf inf = s2w::read_inf_file(path);
            std::vector<std::tuple<std::string, double, double>> out;
            for (const s2w::SyncEvent& e : inf.triggers()) {
                out.emplace_back(e.name, e.time, e.duration);
            }
            return out;
        },
        "path"_a,
        "Read SYNC-column events from a Siemens INF file.\n\n"
        "Parameters\n"
        "----------\n"
        "path : str\n"
        "    Path to a `*_INF.dsv` file.\n\n"
        "Returns\n"
        "-------\n"
        "list of tuple\n"
        "    `(name, time_s, duration_s)` per event, in block order. Filter by name in the\n"
        "    caller: external triggers are typically `EXTRIG0`.");

    m.def(
        "read_inf_rf",
        [](const std::string& inf_path, const std::string& rfd_path) {
            const s2w::Inf inf = s2w::read_inf_file(inf_path);
            const s2w::DsvChannel rfd = s2w::read_dsv_file(rfd_path);
            const std::vector<s2w::RfEvent> events = inf.rf_events();
            std::vector<std::tuple<double, double, double, double, double, double>> out;
            out.reserve(events.size());
            for (const s2w::RfEvent& e : events) {
                out.emplace_back(e.center(rfd), e.start, e.duration, e.flip_deg, e.freq,
                                 e.phase_deg);
            }
            return out;
        },
        "inf_path"_a, "rfd_path"_a,
        "Read RF events from a Siemens INF, with excitation centers refined from the RFD.\n\n"
        "The center is the TXFlip window middle for symmetric pulses, which is exact. For an\n"
        "asymmetric pulse (min-phase, SLR, adiabatic) the excitation is at the peak instead, so\n"
        "the sub-raster RFD peak is used -- limited by the RFD raster, unlike the exact middle.\n\n"
        "Parameters\n"
        "----------\n"
        "inf_path : str\n"
        "    Path to a `*_INF.dsv` file.\n"
        "rfd_path : str\n"
        "    Path to the matching `*_RFD.dsv` envelope channel.\n\n"
        "Returns\n"
        "-------\n"
        "list of tuple\n"
        "    `(center_s, start_s, duration_s, flip_deg, freq_hz, phase_deg)` per RF event.");

    m.def(
        "inf_lead_in",
        [](const std::string& path) { return s2w::read_inf_file(path).lead_in(); }, "path"_a,
        "The Pulseq-to-Siemens lead-in time (s).\n\n"
        "The first INF event block (index 0) is the Siemens `GLOBzSet` setup; the second (index 1)\n"
        "is the first Pulseq block, and *its* Siemens start time is the lead-in (~1020 us). Add it\n"
        "to a Pulseq-frame time to get the Siemens-frame time, where the DSV waveforms live.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "path : str\n"
        "    Path to a `*_INF.dsv` file.\n\n"
        "Returns\n"
        "-------\n"
        "float\n"
        "    Lead-in time (s), or 0.0 if the INF has no second block.");

}
