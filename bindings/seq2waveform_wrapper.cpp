// nanobind bindings for the seq2waveform C++ core.
//
// Keep this layer thin: call into the core, then hand the results to Python. Large numeric outputs
// move to the heap and are owned by a capsule, so numpy views them without copying.
//
// Times crossing this boundary follow the Rust bindings' convention: float64 seconds, or int64
// picoseconds taken verbatim (bit-exact -- the path for exact delay sweeps). Each such method is
// bound twice so the generated stub carries both as @overload.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <exception>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "seq2waveform/error.hpp"
#include "seq2waveform/model.hpp"
#include "seq2waveform/parser.hpp"
#include "seq2waveform/pwl.hpp"
#include "seq2waveform/time.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using Out = nb::ndarray<nb::numpy, T, nb::ndim<1>>;
template <typename T>
using In = nb::ndarray<const T, nb::ndim<1>, nb::c_contig>;

/// Hand a std::vector to Python without copying: the vector moves to the heap and a capsule owns
/// it, so numpy frees it when the last view goes away. (Rust's `into_pyarray` does this for free;
/// in C++ it is this helper -- route every array return through it.)
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
    return std::vector<double>(a.data(), a.data() + a.shape(0));
}

// --- port-only model introspection ---------------------------------------
//
// `parse_model` and `parse_shapes` exist so the parity suite can compare a parsed sequence against
// the Rust implementation field by field -- without them the only model-level oracle is a block
// count. They mirror the Rust bindings' functions of the same names key for key and field for
// field, so the two dicts must compare equal; keep them in step, and drop both at cut-over, when
// the public API is `Sequence` rather than a dict.

using BlockRow = std::tuple<std::uint32_t, double, std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t>;
using RfRow = std::tuple<double, std::uint32_t, std::uint32_t, std::uint32_t, double, double,
                         double, double, double, double, std::string>;
using TrapRow = std::tuple<double, double, double, double, double>;
using ArbRow = std::tuple<double, double, double, std::uint32_t, std::uint32_t, double>;
using AdcRow =
    std::tuple<std::uint32_t, double, double, double, double, double, double, std::uint32_t>;
using TriggerRow = std::tuple<std::uint32_t, std::uint32_t, double, double>;
using ExtNodeRow = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;

nb::dict parse_model(const std::string& path) {
    const s2w::Sequence seq = s2w::parse_file(path);

    std::vector<BlockRow> blocks;
    blocks.reserve(seq.blocks.size());
    for (const s2w::Block& b : seq.blocks) {
        blocks.emplace_back(b.num, b.duration, b.rf, b.gx, b.gy, b.gz, b.adc, b.ext);
    }

    std::map<std::uint32_t, RfRow> rf;
    for (const auto& entry : seq.rf) {
        const s2w::Rf& r = entry.second;
        rf.emplace(entry.first,
                   RfRow{r.amplitude, r.mag_shape, r.phase_shape, r.time_shape, r.center, r.delay,
                         r.freq_ppm, r.phase_ppm, r.freq, r.phase, s2w::rf_use_name(r.use)});
    }

    // The gradient library splits into two dicts, one per variant, so a trapezoid can never
    // silently compare equal to an arbitrary shape that happens to hold the same numbers.
    std::map<std::uint32_t, TrapRow> traps;
    std::map<std::uint32_t, ArbRow> arbs;
    for (const auto& entry : seq.gradients) {
        if (const s2w::Trap* t = std::get_if<s2w::Trap>(&entry.second)) {
            traps.emplace(entry.first, TrapRow{t->amplitude, t->rise, t->flat, t->fall, t->delay});
        } else {
            const s2w::ArbitraryGrad& a = std::get<s2w::ArbitraryGrad>(entry.second);
            arbs.emplace(entry.first,
                         ArbRow{a.amplitude, a.first, a.last, a.amp_shape, a.time_shape, a.delay});
        }
    }

    std::map<std::uint32_t, AdcRow> adc;
    for (const auto& entry : seq.adc) {
        const s2w::Adc& a = entry.second;
        adc.emplace(entry.first, AdcRow{a.num, a.dwell, a.delay, a.freq_ppm, a.phase_ppm, a.freq,
                                        a.phase, a.phase_shape});
    }

    std::map<std::uint32_t, TriggerRow> triggers;
    for (const auto& entry : seq.triggers) {
        const s2w::Trigger& t = entry.second;
        triggers.emplace(entry.first, TriggerRow{t.trig_type, t.channel, t.delay, t.duration});
    }

    std::map<std::uint32_t, ExtNodeRow> ext_nodes;
    for (const auto& entry : seq.ext_nodes) {
        const s2w::ExtNode& n = entry.second;
        ext_nodes.emplace(entry.first, ExtNodeRow{n.type_id, n.ref_id, n.next_id});
    }

    nb::dict d;
    d["version"] = nb::make_tuple(seq.version.major, seq.version.minor, seq.version.revision);
    d["definitions"] = nb::cast(seq.definitions);
    d["signature"] = nb::cast(seq.signature);
    d["blocks"] = nb::cast(blocks);
    d["rf"] = nb::cast(rf);
    d["traps"] = nb::cast(traps);
    d["arbs"] = nb::cast(arbs);
    d["adc"] = nb::cast(adc);
    d["triggers"] = nb::cast(triggers);
    d["ext_nodes"] = nb::cast(ext_nodes);
    d["ext_type_names"] = nb::cast(seq.ext_type_names);
    return d;
}

nb::dict parse_shapes(const std::string& path) {
    const s2w::Sequence seq = s2w::parse_file(path);
    std::map<std::uint32_t, nb::object> out;
    for (const auto& entry : seq.shapes) {
        std::vector<double> raw = entry.second.data;
        std::vector<double> decompressed = entry.second.decompressed();
        out.emplace(entry.first, nb::make_tuple(entry.second.num_samples, to_numpy(std::move(raw)),
                                                to_numpy(std::move(decompressed))));
    }
    // The map caster builds the dict; borrow it as one so the generated stub says `dict` rather
    // than `object`.
    return nb::borrow<nb::dict>(nb::cast(out));
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() =
        "Compiled core of the seq2waveform C++ port.\n\n"
        "Import the public API from the `seq2waveform_cpp` package rather than from here.";

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
                new (self) s2w::PiecewiseLinear{ps_from_seconds(t_seconds), to_vector(values)};
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
                return s2w::PiecewiseLinear{ps_verbatim(t_ps), to_vector(values)};
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
            "Precompute once, then `moment_to` and `moments_at_sorted` evaluate in O(1) per point.")
        .def(
            "moment_to",
            [](const s2w::PiecewiseLinear& w, double t_seconds, In<double> cum) {
                return w.moment_to(s2w::Ps::from_secs(t_seconds), to_vector(cum));
            },
            "t_seconds"_a, "cum"_a,
            "Integral from the first corner to `t_seconds`, using precomputed `cum`.\n\n"
            "k-space referenced to `t_ref` is `moment_to(t) - moment_to(t_ref)`.")
        .def(
            "moments_at_sorted",
            [](const s2w::PiecewiseLinear& w, In<double> times, In<double> cum) {
                return to_numpy(w.moments_at_sorted(ps_from_seconds(times), to_vector(cum)));
            },
            "times"_a, "cum"_a,
            "`moment_to` over non-decreasing `times` (float64 seconds) with one forward sweep.")
        .def(
            "moments_at_sorted",
            [](const s2w::PiecewiseLinear& w, In<std::int64_t> times, In<double> cum) {
                return to_numpy(w.moments_at_sorted(ps_verbatim(times), to_vector(cum)));
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

    // A malformed line raises ValueError, as the Rust bindings' `err()` does, so the parity suite
    // can compare the exception type and its text and not just the fact that both failed.
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) {
            try {
                std::rethrow_exception(p);
            } catch (const s2w::ParseError& e) {
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
        "    Path to a Pulseq `.seq` file (format v1.4/1.5).\n\n"
        "Returns\n"
        "-------\n"
        "int\n"
        "    Number of rows in the `[BLOCKS]` table.\n\n"
        "Raises\n"
        "------\n"
        "ValueError\n"
        "    If a line is malformed; the message names the source line.");

    m.def("parse_model", &parse_model, "path"_a,
          "Dump a parsed `.seq` model as plain Python data.\n\n"
          "Port scaffolding, not public API: it exists so the port's parity suite can compare the\n"
          "parsed sequence against the Rust implementation field by field. Shapes are excluded --\n"
          "see `parse_shapes`.\n\n"
          "Parameters\n"
          "----------\n"
          "path : str\n"
          "    Path to a Pulseq `.seq` file.\n\n"
          "Returns\n"
          "-------\n"
          "dict\n"
          "    Keys `version`, `definitions`, `signature`, `blocks`, `rf`, `traps`, `arbs`,\n"
          "    `adc`, `triggers`, `ext_nodes` and `ext_type_names`. The event libraries are\n"
          "    `{id: tuple}` in the field order of the corresponding struct; `blocks` is a list\n"
          "    of `(num, duration_s, rf, gx, gy, gz, adc, ext)`. All times are in seconds.");

    m.def("parse_shapes", &parse_shapes, "path"_a,
          "Dump the `[SHAPES]` library of a `.seq` file.\n\n"
          "Port scaffolding, like `parse_model`: kept separate because the arrays are large enough\n"
          "that round-tripping them through Python lists would dominate the parity run.\n\n"
          "Parameters\n"
          "----------\n"
          "path : str\n"
          "    Path to a Pulseq `.seq` file.\n\n"
          "Returns\n"
          "-------\n"
          "dict\n"
          "    `{shape_id: (num_samples, raw, decompressed)}`, where `raw` is the on-disk (often\n"
          "    run-length-compressed) data and `decompressed` is the expanded sample vector.");
}
