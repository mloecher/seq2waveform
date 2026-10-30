"""Differential tests: the C++ port must agree with the Rust implementation, bit for bit.

Run:  pixi run test-parity   (installs both extensions into this environment first)

This is the port's main safety net. The Rust crate is already validated against the Pulseq h5 and
the Siemens DSV, so for every module we port we can assert the C++ result equals the Rust result
on the real datasets, rather than re-deriving ground truth. Exact equality is the bar: both
implementations do the same arithmetic in the same order on int64 picosecond times, so any
difference is a porting bug, not float noise.

The Rust checkout supplies both the oracle and the test corpus (`testdata/` is gitignored, so it
exists only there). It is located via S2W_RUST_REPO, defaulting to the sibling `../seq2waveform`.
A run that reports *skipped* found neither -- treat that as a failure, not a pass.

As modules land, extend this file alongside PORTING.md's checklist.
"""

import functools
import os
from pathlib import Path

import numpy as np
import pytest

import seq2waveform_cpp as cpp

rust = pytest.importorskip(
    "seq2waveform",
    reason="the Rust oracle is not installed in this environment -- run `pixi run build-oracle`",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(raw: str) -> Path:
    """Resolve a configured path, taking relative values against the project root (not cwd)."""
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


RUST_REPO = _resolve(os.environ.get("S2W_RUST_REPO", "../seq2waveform"))
TESTDATA = _resolve(os.environ.get("S2W_TESTDATA", str(RUST_REPO / "testdata")))

# Cap the sample count per dataset: EPI has millions of ADC samples and the point here is
# agreement, not throughput.
MAX_SAMPLES = 200_000


def _datasets():
    if not TESTDATA.exists():
        return []
    return sorted(
        d.name for d in TESTDATA.iterdir() if d.is_dir() and (d / f"{d.name}.seq").exists()
    )


DATASETS = _datasets()
pytestmark = pytest.mark.skipif(
    not DATASETS, reason=f"no datasets found under {TESTDATA} (set S2W_TESTDATA)"
)


@pytest.fixture(params=DATASETS)
def name(request):
    return request.param


@pytest.fixture
def seq(name):
    return rust.Sequence(str(TESTDATA / name / f"{name}.seq"))


def _query_times(seq):
    """A non-decreasing time vector to sample both implementations at."""
    t = np.asarray(seq.adc_times())
    if t.size == 0:
        pytest.skip("dataset has no ADC events")
    return np.sort(t)[:MAX_SAMPLES]


def _axes(seq):
    """Gradient axes that actually carry corners in this dataset."""
    return [ax for ax in "xyz" if len(seq.gradient(ax)[0]) > 1]


def test_corner_times_roundtrip_exactly(seq):
    """Rebuilding a C++ waveform from Rust's seconds recovers the same picosecond grid.

    The ports only agree bit-for-bit if the shared time grid is identical, so this guards the
    float-seconds hop that the Python boundary forces on us.
    """
    for ax in _axes(seq):
        ct, _ = seq.gradient(ax)
        w = cpp.PiecewiseLinear(np.asarray(ct), np.zeros_like(ct))
        expected = cpp.ps_from_secs(np.asarray(ct))
        assert np.array_equal(w.corner_times_ps(), expected), f"g{ax}: corner grid shifted"


def test_gradient_sampling_matches_rust(seq):
    """C++ sample_at_sorted == Rust Sampler.gradients on every axis, exactly."""
    times = _query_times(seq)
    _, *rust_axes = seq.sampler().gradients(times)
    for ax, rust_vals in zip("xyz", rust_axes):
        ct, cv = seq.gradient(ax)
        w = cpp.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        cpp_vals = w.sample_at_sorted(times)
        assert np.array_equal(cpp_vals, np.asarray(rust_vals)), f"g{ax}: sampled values differ"


def test_kspace_matches_rust(seq):
    """C++ moments (minus the reference moment) == Rust Sampler.kspace, exactly.

    Covers cumulative_moment + moments_at_sorted, i.e. the k-space fast path.
    """
    times = _query_times(seq)
    specs = seq.adc_specs()
    if not specs:
        pytest.skip("dataset has no ADC specs")
    reference = specs[0][1]

    _, *rust_axes = seq.sampler().kspace(times, reference)
    for ax, rust_vals in zip("xyz", rust_axes):
        ct, cv = seq.gradient(ax)
        w = cpp.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        cum = w.cumulative_moment()
        cpp_vals = w.moments_at_sorted(times, cum) - w.moment_to(reference, cum)
        assert np.array_equal(cpp_vals, np.asarray(rust_vals)), f"k{ax}: k-space differs"


def test_unsorted_path_matches_sorted_path(seq):
    """sample_at (binary search) and sample_at_sorted (forward sweep) agree.

    The sweep is only taken when times are non-decreasing; this pins the two paths together so a
    future optimisation of either cannot drift.
    """
    times = _query_times(seq)[:10_000]
    for ax in _axes(seq):
        ct, cv = seq.gradient(ax)
        w = cpp.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        assert np.array_equal(w.sample_at(times), w.sample_at_sorted(times)), f"g{ax}: paths differ"


def test_picosecond_times_are_taken_verbatim(seq):
    """Sampling at int64 picoseconds equals sampling at the equivalent float seconds.

    The exact path exists for delay sweeps; this checks it is wired to the same arithmetic.
    """
    times = _query_times(seq)[:10_000]
    times_ps = cpp.ps_from_secs(times)
    for ax in _axes(seq):
        ct, cv = seq.gradient(ax)
        w = cpp.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        assert np.array_equal(w.sample_at_sorted(times_ps), w.sample_at_sorted(times)), (
            f"g{ax}: picosecond and seconds paths differ"
        )


# --- the parser (port step 2) --------------------------------------------------------------
#
# `parse_model` and `parse_shapes` are port scaffolding on both sides, reached through the private
# extension module because neither package re-exports them. They are the only way to compare the
# event libraries field by field: the public surface stops at a block count.

rust_model = rust._seq2waveform.parse_model
rust_shapes = rust._seq2waveform.parse_shapes
cpp_model = cpp._core.parse_model
cpp_shapes = cpp._core.parse_shapes

# Every key both dumps must carry. Asserted on each side, so adding a field to one reader without
# the other fails here rather than going unnoticed.
MODEL_KEYS = {
    "version",
    "definitions",
    "signature",
    "blocks",
    "rf",
    "traps",
    "arbs",
    "adc",
    "triggers",
    "ext_nodes",
    "ext_type_names",
}
# The id-keyed event libraries, compared entry by entry for a legible failure.
LIBRARIES = ("rf", "traps", "arbs", "adc", "triggers", "ext_nodes")


@pytest.fixture
def seq_path(name):
    return str(TESTDATA / name / f"{name}.seq")


def test_block_count_matches_rust(seq_path):
    """The `[BLOCKS]` row count agrees -- the coarsest check that the walk saw the same file."""
    assert cpp.parse_file(seq_path) == rust.parse_file(seq_path)


def test_parsed_model_matches_rust(seq_path):
    """Every section the parser fills, compared field by field.

    Exact equality on the floats: both readers scan the same decimal text with a correctly rounded
    parser and apply the same unit scale in the same order, so even a one-ulp difference is a
    porting bug -- most likely in the scanner or in the order of a multiplication.
    """
    got = cpp_model(seq_path)
    want = rust_model(seq_path)
    assert set(got) == MODEL_KEYS, "the C++ dump gained or lost a key"
    assert set(want) == MODEL_KEYS, "the Rust dump gained or lost a key"

    assert got["version"] == want["version"]
    assert got["definitions"] == want["definitions"]
    assert got["signature"] == want["signature"]
    assert got["ext_type_names"] == want["ext_type_names"]

    assert len(got["blocks"]) == len(want["blocks"]), "different number of blocks"
    for i, (g, w) in enumerate(zip(got["blocks"], want["blocks"])):
        assert g == w, f"block {i} differs"

    for key in LIBRARIES:
        assert set(got[key]) == set(want[key]), f"{key}: different ids"
        for i in sorted(got[key]):
            assert got[key][i] == want[key][i], f"{key}[{i}] differs"


def test_shapes_match_rust(seq_path):
    """`[SHAPES]` entries agree, both as stored and once decompressed.

    The decompressed half is the real target: the run-length-on-derivative expansion and the
    cumulative sum that follows it are where a port drifts, and the sum's result depends on the
    accumulation order.
    """
    got = cpp_shapes(seq_path)
    want = rust_shapes(seq_path)
    assert set(got) == set(want), "different shape ids"
    for i in sorted(got):
        got_n, got_raw, got_dec = got[i]
        want_n, want_raw, want_dec = want[i]
        assert got_n == want_n, f"shape {i}: num_samples differs"
        assert np.array_equal(got_raw, want_raw), f"shape {i}: stored data differs"
        assert np.array_equal(got_dec, want_dec), f"shape {i}: decompressed samples differ"


def test_parse_error_matches_rust(tmp_path):
    """A malformed row fails the same way from both readers: same exception, same text.

    This is the line counter end to end. Blank and comment lines count toward the number (Rust
    enumerates before skipping them), the row parser names the offending field, and both sides
    surface it as a ValueError -- so the message can be compared verbatim.
    """
    bad = tmp_path / "bad.seq"
    bad.write_text("# comment\n\n[TRAP]\n\n1 100 x 6000 200 0\n")

    with pytest.raises(ValueError) as cpp_err:
        cpp.parse_file(str(bad))
    with pytest.raises(ValueError) as rust_err:
        rust.parse_file(str(bad))

    assert str(cpp_err.value) == str(rust_err.value)
    assert "line 5" in str(cpp_err.value), "the blank and comment lines were not counted"


# --- waveform expansion (port step 3) ------------------------------------------------------
#
# These compare the C++ Sequence against the Rust one directly, rather than feeding Rust's output
# into a C++ PiecewiseLinear as the step-1 tests above do. Everything in step 3 lands in
# `gradient()`'s two arrays: the block walk, the trapezoid corners, restoreAdditionalShapeSamples,
# the cross-block stitch and the DSV resampling.

GRAD_MODES = ("pulseq", "centers", "dsv")


@pytest.mark.parametrize("mode", GRAD_MODES)
def test_gradient_corners_match_rust(seq_path, mode):
    """Every gradient channel, in every mode, corner for corner.

    Exact equality: both ports accumulate the same int64 picosecond times in the same order, and
    the corner *count* is checked first because a reconstruction that drops or adds a corner is the
    most likely porting failure and the least readable as an array diff.
    """
    r = rust.Sequence(seq_path, mode)
    c = cpp.Sequence(seq_path, mode)
    assert c.num_blocks == r.num_blocks, "block count differs"
    assert c.gradient_mode == r.gradient_mode == mode

    corners = 0
    for ax in "xyz":
        rt, rv = r.gradient(ax)
        ct, cv = c.gradient(ax)
        assert len(ct) == len(rt), f"g{ax} [{mode}]: {len(ct)} corners vs Rust's {len(rt)}"
        assert np.array_equal(ct, rt), f"g{ax} [{mode}]: corner times differ"
        assert np.array_equal(cv, rv), f"g{ax} [{mode}]: corner values differ"
        corners += len(ct)
    # Guard against a vacuous pass: a dataset with no gradient corners at all would compare equal
    # trivially, and silently stop testing the thing this step is about.
    assert corners > 0, f"[{mode}] no gradient corners on any axis -- nothing was compared"


def test_adc_times_match_rust(seq_path):
    """Flattened ADC sample times, which pin the block walk's accumulated time base."""
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    assert np.array_equal(c.adc_times(), r.adc_times()), "ADC sample times differ"


def test_adc_events_match_rust(seq_path):
    """The per-event ADC split, with each event's freq/phase offsets.

    `adc_times` and `adc_events` are separate focused walks on both sides, so comparing both
    catches one of them drifting from the other.
    """
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    rust_meta = r.adc_events_meta()
    cpp_meta = c.adc_events_meta()
    assert len(cpp_meta) == len(rust_meta), "number of ADC events differs"
    for i, (want, got) in enumerate(zip(rust_meta, cpp_meta)):
        assert np.array_equal(got[0], want[0]), f"ADC event {i}: sample times differ"
        assert got[1] == want[1], f"ADC event {i}: freq differs"
        assert got[2] == want[2], f"ADC event {i}: phase differs"


def test_rf_events_match_rust(seq_path):
    """RF centers, offsets and `use` names.

    The `use` name matters: both ports deliberately keep the real use rather than folding
    undefined pulses into excitation, so a port that "helpfully" relabelled them would fail here.
    """
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    assert c.rf_events() == r.rf_events(), "RF events differ"


def test_triggers_match_rust(seq_path):
    """Resolved trigger events -- the extension linked-list walk."""
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    assert c.triggers() == r.triggers(), "trigger events differ"


# --- the sampler and k-space (port step 4) --------------------------------------------------


def test_duration_matches_rust(seq_path):
    """Summed block durations, accumulated in the same order so the f64 sum is identical."""
    assert cpp.Sequence(seq_path).duration == rust.Sequence(seq_path).duration


def test_sampler_gradients_match_rust(seq_path):
    """Sampler.gradients at the ADC times, all three axes."""
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    times = _query_times(r)
    _, rx, ry, rz = r.sampler().gradients(times)
    cx, cy, cz = c.sampler().gradients(times)
    for ax, want, got in (("x", rx, cx), ("y", ry, cy), ("z", rz, cz)):
        assert np.array_equal(got, np.asarray(want)), f"g{ax}: sampled values differ"


def test_sampler_kspace_match_rust(seq_path):
    """Sampler.kspace against the Rust sampler, referenced to the first ADC's excitation."""
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    times = _query_times(r)
    specs = r.adc_specs()
    if not specs:
        pytest.skip("dataset has no ADC specs")
    reference = specs[0][1]
    _, rx, ry, rz = r.sampler().kspace(times, reference)
    cx, cy, cz = c.sampler().kspace(times, reference)
    for ax, want, got in (("x", rx, cx), ("y", ry, cy), ("z", rz, cz)):
        assert np.array_equal(got, np.asarray(want)), f"k{ax}: k-space differs"


def test_adc_specs_and_excitations_match_rust(seq_path):
    """The ADC-to-excitation matching: which excitation each readout integrates from.

    This is where an off-by-one would be silent and wrong -- a readout referenced to the *next*
    excitation still produces plausible-looking k-space.
    """
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    rs, cs = r.adc_specs(), c.adc_specs()
    assert len(cs) == len(rs), "number of ADC specs differs"
    for i, (want, got) in enumerate(zip(rs, cs)):
        assert np.array_equal(got[0], want[0]), f"spec {i}: times differ"
        assert got[1] == want[1], f"spec {i}: k-space reference differs"
    # NaN-aware, since an ADC with no preceding excitation reports (nan, nan, nan).
    assert np.array_equal(
        np.asarray(c.adc_excitations(), dtype=float),
        np.asarray(r.adc_excitations(), dtype=float),
        equal_nan=True,
    ), "matched excitations differ"


def test_adc_kspace_flat_matches_rust(seq_path):
    """The batched flat path, including its event offsets."""
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    rt, rkx, rky, rkz, roff = r.adc_kspace_flat()
    ct, ckx, cky, ckz, coff = c.adc_kspace_flat()
    assert np.array_equal(coff, np.asarray(roff)), "event offsets differ"
    assert np.array_equal(ct, np.asarray(rt)), "flat times differ"
    for ax, want, got in (("x", rkx, ckx), ("y", rky, cky), ("z", rkz, ckz)):
        assert np.array_equal(got, np.asarray(want)), f"flat k{ax} differs"


def test_flat_path_matches_per_event_path(seq_path):
    """The C++ flat and per-event k-space paths agree with each other.

    Both ports keep two implementations of the same quantity; this pins the C++ pair together so a
    future optimisation of either cannot drift, independently of the Rust comparison.
    """
    c = cpp.Sequence(seq_path)
    t, kx, ky, kz, off = c.adc_kspace_flat()
    per = c.adc_kspace()
    assert len(off) == len(per) + 1, "offsets and event count disagree"
    for e, (et, ex, ey, ez) in enumerate(per):
        a, b = off[e], off[e + 1]
        assert np.array_equal(t[a:b], et), f"event {e}: times differ"
        assert np.array_equal(kx[a:b], ex), f"event {e}: kx differs"
        assert np.array_equal(ky[a:b], ey), f"event {e}: ky differs"
        assert np.array_equal(kz[a:b], ez), f"event {e}: kz differs"


def test_dense_gradients_match_rust(seq_path):
    """The uniform-raster dense sampling, which is what a plot or an h5 comparison uses."""
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    raster = 1e-4
    rt, rx, ry, rz = r.gradients_dense(raster)
    ct, cx, cy, cz = c.gradients_dense(raster)
    assert np.array_equal(ct, np.asarray(rt)), "dense times differ"
    for ax, want, got in (("x", rx, cx), ("y", ry, cy), ("z", rz, cz)):
        assert np.array_equal(got, np.asarray(want)), f"dense g{ax} differs"
    for ax in "xyz":
        at, av = r.gradient_axis_dense(ax, raster)
        bt, bv = c.gradient_axis_dense(ax, raster)
        assert np.array_equal(bt, np.asarray(at)), f"axis-dense g{ax} times differ"
        assert np.array_equal(bv, np.asarray(av)), f"axis-dense g{ax} values differ"


def test_trigger_sampling_matches_rust(seq_path):
    """Trigger-based (Skope) sampling, which references the trigger rather than an excitation."""
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    if not r.triggers():
        pytest.skip("dataset has no triggers")
    delay, dt, n = 0.0, 10e-6, 64
    rspecs, cspecs = r.trigger_specs(delay, dt, n), c.trigger_specs(delay, dt, n)
    assert len(cspecs) == len(rspecs), "number of trigger specs differs"
    for i, (want, got) in enumerate(zip(rspecs, cspecs)):
        assert np.array_equal(got[0], want[0]), f"trigger spec {i}: times differ"
        assert got[1] == want[1], f"trigger spec {i}: reference differs"

    rt, rkx, rky, rkz, roff = r.trigger_kspace_flat(delay, dt, n)
    ct, ckx, cky, ckz, coff = c.trigger_kspace_flat(delay, dt, n)
    assert np.array_equal(coff, np.asarray(roff)), "trigger offsets differ"
    assert np.array_equal(ct, np.asarray(rt)), "trigger times differ"
    for ax, want, got in (("x", rkx, ckx), ("y", rky, cky), ("z", rkz, ckz)):
        assert np.array_equal(got, np.asarray(want)), f"trigger k{ax} differs"


def test_nd_time_arrays_keep_their_shape(seq_path):
    """An N-D query comes back shaped like the query, with the same values as the flat form."""
    r = rust.Sequence(seq_path)
    c = cpp.Sequence(seq_path)
    flat = _query_times(r)[: 8 * 16]
    if flat.size < 8 * 16:
        pytest.skip("not enough ADC samples for a 2-D query")
    grid = flat.reshape(8, 16)
    s = c.sampler()
    gx_flat = s.gradients(flat)[0]
    gx_grid = s.gradients(grid)[0]
    assert gx_grid.shape == (8, 16), f"expected (8, 16), got {gx_grid.shape}"
    assert np.array_equal(gx_grid.ravel(), gx_flat), "N-D and flat results differ"


def test_unreferenced_adc_matches_rust(tmp_path):
    """An ADC with no excitation before it: both ports must agree on the sentinel behaviour.

    No dataset reaches this -- every ADC in all five corpora has a preceding excitation -- so
    without a synthetic sequence the branch is untested on both sides. The reference becomes a
    far-past sentinel, so k-space integrates from the start of the waveform, and the matched
    excitation is reported as NaN rather than as a plausible wrong number.
    """
    seq_text = (
        "[DEFINITIONS]\n"
        "GradientRasterTime 1e-05\n"
        "BlockDurationRaster 1e-05\n"
        "[BLOCKS]\n"
        " 1 100 0 0 0 1 1 0\n"
        "[TRAP]\n"
        "1 100 100 800 100 0\n"
        "[ADC]\n"
        "1 3 1000 0 0 0 0 0 0\n"
    )
    p = tmp_path / "no_rf.seq"
    p.write_text(seq_text)

    r = rust.Sequence(str(p))
    c = cpp.Sequence(str(p))
    assert np.array_equal(
        np.asarray(c.adc_excitations(), dtype=float),
        np.asarray(r.adc_excitations(), dtype=float),
        equal_nan=True,
    ), "matched excitations differ for an unreferenced ADC"

    rs, cs = r.adc_specs(), c.adc_specs()
    assert len(cs) == len(rs) == 1
    assert cs[0][1] == rs[0][1], "sentinel reference differs"

    _, rkx, rky, rkz, _ = r.adc_kspace_flat()
    _, ckx, cky, ckz, _ = c.adc_kspace_flat()
    for ax, want, got in (("x", rkx, ckx), ("y", rky, cky), ("z", rkz, ckz)):
        assert np.array_equal(got, np.asarray(want)), f"k{ax} differs for an unreferenced ADC"



# --- the Siemens .dsv reader (port step 5) --------------------------------------------------
#
# Parameterised over the .dsv channels in the corpus rather than just the three gradients. There
# are 14 kinds and they cover a wide spread: rasters of 1, 5 and 10 us, and units of mT/m,
# "mT/m * ms", Volt, Degree, T/s and "-". Two of them (CON, ERR) carry no VERTFACTOR at all and
# are rejected by both readers -- the rejection is compared too, since agreeing on what to refuse
# is as much a part of the port as agreeing on what to decode.
#
# Sizes force a split, and the numbers are worth recording. The corpus holds 70 non-INF channels
# over ~260 MB, and a `cones` channel decodes to 63.5 MILLION samples -- one dense reconstruction
# is a 508 MB array, so asking for one per channel per implementation exhausts memory long before
# patience. So: corners (the decode itself, which streams) for every channel; the dense
# reconstruction for one representative channel per kind, which still exercises `value_at` over
# every raster and unit in the corpus.

DSV_SIZE_LIMIT = 2_000_000


def _dsv_channels():
    if not TESTDATA.exists():
        return []
    return sorted(
        f
        for d in TESTDATA.iterdir()
        if d.is_dir() and (d / "dsv").is_dir()
        for f in (d / "dsv").glob("*.dsv")
        if not f.name.endswith("_INF.dsv")  # not a waveform channel; see the rejection test
    )


def _kind(p):
    return p.stem.replace("SimulationProtocol_", "")


def _dsv_id(p):
    return f"{p.parent.parent.name}/{_kind(p)}"


_ALL = _dsv_channels()
DSV_SMALL = [f for f in _ALL if f.stat().st_size < DSV_SIZE_LIMIT]
DSV_LARGE = [f for f in _ALL if f.stat().st_size >= DSV_SIZE_LIMIT]
# The smallest file of each kind: one per raster/unit combination, for the expensive checks.
DSV_PER_KIND = [
    min((f for f in _ALL if _kind(f) == k), key=lambda p: p.stat().st_size)
    for k in sorted({_kind(f) for f in _ALL})
]


@pytest.fixture(params=DSV_SMALL, ids=_dsv_id)
def dsv_small(request):
    return str(request.param)


@pytest.fixture(params=DSV_LARGE, ids=_dsv_id)
def dsv_large(request):
    return str(request.param)


@pytest.fixture(params=DSV_PER_KIND, ids=_dsv_id)
def dsv_kind(request):
    return str(request.param)


def _compare_corners(path):
    """Compare the decoded corners, or the rejection if Rust refuses the channel.

    Returns the corner count, or -1 when both readers rejected it.
    """
    try:
        rt, rv, rraster, runit = rust.read_dsv_pwl(path)
    except ValueError as exc:
        with pytest.raises(ValueError) as cpp_exc:
            cpp.read_dsv_pwl(path)
        assert str(cpp_exc.value) == str(exc), "rejection messages differ"
        return -1

    ct, cv, craster, cunit = cpp.read_dsv_pwl(path)
    assert craster == rraster, "raster_time differs"
    assert cunit == runit, "unit differs"
    assert len(ct) == len(rt), f"{len(ct)} corners vs Rust's {len(rt)}"
    assert np.array_equal(ct, rt), "corner times differ"
    assert np.array_equal(cv, rv), "corner values differ"
    return len(ct)


@pytest.mark.skipif(not DSV_SMALL, reason="no .dsv channels in the corpus")
def test_dsv_corners_match_rust(dsv_small):
    """The decode itself: run-length expansion, slope-change detection, VERTFACTOR division.

    Exact equality -- the accumulator is an integer, so there is nowhere for a tolerance to hide.
    """
    assert _compare_corners(dsv_small) != 0, "decoded zero corners; nothing was compared"


@pytest.mark.skipif(not DSV_LARGE, reason="no large .dsv channels in the corpus")
def test_dsv_large_channel_corners_match_rust(dsv_large):
    """The same decode on the multi-megabyte channels.

    These are where a streaming bug would show: `cones` runs to 63.5 M samples, and the whole
    point of decoding to corners is never materialising that.
    """
    assert _compare_corners(dsv_large) != 0, "decoded zero corners; nothing was compared"


@pytest.mark.skipif(not DSV_PER_KIND, reason="no .dsv channels in the corpus")
def test_dsv_dense_matches_rust(dsv_kind):
    """The dense reconstruction: the decode plus `value_at` at every sample center."""
    try:
        rvals, rraster, runit = rust.read_dsv(dsv_kind)
    except ValueError:
        pytest.skip("both readers reject this kind; compared in test_dsv_corners_match_rust")
    cvals, craster, cunit = cpp.read_dsv(dsv_kind)
    assert craster == rraster and cunit == runit
    assert len(cvals) == len(rvals), f"{len(cvals)} samples vs Rust's {len(rvals)}"
    assert np.array_equal(cvals, rvals), "dense values differ"


@pytest.mark.skipif(not DSV_PER_KIND, reason="no .dsv channels in the corpus")
def test_dsv_corners_reproduce_the_dense_samples(dsv_kind):
    """Sampling the decoded corners at `(k+0.5)*raster` reproduces the dense samples exactly.

    The decoder's own invariant, independent of Rust: a run of equal deltas is a straight line and
    interpolating a line gives the line back, so the corner form loses nothing relative to the
    dense one -- which is the claim that justifies never building the dense array.

    Resampled through `PiecewiseLinear`, not `np.interp`, and that matters: `PiecewiseLinear`
    interpolates from integer picosecond times, exactly as `dense()` does internally, so this can
    assert bit equality. `np.interp` works in float seconds and disagrees in the last few digits
    wherever the slope is steep, which would force a tolerance and turn a sharp test into a vague
    one.
    """
    try:
        dense, raster, _ = cpp.read_dsv(dsv_kind)
    except ValueError:
        pytest.skip("both readers reject this kind")
    if dense.size == 0:
        pytest.skip("empty channel")
    ct, cv, _, _ = cpp.read_dsv_pwl(dsv_kind)
    w = cpp.PiecewiseLinear(ct, cv)
    times = (np.arange(dense.size, dtype=np.float64) + 0.5) * raster
    assert np.array_equal(w.sample_at_sorted(times), dense), "corners do not reproduce the samples"


def test_dsv_rejects_the_inf_identically():
    """The INF is not a waveform channel, and both readers must say so the same way.

    It has `[SYNCTYPE]`/`[INFO]` and an ASCII event table instead of `[DEFINITIONS]`/`[VALUES]`,
    so the reader runs to EOF and then fails on the missing `VERTFACTOR`. Checked on the smallest
    INF only: they run to 142 MB, and reading one is the whole cost of this test.
    """
    infs = sorted(
        (f for d in TESTDATA.iterdir() if d.is_dir() and (d / "dsv").is_dir()
         for f in (d / "dsv").glob("*_INF.dsv")),
        key=lambda p: p.stat().st_size,
    )
    if not infs:
        pytest.skip("no INF files in the corpus")
    path = str(infs[0])
    with pytest.raises(ValueError) as rust_exc:
        rust.read_dsv(path)
    with pytest.raises(ValueError) as cpp_exc:
        cpp.read_dsv(path)
    assert str(cpp_exc.value) == str(rust_exc.value), "rejection messages differ"
    assert "VERTFACTOR" in str(cpp_exc.value)


# --- the Siemens INF reader (port step 6) ---------------------------------------------------
#
# The INFs are the largest files in the corpus (142 MB for `cones`) and parsing one costs ~1.5 s in
# C++ and ~4 s against the debug oracle, so the results are cached per dataset and the tests read
# from the cache. Without that, five test functions re-parsing both sides would be ~80 s on its
# own; with it, ~27 s.


@functools.lru_cache(maxsize=None)
def _inf_pair(name):
    """Every INF-derived quantity for one dataset, from both readers. Parsed once each."""
    d = TESTDATA / name / "dsv"
    inf = next(iter(sorted(d.glob("*_INF.dsv"))), None)
    rfd = next(iter(sorted(d.glob("*_RFD.dsv"))), None)
    if inf is None:
        return None
    ip = str(inf)
    out = {
        "rust": {
            "adc": rust.read_inf_adc(ip),
            "events": rust.read_inf_adc_events(ip),
            "sync": rust.read_inf_sync(ip),
            "lead_in": rust.inf_lead_in(ip),
            "rf": rust.read_inf_rf(ip, str(rfd)) if rfd else None,
        },
        "cpp": {
            "adc": cpp.read_inf_adc(ip),
            "events": cpp.read_inf_adc_events(ip),
            "sync": cpp.read_inf_sync(ip),
            "lead_in": cpp.inf_lead_in(ip),
            "rf": cpp.read_inf_rf(ip, str(rfd)) if rfd else None,
        },
    }
    return out


@pytest.fixture
def inf(name):
    pair = _inf_pair(name)
    if pair is None:
        pytest.skip("dataset has no INF file")
    return pair


def test_inf_adc_times_match_rust(inf):
    """All ADC sample times and the raw segment count, before the merge."""
    (rt, rn), (ct, cn) = inf["rust"]["adc"], inf["cpp"]["adc"]
    assert cn == rn, f"{cn} raw segments vs Rust's {rn}"
    assert len(ct) == len(rt), f"{len(ct)} sample times vs Rust's {len(rt)}"
    assert np.array_equal(ct, rt), "ADC sample times differ"
    assert len(ct) > 0, "no ADC samples found; nothing was compared"


def test_inf_adc_event_grouping_matches_rust(inf):
    """The MaxAdcSegmentLength merge: which chunks belong to the same logical readout.

    This is the part that has to agree with the `.seq`'s per-event grouping, and the corpus does
    exercise it -- `sync_osc` merges 20 raw segments into 5 events and `girf` 1200 into 300.
    """
    rev, cev = inf["rust"]["events"], inf["cpp"]["events"]
    assert len(cev) == len(rev), f"{len(cev)} logical events vs Rust's {len(rev)}"
    for i, (want, got) in enumerate(zip(rev, cev)):
        assert len(got) == len(want), f"event {i}: {len(got)} samples vs {len(want)}"
        assert np.array_equal(got, want), f"event {i}: sample times differ"


def test_inf_sync_events_match_rust(inf):
    """SYNC-column events: name, time and duration."""
    assert inf["cpp"]["sync"] == inf["rust"]["sync"], "SYNC events differ"


def test_inf_lead_in_matches_rust(inf):
    """The Pulseq-to-Siemens lead-in, which every DSV-frame comparison depends on."""
    assert inf["cpp"]["lead_in"] == inf["rust"]["lead_in"]
    # It is ~1020 us on every dataset in this corpus; a zero would mean the second block was
    # never found, which would silently shift every DSV comparison.
    assert inf["cpp"]["lead_in"] > 0.0, "lead-in is zero; the second event block was not found"


def test_inf_rf_events_match_rust(inf):
    """RF events, including the excitation center refined from the RFD envelope.

    The center is the fiddliest number in the INF: the window middle for a symmetric pulse, the
    sub-raster RFD peak (plateau midpoint or parabolic refinement) for an asymmetric one.
    """
    rrf, crf = inf["rust"]["rf"], inf["cpp"]["rf"]
    if rrf is None:
        pytest.skip("dataset has no RFD channel")
    assert len(crf) == len(rrf), f"{len(crf)} RF events vs Rust's {len(rrf)}"
    for i, (want, got) in enumerate(zip(rrf, crf)):
        assert got == want, f"RF event {i} differs: {got} vs {want}"


def _write_rfd(path, deltas, horidelta_us=1.0, vertfactor=1.0):
    """A minimal RFD channel whose dense samples are cumsum(deltas)/vertfactor."""
    body = "\n".join(str(d) for d in deltas)
    path.write_text(
        "[DEFINITIONS]\n"
        f"HORIDELTA={horidelta_us:.6f}\n"
        f"VERTFACTOR={vertfactor:.6f}\n"
        "VERTUNITNAME=Volt\n"
        "[VALUES]\n" + body + "\n;  end of graphic data file\n"
    )


def _write_inf(path, flip_deg=90.0, dur_us=9.0):
    """A minimal INF with one event block holding one RF row at Rel.Time 0."""
    path.write_text(
        "* EventBlock 0 20\n"
        "| Rel.Time | TXFlip/Dur | Freq/Phase | SYNC/Dur |\n"
        f"|        0 | [sX]: {flip_deg}/{dur_us} |  |  |\n"
    )


@pytest.mark.parametrize(
    "case,deltas,expect_center_us",
    [
        # A sharp peak near the end of the window with unequal neighbours, so the parabolic
        # refinement lands strictly between raster samples. cumsum = [0,1,3,6,10,15,21,30,10]:
        # peak 30 at 7.5 us, vl=21, vr=10, denom=21-60+10=-29, p=0.5*(21-10)/-29 -> 7.310344... us.
        ("sharp asymmetric peak", [0, 1, 2, 3, 4, 5, 6, 9, -20], 7.310344827586207),
        # A two-sample flat maximum, off-centre: cumsum = [0,2,5,7,8,10,10,4,0], so the plateau
        # spans 5.5..6.5 us and that branch returns its midpoint, 6.0 us.
        ("off-centre plateau", [0, 2, 3, 2, 1, 2, 0, -6, -4], 6.0),
    ],
)
def test_asymmetric_rf_center_matches_rust(tmp_path, case, deltas, expect_center_us):
    """The asymmetric RF-center refinement, which no dataset in the corpus reaches.

    Every pulse in all five corpora is symmetric (measured: zero off-centre pulses), so the
    sub-raster RFD peak path -- plateau midpoint, or parabolic interpolation of the neighbouring
    raster samples -- is untested by the parity suite without a synthetic pair. It is also the
    fiddliest arithmetic in the INF reader, and an error there would shift a k-space reference by
    up to half a pulse while still looking entirely plausible.

    `expect_center_us` is derived by hand from the algorithm, so this pins the absolute answer as
    well as the agreement: if both ports drifted the same way, the hand-computed value still
    catches it.

    Note the deltas: no two *adjacent* values are equal, deliberately. The `[VALUES]` body is
    run-length encoded, so `d, d, k` means "d repeated k+2 times" -- a delta list with a repeated
    neighbour decodes to something else entirely, which is easy to forget when hand-building a
    fixture.
    """
    inf_path = tmp_path / "synthetic_INF.dsv"
    rfd_path = tmp_path / "synthetic_RFD.dsv"
    _write_inf(inf_path)
    _write_rfd(rfd_path, deltas)

    rust_rf = rust.read_inf_rf(str(inf_path), str(rfd_path))
    cpp_rf = cpp.read_inf_rf(str(inf_path), str(rfd_path))
    assert len(cpp_rf) == len(rust_rf) == 1, f"{case}: expected one RF event"
    assert cpp_rf == rust_rf, f"{case}: RF events differ"

    center = cpp_rf[0][0]
    middle = cpp_rf[0][1] + cpp_rf[0][2] / 2
    assert center == pytest.approx(expect_center_us * 1e-6, abs=1e-15), (
        f"{case}: center {center * 1e6:.9f} us, expected {expect_center_us} us"
    )
    assert abs(center - middle) > 1e-9, f"{case}: the center was not moved off the window middle"
