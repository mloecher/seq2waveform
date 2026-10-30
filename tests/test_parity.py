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
