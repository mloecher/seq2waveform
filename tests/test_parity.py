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
