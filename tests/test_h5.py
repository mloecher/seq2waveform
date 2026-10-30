"""Independent ground truth: the C++ port against Pulseq's own reference `.h5`.

Run:  pixi run test-h5

Why this exists separately from `test_parity.py`: every one of those tests compares the port
against the Rust implementation, so an error the *two ports share* is invisible to all of them.
The `.h5` is written by Pulseq itself, so it is the only check in the suite that does not depend on
the Rust code being right. It is a tolerance comparison (1e-4 relative), not an exact one -- the
two implementations are independent, not bit-compatible -- which is exactly why it cannot replace
the parity suite either. They answer different questions.

Mirrors `scripts/test_datasets.py` in the Rust repo, including its orientation handling and its
threshold, so a failure here can be compared directly against the Rust result.
"""

import functools
import os
from pathlib import Path

import numpy as np
import pytest

import seq2waveform_cpp as cpp

h5py = pytest.importorskip("h5py", reason="h5py is needed to read the reference datasets")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


RUST_REPO = _resolve(os.environ.get("S2W_RUST_REPO", "../seq2waveform"))
TESTDATA = _resolve(os.environ.get("S2W_TESTDATA", str(RUST_REPO / "testdata")))


def _datasets():
    if not TESTDATA.exists():
        return []
    return sorted(
        d.name
        for d in TESTDATA.iterdir()
        if d.is_dir() and (d / f"{d.name}.seq").exists() and (d / f"{d.name}.h5").exists()
    )


def _orient(a, ncol):
    """Datasets arrive as (N,), (N, ncol) or (ncol, N); normalise to (N, ncol)."""
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else (a if a.shape[1] == ncol else a.T)


@functools.lru_cache(maxsize=None)
def _seq(name):
    """Parse a dataset's .seq once and reuse it across tests."""
    return cpp.Sequence(str(TESTDATA / name / f"{name}.seq"))


DATASETS = _datasets()
pytestmark = pytest.mark.skipif(
    not DATASETS, reason=f"no .seq/.h5 dataset pairs under {TESTDATA} (set S2W_TESTDATA)"
)


@pytest.fixture(params=DATASETS)
def name(request):
    return request.param


def test_gradients_match_h5(name):
    """Each active gradient axis matches the h5 to the `.seq` amplitude precision.

    The h5 samples the waveform on its own grid, so the comparison interpolates the port's corners
    onto the h5 times (`left`/`right` are 0: outside the channel the gradient is at rest).
    """
    seq = _seq(name)
    with h5py.File(TESTDATA / name / f"{name}.h5") as f:
        axes = [ax for ax in "xyz" if f"g{ax}" in f]
        truth = {ax: _orient(f[f"g{ax}"][()], 2) for ax in axes}
    assert axes, f"{name}: the h5 has no gradient datasets"
    for ax in axes:
        ht, hv = truth[ax][:, 0], truth[ax][:, 1]
        st, sv = seq.gradient(ax)
        s = np.interp(ht, st, sv, left=0.0, right=0.0)
        rel = np.abs(s - hv).max() / max(np.abs(hv).max(), 1.0)
        assert rel < 1e-4, f"{name} g{ax}: relative max diff {rel:.2e}"


def test_excitation_times_match_h5(name):
    """Excitation centers match the h5's `/excitation`.

    Pulseq routes both `excitation` and `undefined` pulses into that dataset while this port keeps
    the real use, so the comparison has to union the two -- the one place that deliberate
    divergence has to be undone to compare at all.
    """
    seq = _seq(name)
    exc = np.sort(
        np.array([c for c, _, _, u in seq.rf_events() if u in ("excitation", "undefined")])
    )
    with h5py.File(TESTDATA / name / f"{name}.h5") as f:
        truth = np.sort(_orient(f["excitation"][()], 3)[:, 0]) if "excitation" in f else np.array([])
    assert exc.size == truth.size, f"{name}: {exc.size} excitations vs h5 {truth.size}"
    if exc.size:
        assert np.abs(exc - truth).max() < 1e-6, f"{name}: excitation centers differ"


def test_adc_times_match_h5(name):
    """ADC sample times match the h5's `/adc`."""
    seq = _seq(name)
    st = np.sort(seq.adc_times())
    with h5py.File(TESTDATA / name / f"{name}.h5") as f:
        ht = np.sort(_orient(f["adc"][()], 4)[:, 0]) if "adc" in f else np.array([])
    assert st.size == ht.size, f"{name}: {st.size} ADC samples vs h5 {ht.size}"
    if st.size:
        assert np.abs(st - ht).max() < 1e-6, f"{name}: ADC times differ"
