"""Ground truth: the library against Pulseq's own reference `.h5`.

Run:  pixi run test-h5

This is the suite that does not depend on our own arithmetic being right. The `.h5` is written by
Pulseq itself, from the same `.seq`, so it is an outside answer -- which is why it is the one to
run after touching a waveform, and why it survived the cut-over from the port's differential suite
unchanged. It is a tolerance comparison (1e-4 relative), not an exact one: Pulseq and this library
are independent implementations, not bit-compatible.

`test_dsv_gradients_match_h5` is the other half of the project's actual question -- that what the
scanner played matches what Pulseq asked for -- and is the only test here that reads the Siemens
side. It came from the Rust repo's `scripts/test_datasets.py`, whose other three tests are the
three above it, already present.

See `tests/conftest.py` for how the corpus is found.
"""

import functools

import numpy as np
import pytest
from conftest import TESTDATA, datasets, requires_corpus, seq_path

import seq2waveform as s2w

h5py = pytest.importorskip("h5py", reason="h5py is needed to read the reference datasets")


def _orient(a, ncol):
    """Datasets arrive as (N,), (N, ncol) or (ncol, N); normalise to (N, ncol)."""
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else (a if a.shape[1] == ncol else a.T)


@functools.lru_cache(maxsize=None)
def _seq(name):
    """Parse a dataset's .seq once and reuse it across tests."""
    return s2w.Sequence(seq_path(name))


DATASETS = datasets(".seq", ".h5")
pytestmark = requires_corpus(DATASETS)


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


def test_dsv_gradients_match_h5(name):
    """The Siemens DSV gradient agrees with the Pulseq h5 -- the project's actual question.

    The two live in different frames. The DSV is what the scanner played, in mT/m on a 10 us grid
    whose samples sit at interval centers, starting after an INF-encoded lead-in. The h5 is what
    Pulseq asked for, in Hz/m at t=0. So the h5 is converted with the Siemens gamma and shifted by
    the lead-in, and the two are compared on the DSV's own grid.

    The lead-in comes from the INF (the start of event block 1), never from a fit -- a fitted shift
    would absorb exactly the error this test exists to find. The assertion on its magnitude is
    there because a wrong lead-in otherwise shows up only as a small amplitude mismatch.

    Compared on the 10 us DSV grid rather than the h5's: the h5 carries extra 5 us corners from
    Pulseq's `restoreAdditionalShapeSamples`, which is reconstruction detail Siemens never had, so
    comparing there would hold the scanner to a resolution it was never given.
    """
    dsv = TESTDATA / name / "dsv"
    if not dsv.is_dir():
        pytest.skip("dataset has no dsv/ folder")
    inf = sorted(dsv.glob("*_INF.dsv"))
    if not inf:
        pytest.skip("dataset has no INF channel")

    lead = s2w.inf_lead_in(str(inf[0]))  # seconds
    assert 0.9e-3 < lead < 1.2e-3, f"{name}: unexpected lead-in {lead * 1e6:.1f} us"

    with h5py.File(TESTDATA / name / f"{name}.h5") as f:
        axes = [ax for ax in "xyz" if f"g{ax}" in f]
        truth = {ax: _orient(f[f"g{ax}"][()], 2) for ax in axes}

    compared = 0
    for ax in axes:
        hits = sorted(dsv.glob(f"*_GR{ax.upper()}.dsv"))
        if not hits:
            continue
        dt, dv, raster, _ = s2w.read_dsv_pwl(str(hits[0]))
        dt, dv = np.asarray(dt), np.asarray(dv)
        if dt.size < 2:
            continue
        ht = truth[ax][:, 0]
        hv = truth[ax][:, 1] / s2w.SCAN_GAMMA * 1e3  # mT/m, Pulseq frame
        grid = np.arange(dt[0], dt[-1] + raster, raster)
        d = np.interp(grid, dt, dv)
        h = np.interp(grid - lead, ht, hv, left=0.0, right=0.0)
        rel = np.abs(d - h).max() / max(np.abs(hv).max(), 1e-3)
        assert rel < 1e-3, f"{name} g{ax} DSV vs h5: relative max diff {rel:.2e}"
        compared += 1

    if compared == 0:
        pytest.skip("no gradient channel present in both the h5 and the dsv/ folder")
