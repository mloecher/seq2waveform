"""Compare equivalent sampling paths on real datasets.

Covers sorted versus unsorted sampling, flat versus per-event k-space and .seq
versus DSV outputs. Dataset discovery is in conftest.py.
"""

import functools

import numpy as np
import pytest
from conftest import SOURCES, datasets, dsv_paths, requires_corpus, seq_path

import seq2waveform as s2w

# Cap the sample count per dataset: EPI has millions of ADC samples and the point here is
# agreement between paths, not throughput.
MAX_SAMPLES = 200_000

DATASETS = datasets(".seq")
pytestmark = requires_corpus(DATASETS)


@pytest.fixture(params=DATASETS)
def name(request):
    return request.param


@functools.lru_cache(maxsize=None)
def _seq(nm):
    """Parse a dataset's `.seq` once and reuse it; nothing here mutates the Sequence."""
    return s2w.Sequence(seq_path(nm))


def _query_times(seq):
    """A non-decreasing time vector to sample at."""
    t = np.asarray(seq.adc_times())
    if t.size == 0:
        pytest.skip("dataset has no ADC events")
    return np.sort(t)[:MAX_SAMPLES]


def _axes(seq):
    """Gradient axes that actually carry corners in this dataset."""
    return [ax for ax in "xyz" if len(seq.gradient(ax)[0]) > 1]


# --- the time grid ---------------------------------------------------------------------------


def test_corner_times_roundtrip_exactly(name):
    """Corner times round-trip between float seconds and integer picoseconds."""
    seq = _seq(name)
    for ax in _axes(seq):
        ct, _ = seq.gradient(ax)
        w = s2w.PiecewiseLinear(np.asarray(ct), np.zeros_like(ct))
        expected = s2w.ps_from_secs(np.asarray(ct))
        assert np.array_equal(w.corner_times_ps(), expected), f"g{ax}: corner grid shifted"


def test_picosecond_times_are_taken_verbatim(name):
    """Integer picosecond and equivalent float-second queries agree."""
    seq = _seq(name)
    times = _query_times(seq)[:10_000]
    times_ps = s2w.ps_from_secs(times)
    for ax in _axes(seq):
        ct, cv = seq.gradient(ax)
        w = s2w.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        assert np.array_equal(w.sample_at_sorted(times_ps), w.sample_at_sorted(times)), (
            f"g{ax}: picosecond and seconds paths differ"
        )


# --- paths that must not drift apart ---------------------------------------------------------


def test_unsorted_path_matches_sorted_path(name):
    """Binary searches and the sorted sweep return the same values."""
    seq = _seq(name)
    times = _query_times(seq)[:10_000]
    for ax in _axes(seq):
        ct, cv = seq.gradient(ax)
        w = s2w.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        assert np.array_equal(w.sample_at(times), w.sample_at_sorted(times)), (
            f"g{ax}: sorted and unsorted sampling differ"
        )


def test_moments_sorted_path_matches_pointwise(name):
    """The moment sweep matches independent point queries."""
    seq = _seq(name)
    times = _query_times(seq)[:5_000]
    checked = 0
    for ax in _axes(seq):
        ct, cv = seq.gradient(ax)
        w = s2w.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        cum = w.cumulative_moment()
        batched = w.moments_at_sorted(times, cum)
        pointwise = np.array([w.moment_to(t, cum) for t in times])
        assert np.array_equal(batched, pointwise), f"g{ax}: batched and pointwise moments differ"
        checked += times.size
    assert checked > 0, "no axis carried corners; nothing was compared"


def test_flat_path_matches_per_event_path(name):
    """Flat and per-event k-space agree exactly, including event boundaries."""
    c = _seq(name)
    t, kx, ky, kz, off = c.adc_kspace_flat()
    per = c.adc_kspace()
    assert len(off) == len(per) + 1, "offsets and event count disagree"
    for e, (et, ex, ey, ez) in enumerate(per):
        a, b = off[e], off[e + 1]
        assert np.array_equal(t[a:b], et), f"event {e}: times differ"
        assert np.array_equal(kx[a:b], ex), f"event {e}: kx differs"
        assert np.array_equal(ky[a:b], ey), f"event {e}: ky differs"
        assert np.array_equal(kz[a:b], ez), f"event {e}: kz differs"


def test_dense_sampling_matches_the_corners(name):
    """Dense output equals interpolation of the corners at the same times."""
    seq = _seq(name)
    t, gx, gy, gz = seq.gradients_dense(1e-3)
    for ax, want in zip("xyz", (gx, gy, gz)):
        ct, cv = seq.gradient(ax)
        if len(ct) < 2:
            continue
        w = s2w.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        assert np.array_equal(w.sample_at_sorted(t), want), f"g{ax}: dense differs from corners"


def test_nd_time_arrays_keep_their_shape(name):
    """An N-D query comes back shaped like the query, with the same values as the flat form."""
    c = _seq(name)
    flat = _query_times(c)[: 8 * 16]
    if flat.size < 8 * 16:
        pytest.skip("not enough ADC samples for a 2-D query")
    grid = flat.reshape(8, 16)
    s = c.sampler()
    gx_flat = s.gradients(flat)[1]
    gx_grid = s.gradients(grid)[1]
    assert gx_grid.shape == (8, 16), f"expected (8, 16), got {gx_grid.shape}"
    assert np.array_equal(gx_grid.ravel(), gx_flat), "N-D and flat results differ"


# --- the DSV decoder against its own definition ----------------------------------------------


def _dsv_channels():
    return sorted(
        f
        for root in SOURCES
        if root.is_dir()
        for d in root.iterdir()
        if d.is_dir() and (d / "dsv").is_dir()
        for f in (d / "dsv").glob("*.dsv")
        if not f.name.endswith("_INF.dsv")  # not a waveform channel
    )


def _kind(p):
    return p.stem.replace("SimulationProtocol_", "")


def _dsv_id(p):
    return f"{p.parent.parent.name}/{_kind(p)}"


_ALL_DSV = _dsv_channels()
# The smallest file of each kind: one per raster/unit combination. Decoding a large channel to its
# dense form costs hundreds of MB, and one file per kind covers every code path in the decoder.
DSV_PER_KIND = [
    min((f for f in _ALL_DSV if _kind(f) == k), key=lambda p: p.stat().st_size)
    for k in sorted({_kind(f) for f in _ALL_DSV})
]


@pytest.fixture(params=DSV_PER_KIND, ids=_dsv_id)
def dsv_kind(request):
    return str(request.param)


@pytest.mark.skipif(not DSV_PER_KIND, reason="no .dsv channels in the corpus")
def test_dsv_corners_reproduce_the_dense_samples(dsv_kind):
    """Decoded DSV corners reproduce the dense output exactly at raster centers."""
    try:
        dense, raster, _ = s2w.read_dsv(dsv_kind)
    except ValueError:
        pytest.skip("the reader rejects this kind (no VERTFACTOR); see test_dsv.cpp")
    if dense.size == 0:
        pytest.skip("empty channel")
    ct, cv, _, _ = s2w.read_dsv_pwl(dsv_kind)
    w = s2w.PiecewiseLinear(ct, cv)
    times = (np.arange(dense.size, dtype=np.float64) + 0.5) * raster
    assert np.array_equal(w.sample_at_sorted(times), dense), "corners do not reproduce the samples"


# --- the two sources against each other ------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _peak_k(nm):
    """Peak absolute k-space per axis from the .seq and DSV sources."""
    paths = dsv_paths(nm)
    if paths is None:
        return None

    def peaks(src):
        _t, kx, ky, kz, _off = src.adc_kspace_flat()
        if kx.size == 0:
            return None
        return np.array([np.abs(kx).max(), np.abs(ky).max(), np.abs(kz).max()])

    a = peaks(s2w.Sequence(seq_path(nm)))
    b = peaks(s2w.DsvSet(*paths))
    return None if a is None or b is None else (a, b)


def test_seq_and_dsv_kspace_agree(name):
    """Sequence and simulator DSV peak k-space agree within 0.1% of full scale.

    The corpus differs by at most 0.0091%. This checks units and time alignment;
    measured scanner traces may need a different tolerance."""
    pair = _peak_k(name)
    if pair is None:
        pytest.skip("dataset has no DSV set, or no ADC samples on one route")
    seq_peaks, dsv_peaks = pair
    scale = max(seq_peaks.max(), dsv_peaks.max())
    if scale == 0.0:
        pytest.skip("no k-space on either route")
    for ax, a, b in zip("xyz", seq_peaks, dsv_peaks):
        rel = abs(a - b) / scale
        assert rel < 1e-3, (
            f"k{ax}: .seq peaks at {a:.1f} 1/m but the DSV set peaks at {b:.1f} 1/m "
            f"({rel:.1%} of full scale)"
        )


def test_dsvset_and_sequence_expose_the_same_interface():
    """Sequence and DsvSet expose every shared sampling method."""
    shared = {
        "gradient", "duration", "adc_events", "adc_events_meta", "triggers", "sampler",
        "adc_specs", "adc_excitations", "trigger_specs", "adc_gradient_samples", "adc_kspace",
        "adc_kspace_flat", "trigger_gradient_samples", "trigger_kspace", "trigger_kspace_flat",
        "gradients_dense", "gradient_axis_dense",
    }
    for cls in (s2w.Sequence, s2w.DsvSet):
        missing = shared - {n for n in dir(cls) if not n.startswith("_")}
        assert not missing, f"{cls.__name__} is missing {sorted(missing)}"
