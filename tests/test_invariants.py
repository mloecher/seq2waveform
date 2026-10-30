"""Internal invariants on the real datasets, with no external reference.

Run:  pixi run test-invariants

These were the part of the port's differential suite that never needed the Rust oracle: each one
pins two of this library's own code paths to each other, or a decoder to its own definition. The
differential suite went away at cut-over -- it compared against an implementation that no longer
exists -- but these kept their value and so were carried over.

What they are good for is the failure the h5 comparison cannot see. `test_h5.py` checks the
gradients and the ADC/excitation times against Pulseq's own files; it says nothing about whether
the sorted and unsorted samplers still agree, whether the flat k-space still matches the
per-event one, or whether `DsvSet` still lands where the `.seq` says it should. Those are exactly
the pairs a performance change moves one half of -- and the allocation work in `fe2413c` touched
four of them.

See `tests/conftest.py` for how the corpus is found.
"""

import functools

import numpy as np
import pytest
from conftest import TESTDATA, datasets, dsv_paths, requires_corpus, seq_path

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
    """Corner times survive the float-seconds hop that the Python boundary forces on us.

    Everything downstream assumes the picosecond grid is recoverable from the seconds the API
    hands out. If that stopped being true, delay sweeps would quietly stop being reproducible.
    """
    seq = _seq(name)
    for ax in _axes(seq):
        ct, _ = seq.gradient(ax)
        w = s2w.PiecewiseLinear(np.asarray(ct), np.zeros_like(ct))
        expected = s2w.ps_from_secs(np.asarray(ct))
        assert np.array_equal(w.corner_times_ps(), expected), f"g{ax}: corner grid shifted"


def test_picosecond_times_are_taken_verbatim(name):
    """Sampling at int64 picoseconds equals sampling at the equivalent float seconds.

    The exact path exists for delay sweeps; this checks it is wired to the same arithmetic rather
    than to a parallel implementation of it.
    """
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
    """`sample_at` (binary search) and `sample_at_sorted` (forward sweep) agree.

    The sweep is only taken when times are non-decreasing, so the two exist side by side and an
    optimisation of either could silently move only one.
    """
    seq = _seq(name)
    times = _query_times(seq)[:10_000]
    for ax in _axes(seq):
        ct, cv = seq.gradient(ax)
        w = s2w.PiecewiseLinear(np.asarray(ct), np.asarray(cv))
        assert np.array_equal(w.sample_at(times), w.sample_at_sorted(times)), (
            f"g{ax}: sorted and unsorted sampling differ"
        )


def test_moments_sorted_path_matches_pointwise(name):
    """`moments_at_sorted` (one forward cursor) agrees with repeated `moment_to`.

    The batched form is what k-space actually runs through, and it carries all the cursor
    bookkeeping; `moment_to` is the simple definition. They must not diverge.
    """
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
    """The flat and per-event k-space paths agree with each other.

    Two implementations of the same quantity: `adc_kspace_flat` fills one array in place while
    `adc_kspace` builds a trace per event. This is the test that would have caught it if the
    in-place rewrite had got an offset wrong.
    """
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
    """A uniform-raster sample equals interpolating the corner form at the same times.

    `gradients_dense` goes through the sampler; this reconstructs it from the published corners.
    Both are piecewise-linear interpolation of the same waveform, so they agree exactly.
    """
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
    gx_flat = s.gradients(flat)[0]
    gx_grid = s.gradients(grid)[0]
    assert gx_grid.shape == (8, 16), f"expected (8, 16), got {gx_grid.shape}"
    assert np.array_equal(gx_grid.ravel(), gx_flat), "N-D and flat results differ"


# --- the DSV decoder against its own definition ----------------------------------------------


def _dsv_channels():
    if not TESTDATA.is_dir():
        return []
    return sorted(
        f
        for d in TESTDATA.iterdir()
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
    """Sampling the decoded corners at `(k+0.5)*raster` reproduces the dense samples exactly.

    The decoder's own invariant, and the claim that justifies never building the dense array: a run
    of equal deltas is a straight line, and interpolating a line gives the line back.

    Resampled through `PiecewiseLinear`, not `np.interp`, and that matters: `PiecewiseLinear`
    interpolates from integer picosecond times, exactly as `dense()` does internally, so this can
    assert bit equality. `np.interp` works in float seconds and disagrees in the last few digits
    wherever the slope is steep, which would force a tolerance and turn a sharp test into a vague
    one.
    """
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
    """Peak |k| per axis from the `.seq` route and from the DSV route, or None if either is absent.

    Returns
    -------
    tuple of (ndarray, ndarray), or None
        ``(seq_peaks, dsv_peaks)``, each ``[|kx|, |ky|, |kz|]`` in 1/m.
    """
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
    """The same k-space, reached two independent ways, peaks at the same magnitude.

    The `.seq` route expands Pulseq blocks; the DSV route decodes what the scanner actually played.
    Different files, different arithmetic, same physics -- so this is a tolerance check, not an
    exact one, and it is the only test here that would notice if the DSV frame, the ~1020 us
    Siemens lead-in or the mT/m -> Hz/m conversion went wrong, since each of those shifts one route
    and not the other.

    The threshold is 0.1%, which is tight, and it is tight because the measured agreement is
    0.0091% at worst across the corpus (`cones`, `girf` and `sync_osc` are exact to the printed
    digits; `epi` differs by 0.0091% on kx). It can be this tight because the corpus's DSVs are
    `SimulationProtocol_*` files from the Siemens sequence simulator rather than scope traces --
    so what this compares is the frame, the units and the lead-in, not gradient-system fidelity. A
    real measured DSV would not pass at 0.1%, and if one is ever added to the corpus this threshold
    is the thing to revisit, not the code.

    Calibrated against a deliberately wrong gamma on `cones`, to check it is not vacuous: a 0.1%
    error lands on the bar, 0.5% and 2% are caught. So this detects a scale error of roughly 0.2%
    upward, which covers every units mistake worth worrying about.
    """
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
    """`Sequence` and `DsvSet` present the same WaveformSource API.

    The claim that a DSV+INF set is a drop-in for a `.seq`. Both register their shared surface from
    one template in the bindings, so this mainly guards against someone splitting them again.
    """
    shared = {
        "gradient", "duration", "adc_events", "adc_events_meta", "triggers", "sampler",
        "adc_specs", "adc_excitations", "trigger_specs", "adc_gradient_samples", "adc_kspace",
        "adc_kspace_flat", "trigger_gradient_samples", "trigger_kspace", "trigger_kspace_flat",
        "gradients_dense", "gradient_axis_dense",
    }
    for cls in (s2w.Sequence, s2w.DsvSet):
        missing = shared - {n for n in dir(cls) if not n.startswith("_")}
        assert not missing, f"{cls.__name__} is missing {sorted(missing)}"
