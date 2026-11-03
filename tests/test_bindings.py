"""Array validation and shape handling at the Python boundary."""

import numpy as np
import pytest

import seq2waveform as s2w
from conftest import BUNDLED

pytestmark = pytest.mark.no_corpus


@pytest.fixture(scope="module")
def sampler():
    return s2w.Sequence(str(BUNDLED / "epi" / "epi.seq")).sampler()


@pytest.mark.parametrize("picoseconds", [False, True])
@pytest.mark.parametrize("times,values", [([0, 2], [1]), ([2, 0], [1, 2]), ([0, 0], [1, 2])])
def test_invalid_waveform_corners_raise(times, values, picoseconds):
    build = s2w.PiecewiseLinear.from_picoseconds if picoseconds else s2w.PiecewiseLinear
    times = np.array(times, dtype=np.int64 if picoseconds else np.float64)
    with pytest.raises(ValueError, match="same length|strictly increasing"):
        build(times, np.array(values, dtype=np.float64))


def test_corner_times_must_stay_distinct_after_rounding():
    with pytest.raises(ValueError, match="strictly increasing"):
        s2w.PiecewiseLinear(np.array([0.0, 0.1e-12]), np.array([1.0, 2.0]))


@pytest.mark.parametrize("picoseconds", [False, True])
def test_empty_waveform_is_valid(picoseconds):
    build = s2w.PiecewiseLinear.from_picoseconds if picoseconds else s2w.PiecewiseLinear
    waveform = build(np.array([], dtype=np.int64 if picoseconds else np.float64), np.array([]))
    assert waveform.value_at(0.0) == 0.0
    assert waveform.cumulative_moment().size == 0


def test_moments_require_one_value_per_corner():
    waveform = s2w.PiecewiseLinear(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    with pytest.raises(ValueError, match="one value per waveform corner"):
        waveform.moment_to(0.5, np.array([0.0]))
    for dtype in (np.float64, np.int64):
        with pytest.raises(ValueError, match="one value per waveform corner"):
            waveform.moments_at_sorted(np.array([0], dtype=dtype), np.array([]))


def test_scalar_moments_match_integrals_without_changing_the_table():
    waveform = s2w.PiecewiseLinear(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 0.0]))
    moments = waveform.cumulative_moment()
    moments.setflags(write=False)
    for time in (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        expected = waveform.integral(0.0, min(max(time, 0.0), 2.0))
        assert waveform.moment_to(time, moments) == expected
    np.testing.assert_array_equal(moments, [0.0, 1.0, 2.0])


@pytest.mark.parametrize("offsets,references", [
    ([], []), ([0], [0.0]), ([1, 2], [0.0]), ([0, 1], [0.0]),
    ([0, 3], [0.0]), ([0, -1, 2], [0.0, 0.0]), ([0, 2, 1, 2], [0.0, 0.0, 0.0]),
])
def test_invalid_flat_offsets_raise(sampler, offsets, references):
    with pytest.raises(ValueError, match="offsets must"):
        sampler.kspace_flat(np.array([0.0, 1e-3]), np.array(offsets, dtype=np.int64),
                            np.array(references, dtype=np.float64))


def test_flat_sampling_accepts_empty_events(sampler):
    times = np.array([0.0, 1e-3])
    references = np.array([0.0, 0.0, 0.0])
    result = sampler.kspace_flat(times, np.array([0, 0, 2, 2]), references)
    for got, expected in zip(result, sampler.kspace(times, 0.0)[1:]):
        np.testing.assert_array_equal(got, expected)
    for references, offsets in ((np.array([]), np.array([0])), (np.array([0.0]), np.array([0, 0]))):
        assert all(axis.size == 0 for axis in sampler.kspace_flat(np.array([]), offsets, references))


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (2, 3)])
def test_sampler_preserves_query_shape(sampler, dtype, shape):
    times = np.zeros(shape, dtype=dtype)
    for result in (sampler.gradients(times), sampler.kspace(times, 0 if dtype == np.int64 else 0.0)):
        assert all(axis.shape == shape for axis in result)
