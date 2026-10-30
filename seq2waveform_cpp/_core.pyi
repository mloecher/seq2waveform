"""
Compiled core of the seq2waveform C++ port.

Import the public API from the `seq2waveform_cpp` package rather than from here.
"""

from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


class PiecewiseLinear:
    """
    A piecewise-linear waveform: the value is linear between
    `(t, v)` corners, clamped to the endpoints outside the corner
    range. The canonical gradient form -- what Pulseq emits
    (a trapezoid is 4 corners) and what the DSV collapses to.

    Corner times are held as int64 picoseconds internally, so
    differences are exact at any absolute time.
    """

    def __init__(self, t_seconds: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)], values: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> None:
        """
        Build from corner times in seconds.

        Parameters
        ----------
        t_seconds : ndarray of float64
            Strictly increasing corner times (s); rounded to the nearest picosecond.
        values : ndarray of float64
            Corner values, same length as `t_seconds` (Hz/m for gradients).
        """

    @staticmethod
    def from_picoseconds(t_ps: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', writable=False)], values: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> PiecewiseLinear:
        """
        Build from corner times in int64 picoseconds, taken verbatim.

        The bit-exact path: use it when corner times came from another exact source and
        must not round-trip through float seconds.

        Parameters
        ----------
        t_ps : ndarray of int64
            Strictly increasing corner times (ps).
        values : ndarray of float64
            Corner values, same length as `t_ps`.
        """

    def __len__(self) -> int:
        """Number of corners."""

    def corner_times(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """Corner times in seconds."""

    def corner_times_ps(self) -> Annotated[NDArray[numpy.int64], dict(shape=(None,))]:
        """Corner times as exact int64 picoseconds."""

    def values(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """Corner values."""

    def value_at(self, t_seconds: float) -> float:
        """
        Linearly interpolated value at `t_seconds`, clamped outside the corner range.
        """

    def integral(self, t0: float, t1: float) -> float:
        """
        Exact integral of the waveform from `t0` to `t1` (s), as a sum of trapezoids.

        Negative when `t1 < t0`. This is the k-space building block.
        """

    @overload
    def sample_at(self, times: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """
        Sample at arbitrary `times` (float64 seconds), in any order.

        Returns
        -------
        ndarray of float64
            One value per entry of `times`.
        """

    @overload
    def sample_at(self, times: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """Sample at arbitrary `times` (int64 picoseconds, taken verbatim)."""

    @overload
    def sample_at_sorted(self, times: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """
        Sample at non-decreasing `times` (float64 seconds) with one forward sweep.

        O(corners + times) instead of a binary search per point -- the hot path for an ADC
        readout. Results are undefined if `times` is not non-decreasing.
        """

    @overload
    def sample_at_sorted(self, times: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """
        Sample at non-decreasing `times` (int64 picoseconds) with one forward sweep.
        """

    def cumulative_moment(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """
        Running integral at each corner: `cum[i]` is the integral from `t[0]` to `t[i]`.

        Precompute once, then `moment_to` and `moments_at_sorted` evaluate in O(1) per point.
        """

    def moment_to(self, t_seconds: float, cum: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> float:
        """
        Integral from the first corner to `t_seconds`, using precomputed `cum`.

        k-space referenced to `t_ref` is `moment_to(t) - moment_to(t_ref)`.
        """

    @overload
    def moments_at_sorted(self, times: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)], cum: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """
        `moment_to` over non-decreasing `times` (float64 seconds) with one forward sweep.
        """

    @overload
    def moments_at_sorted(self, times: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', writable=False)], cum: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """
        `moment_to` over non-decreasing `times` (int64 picoseconds) with one forward sweep.
        """

def ps_from_secs(seconds: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> Annotated[NDArray[numpy.int64], dict(shape=(None,))]:
    """
    Convert seconds to exact int64 picoseconds (rounds half away from zero).

    Use this to pin a delay sweep to an exact grid before sampling.
    """

def ps_to_secs(picoseconds: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """Convert exact int64 picoseconds back to float64 seconds."""
