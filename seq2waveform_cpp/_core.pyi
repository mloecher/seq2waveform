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

def parse_file(path: str) -> int:
    """
    Parse a `.seq` file and return the number of blocks.

    Parameters
    ----------
    path : str
        Path to a Pulseq `.seq` file (format v1.4/1.5).

    Returns
    -------
    int
        Number of rows in the `[BLOCKS]` table.

    Raises
    ------
    ValueError
        If a line is malformed; the message names the source line.
    """

def parse_model(path: str) -> dict:
    """
    Dump a parsed `.seq` model as plain Python data.

    Port scaffolding, not public API: it exists so the port's parity suite can compare the
    parsed sequence against the Rust implementation field by field. Shapes are excluded --
    see `parse_shapes`.

    Parameters
    ----------
    path : str
        Path to a Pulseq `.seq` file.

    Returns
    -------
    dict
        Keys `version`, `definitions`, `signature`, `blocks`, `rf`, `traps`, `arbs`,
        `adc`, `triggers`, `ext_nodes` and `ext_type_names`. The event libraries are
        `{id: tuple}` in the field order of the corresponding struct; `blocks` is a list
        of `(num, duration_s, rf, gx, gy, gz, adc, ext)`. All times are in seconds.
    """

class Sequence:
    """
    A parsed Pulseq sequence: parse once, then query waveforms many times without
    re-parsing.

    Parameters
    ----------
    path : str
        Path to a Pulseq `.seq` file (format v1.4/1.5).
    gradient_mode : {'pulseq', 'centers', 'dsv'}, optional
        How centers-sampled arbitrary gradients render; see the `gradient_mode`
        property. Default 'pulseq'.

    Raises
    ------
    ValueError
        If a line is malformed, naming the source line, or if `gradient_mode` is not
        one of the three names.
    """

    def __init__(self, path: str, gradient_mode: str = 'pulseq') -> None: ...

    @property
    def num_blocks(self) -> int:
        """Number of blocks in the parsed sequence."""

    @property
    def gradient_mode(self) -> str:
        """
        How arbitrary/centers-sampled gradients render:

        - `"pulseq"` (default): edge reconstruction, matches the h5 (can staircase on
          fast waveforms);
        - `"centers"`: connect the center samples for *arbitrary* shapes (smooth),
          trapezoids stay sharp;
        - `"dsv"`: resample the whole channel onto the raster centers -- rounds
          trapezoid corners *and* smooths arbitrary shapes, i.e. the Siemens DSV /
          commanded waveform, one consistent representation.
        """

    @gradient_mode.setter
    def gradient_mode(self, arg: str, /) -> None: ...

    def gradient(self, axis: str) -> tuple:
        """
        Expand one gradient channel to piecewise-linear corners.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Which gradient channel.

        Returns
        -------
        corner_times : ndarray of float64
            Corner times (s), strictly increasing.
        corner_values : ndarray of float64
            Corner amplitudes (Hz/m).
        """

    def adc_times(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """
        All ADC sample times (s), flattened across every ADC event.

        Returns
        -------
        ndarray of float64
            Sample times (s), at dwell centers: `delay + dwell * (k + 0.5)`.
        """

    def adc_events(self) -> list[object]:
        """
        ADC sample times split per event, as the `.seq` records them.

        Returns
        -------
        list of ndarray
            One float64 array of sample times (s) per ADC event.
        """

    def adc_events_meta(self) -> list[object]:
        """
        ADC events with their per-event frequency and phase offsets.

        Returns
        -------
        list of tuple
            `(times_s, freq_hz, phase_rad)` per ADC event.
        """

    def rf_events(self) -> list[tuple[float, float, float, str]]:
        """
        RF pulses as `(center_s, freq_hz, phase_rad, use_name)`.

        `use_name` is the real RF use; undefined pulses are NOT relabelled as excitation,
        which is a deliberate divergence from Pulseq (its h5 routes both into
        `/excitation`, so an h5 comparison has to union the two).

        Returns
        -------
        list of tuple
            One entry per RF event, in block order.
        """

    def triggers(self) -> list[tuple[int, float, float]]:
        """
        Trigger events as `(channel, time_s, duration_s)`.

        Returns
        -------
        list of tuple
            One entry per resolved `extension TRIGGERS` event, in block order.
        """

def parse_shapes(path: str) -> dict:
    """
    Dump the `[SHAPES]` library of a `.seq` file.

    Port scaffolding, like `parse_model`: kept separate because the arrays are large enough
    that round-tripping them through Python lists would dominate the parity run.

    Parameters
    ----------
    path : str
        Path to a Pulseq `.seq` file.

    Returns
    -------
    dict
        `{shape_id: (num_samples, raw, decompressed)}`, where `raw` is the on-disk (often
        run-length-compressed) data and `decompressed` is the expanded sample vector.
    """
