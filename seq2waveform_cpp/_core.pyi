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

    @property
    def duration(self) -> float:
        """Total sequence duration (s): the sum of the block durations."""

    def sampler(self) -> Sampler:
        """
        Build a `Sampler` over this sequence's three gradient channels.

        Precomputes the channels and their cumulative moments, so repeated sampling (a delay
        sweep, say) does not re-expand the waveforms.

        Returns
        -------
        Sampler
        """

    def adc_specs(self) -> list[object]:
        """
        Where to sample for each ADC event, and what to integrate k-space from.

        The reference is the most recent excitation *before* the event. A warning goes to
        stderr if the excitation and ADC counts disagree (fat-sat, for instance) -- that is
        how a mismatched sequence announces itself, so do not silence it.

        Returns
        -------
        list of tuple
            `(times_s, reference_s)` per ADC event.
        """

    def adc_excitations(self) -> list[tuple[float, float, float]]:
        """
        The excitation matched to each ADC event, aligned 1:1 with `adc_specs`.

        Returns
        -------
        list of tuple
            `(center_s, freq_hz, phase_rad)` per ADC event, or `(nan, nan, nan)` when no
            excitation precedes it. Use `freq` and the gradient at `center` to get the slice
            offset of each readout.
        """

    def trigger_specs(self, delay: float, dt: float, n: int) -> list[object]:
        """
        Sample specs for trigger-based (Skope) measurements.

        Parameters
        ----------
        delay : float
            Offset (s) from each trigger to the first sample.
        dt : float
            Sample spacing (s).
        n : int
            Samples per trigger.

        Returns
        -------
        list of tuple
            `(times_s, reference_s)` per trigger. The reference is the trigger itself, not
            an excitation -- that is the external measurement's own reference.
        """

    def adc_gradient_samples(self) -> list[object]:
        """
        Gradient values (Hz/m) at every ADC sample, one trace per event.

        Returns
        -------
        list of tuple
            `(t, gx, gy, gz)` per ADC event.
        """

    def adc_kspace(self) -> list[object]:
        """
        k-space (1/m) at every ADC sample, one trace per event.

        Each event integrates from its own preceding excitation. Gamma is already baked
        into the Hz/m gradients, so this is the integral of G dt directly.

        Returns
        -------
        list of tuple
            `(t, kx, ky, kz)` per ADC event.
        """

    def adc_kspace_flat(self) -> object:
        """
        Whole-sequence ADC k-space as flat arrays -- the fast path for large sequences.

        Returns
        -------
        tuple
            `(t, kx, ky, kz, offsets)`. `offsets` are cumulative event boundaries, so
            `offsets[0] == 0`, `offsets[-1] == len(t)`, and event `e` is
            `offsets[e]:offsets[e+1]`. Recover the per-event split with
            `np.split(kx, offsets[1:-1])`.
        """

    def trigger_gradient_samples(self, delay: float, dt: float, n: int) -> list[object]:
        """Gradient values at trigger-based sample points, one trace per trigger."""

    def trigger_kspace(self, delay: float, dt: float, n: int) -> list[object]:
        """k-space at trigger-based sample points, integrated from each trigger."""

    def trigger_kspace_flat(self, delay: float, dt: float, n: int) -> object:
        """Trigger-based k-space as flat arrays. See `adc_kspace_flat`."""

    def gradients_dense(self, raster: float) -> object:
        """
        All three gradient axes on a uniform raster over `[0, duration]`.

        Parameters
        ----------
        raster : float
            Sample spacing (s).

        Returns
        -------
        tuple
            `(t, gx, gy, gz)`. The last sample is at or past `duration`.
        """

    def gradient_axis_dense(self, axis: str, raster: float) -> tuple:
        """
        One gradient axis on a uniform raster over `[0, duration]`.

        Returns
        -------
        tuple
            `(t, values_hz_per_m)`.
        """

class Sampler:
    """
    Precomputed gradient channels and their cumulative moments, for fast repeated
    sampling. Build one with `Sequence.sampler()`.

    Query times are float64 seconds, or int64 picoseconds taken verbatim -- the exact path,
    for delay sweeps that must stay bit-exact. An N-D time array comes back with its shape
    preserved.
    """

    @overload
    def gradients(self, times: Annotated[NDArray[numpy.float64], dict(order='C', writable=False)]) -> tuple:
        """
        Gradient values (Hz/m) at `times`, interpolated between corners.

        Parameters
        ----------
        times : ndarray
            Query times: float64 seconds, or int64 picoseconds taken verbatim. Any shape.
            A non-decreasing array takes a single forward sweep per axis; otherwise it is a
            binary search per point.

        Returns
        -------
        tuple of ndarray
            `(gx, gy, gz)`, each shaped like `times`.
        """

    @overload
    def gradients(self, times: Annotated[NDArray[numpy.int64], dict(order='C', writable=False)]) -> tuple:
        """As above, with query times given as exact int64 picoseconds."""

    @overload
    def kspace(self, times: Annotated[NDArray[numpy.float64], dict(order='C', writable=False)], reference: float) -> tuple:
        """
        k-space (1/m) at `times`, integrating each gradient from `reference`.

        Parameters
        ----------
        times : ndarray
            Query times (float64 seconds or int64 picoseconds). Any shape.
        reference : float
            Integration start (s) -- an excitation center for an ADC readout, or the
            trigger time for an external measurement.

        Returns
        -------
        tuple of ndarray
            `(kx, ky, kz)`, each shaped like `times`.
        """

    @overload
    def kspace(self, times: Annotated[NDArray[numpy.int64], dict(order='C', writable=False)], reference: int) -> tuple:
        """As above, with times and reference given as exact int64 picoseconds."""

    def kspace_flat(self, times: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)], offsets: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', writable=False)], references: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]) -> tuple:
        """
        Batched k-space over many events in one call, returning flat arrays.

        Parameters
        ----------
        times : ndarray of float64
            All events' query times (s), concatenated.
        offsets : ndarray of int64
            Cumulative event boundaries, length `len(references) + 1`: event `e` is
            `times[offsets[e]:offsets[e+1]]`.
        references : ndarray of float64
            Per-event integration start (s).

        Returns
        -------
        tuple of ndarray
            `(kx, ky, kz)`, each the length of `times`. Reuses the one prebuilt sampler, so
            this is the fast path for whole-sequence k-space and for delay sweeps.
        """

def read_dsv(path: str) -> tuple:
    """
    Read a Siemens `.dsv` channel as dense per-sample values.

    Parameters
    ----------
    path : str
        Path to one `.dsv` channel file.

    Returns
    -------
    values : ndarray of float64
        Per-sample values in physical units (`VERTFACTOR` applied).
    raster_time : float
        Sample spacing (s), from `HORIDELTA`.
    unit : str
        The `VERTUNITNAME` string, e.g. "mT/m".

    Raises
    ------
    ValueError
        If a value line is not an integer, or `HORIDELTA`/`VERTFACTOR` is missing.
    """

def read_dsv_pwl(path: str) -> tuple:
    """
    Read a Siemens `.dsv` channel as piecewise-linear corners.

    The `[VALUES]` body is a run-length-encoded derivative, so a run of equal deltas is
    one constant-slope segment and only slope changes are kept. The dense array is never
    built, so this is both sparser and cheaper than `read_dsv`.

    Parameters
    ----------
    path : str
        Path to one `.dsv` channel file.

    Returns
    -------
    corner_times : ndarray of float64
        Corner times (s), interval-centered at `(k+0.5)*raster_time`.
    corner_values : ndarray of float64
        Corner values in physical units.
    raster_time : float
        Sample spacing (s).
    unit : str
        The `VERTUNITNAME` string.
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
