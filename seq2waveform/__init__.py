"""seq2waveform: read Pulseq .seq and Siemens .dsv files into waveform arrays.

Two sources, one interface. ``Sequence`` reads a Pulseq ``.seq`` and expands its blocks;
``DsvSet`` reads a Siemens measurement's ``.dsv`` channels plus its ``.inf``. Both then offer the
same sampling and k-space API, so a Siemens set is a drop-in for a ``.seq``.

Times are int64 picoseconds internally, which is what makes a delay sweep bit-exact regardless of
where in the sequence it lands; values stay float64.
"""

from ._core import (
    SCAN_GAMMA,
    DsvSet,
    PiecewiseLinear,
    Sampler,
    Sequence,
    inf_lead_in,
    parse_file,
    ps_from_secs,
    ps_to_secs,
    read_dsv,
    read_dsv_pwl,
    read_inf_adc,
    read_inf_adc_events,
    read_inf_rf,
    read_inf_sync,
)

__all__ = [
    "SCAN_GAMMA",
    "DsvSet",
    "PiecewiseLinear",
    "Sampler",
    "Sequence",
    "inf_lead_in",
    "parse_file",
    "ps_from_secs",
    "ps_to_secs",
    "read_dsv",
    "read_dsv_pwl",
    "read_inf_adc",
    "read_inf_adc_events",
    "read_inf_rf",
    "read_inf_sync",
]
