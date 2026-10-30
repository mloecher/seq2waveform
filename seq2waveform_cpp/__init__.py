"""seq2waveform (C++ port): read Pulseq .seq and Siemens .dsv files into waveform arrays.

Installed alongside the Rust `seq2waveform` under a separate name during the port, so the two
implementations can be compared in one environment (see ``tests/test_parity.py``). The package is
renamed to ``seq2waveform`` once the Rust crate is retired.

``_core.parse_model`` and ``_core.parse_shapes`` are deliberately not re-exported here: they are
port scaffolding for the parity suite, mirroring two functions of the same names on the Rust side,
and both go away at cut-over.
"""

from ._core import (
    PiecewiseLinear,
    Sampler,
    Sequence,
    parse_file,
    ps_from_secs,
    ps_to_secs,
    read_dsv,
    read_dsv_pwl,
)

__all__ = [
    "PiecewiseLinear",
    "Sampler",
    "Sequence",
    "parse_file",
    "ps_from_secs",
    "ps_to_secs",
    "read_dsv",
    "read_dsv_pwl",
]
