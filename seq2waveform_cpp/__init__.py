"""seq2waveform (C++ port): read Pulseq .seq and Siemens .dsv files into waveform arrays.

Installed alongside the Rust `seq2waveform` under a separate name during the port, so the two
implementations can be compared in one environment (see ``cpp/tests/test_parity.py``). The package
is renamed to ``seq2waveform`` once the Rust crate is retired.
"""

from ._core import PiecewiseLinear, ps_from_secs, ps_to_secs

__all__ = [
    "PiecewiseLinear",
    "ps_from_secs",
    "ps_to_secs",
]
