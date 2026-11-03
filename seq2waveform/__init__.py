"""Read Pulseq .seq files and DSV exports into waveform arrays.

Sequence and DsvSet share gradient and k-space sampling methods. load() selects
one from a path and defaults to the DSV gradient representation. Times are stored
as int64 picoseconds; values are float64. See USAGE.md for examples.
"""

from pathlib import Path

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

# DsvSet constructor order; other exported channels are ignored.
_DSV_CHANNELS = ("GRX", "GRY", "GRZ", "RFD", "INF")


def _dsv_index(directory):
    """Index DSV paths by upper-case filename for case-insensitive lookup."""
    return {q.name.upper(): q for q in directory.iterdir()
            if q.is_file() and q.suffix.lower() == ".dsv"}


def _dsv_paths(src):
    """Resolve an export directory or member file to (grx, gry, grz, rfd, inf).

    Raise FileNotFoundError for missing channels and ValueError for ambiguous exports."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"no such path: {src}")

    if src.is_dir():
        directory, prefix = src, None
    else:
        # Any member file identifies its own set: strip the trailing _CHANNEL from the stem.
        directory = src.parent
        stem = src.stem
        if "_" not in stem:
            raise ValueError(
                f"{src.name} does not look like a DSV channel file (expected a name ending in "
                f"_GRX, _GRY, _GRZ, _RFD or _INF)")
        prefix = stem.rsplit("_", 1)[0]

    index = _dsv_index(directory)
    if prefix is None:
        # Infer the prefix from GRX, preserving filename case in error messages.
        hits = sorted((q for key, q in index.items() if key.endswith("_GRX.DSV")),
                      key=lambda q: q.name)
        if not hits:
            raise FileNotFoundError(
                f"no *_GRX.dsv in {directory} -- is this a DSV export directory? "
                f"A Pulseq .seq has to be named directly.")
        prefixes = [q.stem.rsplit("_", 1)[0] for q in hits]
        if len(prefixes) > 1:
            raise ValueError(
                f"{directory} holds {len(prefixes)} DSV exports ({', '.join(prefixes)}); name one "
                f"of its .dsv files instead of the directory")
        prefix = prefixes[0]

    out = []
    for channel in _DSV_CHANNELS:
        hit = index.get(f"{prefix}_{channel}.DSV".upper())
        if hit is None:
            raise FileNotFoundError(f"{directory}: missing {prefix}_{channel}.dsv")
        out.append(str(hit))
    return tuple(out)


def load(src, *, mode=None, gamma=None):
    """Open a Pulseq .seq file or DSV export.

    Parameters
    ----------
    src : str or Path
        A .seq file, a directory containing one export, or a member .dsv file.
    mode : {'dsv', 'pulseq', 'centers'}, optional
        Reconstruction mode for .seq files; defaults to 'dsv'. DSV exports accept
        only 'dsv'. Use 'pulseq' to match Pulseq's waveforms_and_times().
    gamma : float, optional
        Gyromagnetic ratio in Hz/T for DSV exports; defaults to SCAN_GAMMA.
        Rejected for .seq files, whose amplitudes are already in Hz/m.

    Returns
    -------
    Sequence or DsvSet
        A source with gradient and k-space sampling methods.

    Raises
    ------
    FileNotFoundError
        The path or a required DSV channel is missing.
    ValueError
        The source is ambiguous or unsupported, or an argument does not apply."""
    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"no such path: {path}")

    suffix = path.suffix.lower()
    is_dsv = path.is_dir() or suffix == ".dsv"

    if is_dsv:
        # Checked before any parsing, so a wrong argument fails fast rather than after a read.
        if mode is not None and mode != "dsv":
            raise ValueError(
                f"mode={mode!r} does not apply to a DSV export: it is already the commanded "
                f"waveform, which is what gradient_mode='dsv' reproduces from a .seq")
        paths = _dsv_paths(path)
        return DsvSet(*paths) if gamma is None else DsvSet(*paths, gamma)

    if suffix == ".seq":
        if gamma is not None:
            raise ValueError("gamma does not apply to a .seq; it applies to a DSV export")
        return Sequence(str(path), gradient_mode="dsv" if mode is None else mode)

    raise ValueError(
        f"{path.name}: expected a .seq file, a DSV export directory, or one of its .dsv files")


__all__ = [
    "SCAN_GAMMA",
    "DsvSet",
    "PiecewiseLinear",
    "Sampler",
    "Sequence",
    "inf_lead_in",
    "load",
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
