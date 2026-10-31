"""seq2waveform: read Pulseq .seq and DSV files into waveform arrays.

``Sequence`` reads a Pulseq ``.seq`` and expands its blocks; ``DsvSet`` reads a measurement's
``.dsv`` channels plus its ``.inf``. Both answer the same sampling and k-space calls, so a DSV set
is a drop-in for a ``.seq``.

Times are int64 picoseconds internally, which is what makes a delay sweep bit-exact regardless of
where in the sequence it lands; values stay float64.

``load`` takes either kind of source and defaults to the gradient representation the two share,
so code written on top of it does not know which it got. See USAGE.md.
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

#: The five channels a ``DsvSet`` needs, in the order its constructor takes them. An export
#: holds many more (ADC, CON, ERR, M0X/Y/Z, ...) that are ignored here.
_DSV_CHANNELS = ("GRX", "GRY", "GRZ", "RFD", "INF")


def _dsv_index(directory):
    """Map ``NAME.DSV`` (upper-cased) -> path, for every ``.dsv`` in one directory.

    Upper-cased because the channel suffix is written in capitals, but a file can arrive from a
    case-insensitive filesystem, an archive or a rename with any casing at all.
    """
    return {q.name.upper(): q for q in directory.iterdir()
            if q.is_file() and q.suffix.lower() == ".dsv"}


def _dsv_paths(src):
    """Resolve a DSV export to the five paths ``DsvSet`` takes.

    Parameters
    ----------
    src : str or Path
        A directory holding one export, or any single ``*.dsv`` file belonging to one. Naming a
        member file is the way to disambiguate a directory that holds several exports.

    Returns
    -------
    tuple of str
        ``(grx, gry, grz, rfd, inf)``.

    Raises
    ------
    FileNotFoundError
        If `src` does not exist, if a directory holds no ``*_GRX.dsv``, or if any one of the five
        channels is missing -- naming which.
    ValueError
        If a directory holds more than one export and no member file was named.
    """
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
        # Infer it from the gradient-x channel, which every export has. Taken from the real
        # filename rather than from the upper-cased lookup key, so the prefix keeps its own
        # casing -- otherwise it reaches error messages as a filename that does not exist.
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
    """Open a Pulseq ``.seq`` or a DSV export, in the representation they share.

    A ``.seq`` is read with ``gradient_mode="dsv"`` unless told otherwise: that is the mode which
    matches what a DSV export holds, so the two sources become comparable. Pass ``mode="pulseq"``
    when the comparison you want is against Pulseq's own ``waveforms_and_times()``.

    Parameters
    ----------
    src : str or Path
        A ``.seq`` file, a directory holding one DSV export, or any single ``*.dsv`` file
        belonging to one.
    mode : {'dsv', 'pulseq', 'centers'}, optional
        Gradient mode for a ``.seq``. Defaults to ``'dsv'``. A DSV export is already the
        commanded waveform, so anything but ``'dsv'`` is rejected rather than ignored.
    gamma : float, optional
        Gyromagnetic ratio (Hz/T) for a DSV export; defaults to `SCAN_GAMMA`. Rejected for a
        ``.seq``, which takes its scaling from the file.

    Returns
    -------
    Sequence or DsvSet
        Whichever `src` named. The 17 methods they share -- `gradient`, `gradients_dense`,
        `sampler`, the ``adc_*`` and ``trigger_*`` families, `duration` -- behave the same on
        both, so callers need not know which they have.

    Raises
    ------
    FileNotFoundError
        If `src` does not exist, or a DSV export is incomplete.
    ValueError
        If `src` is neither a ``.seq`` nor a DSV export, or an argument does not apply to the
        source named.

    Examples
    --------
    >>> wf = load("scan.seq")                     # doctest: +SKIP
    >>> wf.gradient_mode                          # doctest: +SKIP
    'dsv'
    >>> wf = load("export_dir/")                  # doctest: +SKIP
    >>> t, gx, gy, gz = wf.gradients_dense(10e-6) # doctest: +SKIP
    """
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
