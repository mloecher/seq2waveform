"""Digests for the golden-file tests: how a corpus result becomes a committed JSON record.

The corpus is ~1.6 GB and gitignored, so the snapshots cannot hold the arrays themselves. Each
column becomes a digest instead: an exact hash, so a one-ULP change fails, plus a count, the
non-finite tally, summary statistics and a handful of evenly spaced sample values. Those extras
are not the assertion -- the hash is -- they are there so a failure says *what* moved rather than
just "the hash changed".

Non-finite values are canonicalised before hashing, because a NaN's payload bits are not
guaranteed to match across compilers: they are replaced with 0.0 and their positions mixed in as a
separate class array, which keeps the check exact without depending on the bit pattern.

See `scripts/gen_golden.py` to write the snapshots and `tests/test_golden.py` to check them.
"""

from __future__ import annotations

import collections
import hashlib

import numpy as np

#: Evenly spaced values kept per column, for diagnosis only.
N_PROBE = 8

#: Relative tolerance for the fallback comparison when the hash does not match. It only decides
#: *how* a failure is described, never whether it passes -- unless `S2W_GOLDEN_RTOL` is set.
REPORT_RTOL = 1e-12


def _hash(*chunks: bytes) -> str:
    h = hashlib.blake2b(digest_size=16)
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _probe(a: np.ndarray) -> list:
    """Up to `N_PROBE` evenly spaced `[index, value]` pairs, endpoints included."""
    n = a.size
    if n == 0:
        return []
    idx = np.unique(np.linspace(0, n - 1, min(N_PROBE, n)).astype(np.int64))
    out = []
    for i in idx:
        v = a[i]
        out.append([int(i), None if not np.isfinite(v) else float(v)])
    return out


def digest(values) -> dict:
    """Snapshot one column of results.

    Parameters
    ----------
    values : array_like or list of str
        A 1-D numeric array, or a list of strings. Scalars are accepted and treated as
        length-1 arrays.

    Returns
    -------
    dict
        JSON-serialisable record. `hash` is the assertion; everything else is diagnostic.
    """
    if isinstance(values, (list, tuple)) and (len(values) == 0 or isinstance(values[0], str)):
        return _digest_strings(list(values))
    return _digest_numeric(np.atleast_1d(np.asarray(values)))


def _digest_strings(vals: list[str]) -> dict:
    """Strings keep their value counts rather than probes -- far more useful on a failure."""
    return {
        "kind": "str",
        "n": len(vals),
        "hash": _hash(b"\x00".join(s.encode("utf-8") for s in vals)),
        "unique": dict(sorted(collections.Counter(vals).items())),
    }


def _digest_numeric(a: np.ndarray) -> dict:
    """Hash the exact little-endian bytes; store stats and probes for diagnosis."""
    if a.dtype.kind in "iu":
        a = a.astype("<i8", copy=False)
        payload, bad = a.tobytes(), None
    else:
        a = a.astype("<f8", copy=False)
        finite = np.isfinite(a)
        if finite.all():
            payload, bad = a.tobytes(), None
        else:
            # Class array (1 nan, 2 +inf, 3 -inf) pins the positions; zeroed values keep the
            # hash independent of any particular NaN payload.
            cls = np.zeros(a.size, dtype=np.uint8)
            cls[np.isnan(a)] = 1
            cls[np.isposinf(a)] = 2
            cls[np.isneginf(a)] = 3
            payload = np.where(finite, a, 0.0).tobytes() + cls.tobytes()
            bad = {
                "nan": int((cls == 1).sum()),
                "posinf": int((cls == 2).sum()),
                "neginf": int((cls == 3).sum()),
            }

    out = {
        "kind": "int" if a.dtype.kind == "i" else "float",
        "n": int(a.size),
        "hash": _hash(payload),
        "probe": _probe(a),
    }
    if bad is not None:
        out["nonfinite"] = bad
    good = a[np.isfinite(a)] if a.dtype.kind == "f" else a
    if good.size:
        out["stats"] = {
            "min": float(good.min()),
            "max": float(good.max()),
            "sum": float(good.sum()),
        }
    return out


def _rel(got: float, want: float) -> float:
    """Relative difference, falling back to absolute when the reference is zero."""
    if want == got:
        return 0.0
    scale = max(abs(want), abs(got))
    return abs(got - want) / scale if scale else abs(got - want)


def worst_rel(got: dict, want: dict) -> float | None:
    """Largest relative disagreement across the diagnostic fields, or None if not comparable.

    Used only to describe a hash mismatch: a tiny value means the two runs agree numerically and
    the difference is almost certainly the compiler, not the code.
    """
    if got.get("kind") != want.get("kind") or got.get("n") != want.get("n"):
        return None
    worst = 0.0
    for key in ("min", "max", "sum"):
        g, w = got.get("stats", {}).get(key), want.get("stats", {}).get(key)
        if g is None or w is None:
            return None
        worst = max(worst, _rel(g, w))
    gp, wp = got.get("probe", []), want.get("probe", [])
    if len(gp) != len(wp):
        return None
    for (gi, gv), (wi, wv) in zip(gp, wp):
        if gi != wi or (gv is None) != (wv is None):
            return None
        if gv is not None:
            worst = max(worst, _rel(gv, wv))
    return worst


def compare(name: str, got: dict, want: dict, accept_rtol: float | None = None) -> str | None:
    """Check one column against its snapshot.

    Parameters
    ----------
    name : str
        Column key, used in the message.
    got, want : dict
        Digests from `digest`.
    accept_rtol : float, optional
        When given, a hash mismatch whose values agree to within this relative tolerance is
        accepted. This is a weaker check than the hash by construction -- it can only see the
        statistics and the probes, so it would also wave through a reordering -- and exists for
        one job: running snapshots taken on one compiler against another, once, to confirm the
        difference is float noise before regenerating them there. Leave it unset otherwise.

    Returns
    -------
    str or None
        A one-line description of the problem, or None if the column matches.
    """
    if got.get("n") != want.get("n"):
        return f"{name}: length {got.get('n')} != {want.get('n')}"
    if got.get("nonfinite") != want.get("nonfinite"):
        return f"{name}: non-finite tally {got.get('nonfinite')} != {want.get('nonfinite')}"
    if got["hash"] == want["hash"]:
        return None
    if want.get("kind") == "str":
        return f"{name}: strings differ; values now {got.get('unique')} vs {want.get('unique')}"

    rel = worst_rel(got, want)
    if rel is None:
        return f"{name}: hash differs and the digests are not comparable ({got} vs {want})"
    if accept_rtol is not None and rel <= accept_rtol:
        return None
    if rel == 0.0:
        return (
            f"{name}: hash differs although every statistic and sampled value is identical -- "
            "whatever moved is at a position the probes miss. A reordering, or a handful of "
            "values shifted by less than the sum resolves."
        )
    if rel <= REPORT_RTOL:
        return (
            f"{name}: hash differs, values agree to {rel:.3g} -- float noise rather than a code "
            "change. On a new compiler, confirm with S2W_GOLDEN_RTOL=1e-12 and regenerate the "
            "snapshots there."
        )
    return f"{name}: values differ, worst relative {rel:.3g} (over statistics and probes)"


# --------------------------------------------------------------------------------------------
# What gets snapshotted
# --------------------------------------------------------------------------------------------
#
# Chosen to cover what the cut-over left unchecked on real data -- triggers, RF event metadata,
# ADC event grouping after the MaxAdcSegmentLength merge, and the INF/DSV readers -- plus the
# gradient corners in all three modes and the flat k-space, which are the primary outputs and the
# code the performance work rewrote. See PORTING.md, "What the cut-over gave up".

GRAD_MODES = ("pulseq", "centers", "dsv")
AXES = ("x", "y", "z")
DSV_CHANNELS = ("grx", "gry", "grz", "rfd")


def _rows(out: dict, prefix: str, rows: list, fields: tuple) -> None:
    """Unzip a list of same-shaped tuples into one digest per field.

    Parameters
    ----------
    out : dict
        Digest map to write into.
    prefix : str
        Key prefix, e.g. ``"seq.trigger"``.
    rows : list of tuple
        The events, possibly empty.
    fields : tuple of (str, str)
        ``(field_name, dtype)`` per tuple position; dtype is ``"i8"``, ``"f8"`` or ``"str"``.
        Given explicitly so an empty dataset digests to the same kind as a populated one.
    """
    cols = list(zip(*rows)) if rows else [()] * len(fields)
    for (name, dtype), col in zip(fields, cols):
        key = f"{prefix}.{name}"
        out[key] = digest([str(v) for v in col]) if dtype == "str" else digest(
            np.asarray(list(col), dtype=dtype)
        )


def _lengths(events) -> np.ndarray:
    """Per-event sample counts -- the grouping, independent of the values."""
    return np.asarray([len(e) for e in events], dtype="i8")


def snapshot(seq_file: str, dsv_files=None) -> dict:
    """Compute every digested column for one dataset.

    Parameters
    ----------
    seq_file : str
        Path to the dataset's `.seq`.
    dsv_files : tuple of str, optional
        ``(grx, gry, grz, rfd, inf)``. When omitted only the `.seq` columns are produced.

    Returns
    -------
    dict
        Column key -> digest. Keys are stable; a new one is an addition, never a rename.
    """
    import seq2waveform as s2w

    out: dict = {}
    seq = s2w.Sequence(seq_file)
    out["seq.num_blocks"] = digest(np.asarray([seq.num_blocks], dtype="i8"))
    out["seq.duration_s"] = digest(seq.duration)

    _rows(out, "seq.trigger", seq.triggers(), (("channel", "i8"), ("time_s", "f8"),
                                               ("duration_s", "f8")))
    _rows(out, "seq.rf", seq.rf_events(), (("center_s", "f8"), ("freq_hz", "f8"),
                                           ("phase_rad", "f8"), ("use", "str")))

    events = seq.adc_events()
    out["seq.adc.lengths"] = digest(_lengths(events))
    out["seq.adc.times_s"] = digest(np.concatenate(events) if events else np.zeros(0))
    _rows(out, "seq.adc_meta", [(f, p) for _, f, p in seq.adc_events_meta()],
          (("freq_hz", "f8"), ("phase_rad", "f8")))
    _rows(out, "seq.adc_excitation", seq.adc_excitations(),
          (("center_s", "f8"), ("freq_hz", "f8"), ("phase_rad", "f8")))
    out["seq.adc_spec.reference_s"] = digest(
        np.asarray([ref for _, ref in seq.adc_specs()], dtype="f8")
    )

    # All three modes: the differential suite compared 24.8 M corners across them, and `centers`
    # and `dsv` are reachable from Python but checked by nothing else on real data.
    for mode in GRAD_MODES:
        seq.gradient_mode = mode
        for axis in AXES:
            t, v = seq.gradient(axis)
            out[f"seq.grad.{mode}.{axis}.t_s"] = digest(t)
            out[f"seq.grad.{mode}.{axis}.v_hz_per_m"] = digest(v)
    seq.gradient_mode = "pulseq"

    # The headline output, and what the allocation fixes rewrote.
    t, kx, ky, kz, offsets = seq.adc_kspace_flat()
    for name, arr in (("t_s", t), ("kx", kx), ("ky", ky), ("kz", kz), ("offsets", offsets)):
        out[f"seq.kspace_flat.{name}"] = digest(arr)

    if dsv_files:
        out.update(_dsv_snapshot(s2w, dsv_files))
    return out


def _dsv_snapshot(s2w, dsv_files) -> dict:
    """The DSV/INF reader columns, and the `DsvSet` built from them."""
    grx, gry, grz, rfd, inf = dsv_files
    out: dict = {}

    out["inf.lead_in_s"] = digest(s2w.inf_lead_in(inf))
    times, num_segments = s2w.read_inf_adc(inf)
    out["inf.adc.num_segments"] = digest(np.asarray([num_segments], dtype="i8"))
    out["inf.adc.times_s"] = digest(times)
    # Post-merge grouping: the MaxAdcSegmentLength merge is the one piece of INF logic with no
    # other check on real data at all.
    out["inf.adc_events.lengths"] = digest(_lengths(s2w.read_inf_adc_events(inf)))
    _rows(out, "inf.sync", s2w.read_inf_sync(inf),
          (("name", "str"), ("time_s", "f8"), ("duration_s", "f8")))
    _rows(out, "inf.rf", s2w.read_inf_rf(inf, rfd),
          (("center_s", "f8"), ("start_s", "f8"), ("duration_s", "f8"), ("flip_deg", "f8"),
           ("freq_hz", "f8"), ("phase_deg", "f8")))

    for chan, path in zip(DSV_CHANNELS, (grx, gry, grz, rfd)):
        t, v, raster, unit = s2w.read_dsv_pwl(path)
        out[f"dsv.{chan}.pwl.t_s"] = digest(t)
        out[f"dsv.{chan}.pwl.values"] = digest(v)
        out[f"dsv.{chan}.pwl.raster_s"] = digest(raster)
        out[f"dsv.{chan}.pwl.unit"] = digest([unit])
        dense, dense_raster, dense_unit = s2w.read_dsv(path)
        out[f"dsv.{chan}.dense.values"] = digest(dense)
        out[f"dsv.{chan}.dense.raster_s"] = digest(dense_raster)
        out[f"dsv.{chan}.dense.unit"] = digest([dense_unit])

    dsv = s2w.DsvSet(grx, gry, grz, rfd, inf)
    out["dsvset.duration_s"] = digest(dsv.duration)
    out["dsvset.gamma"] = digest(dsv.gamma)
    _rows(out, "dsvset.trigger", dsv.triggers(), (("channel", "i8"), ("time_s", "f8"),
                                                  ("duration_s", "f8")))
    _rows(out, "dsvset.adc_excitation", dsv.adc_excitations(),
          (("center_s", "f8"), ("freq_hz", "f8"), ("phase_rad", "f8")))
    events = dsv.adc_events()
    out["dsvset.adc.lengths"] = digest(_lengths(events))
    out["dsvset.adc.times_s"] = digest(np.concatenate(events) if events else np.zeros(0))
    for axis in AXES:
        t, v = dsv.gradient(axis)
        out[f"dsvset.grad.{axis}.t_s"] = digest(t)
        out[f"dsvset.grad.{axis}.v_hz_per_m"] = digest(v)
    return out
