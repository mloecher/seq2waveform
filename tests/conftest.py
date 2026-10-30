"""Shared test-corpus discovery.

The corpus is matched `.seq` / `.h5` / `.dsv` files, one directory per dataset, ~1.6 GB. It is
gitignored and lives outside the repo, so where it is has to be discovered rather than assumed.

Resolution order, first hit wins:

1. ``S2W_TESTDATA``, if set (relative values resolve against the project root, not the cwd).
2. ``testdata/`` in this repo.
3. ``../seq2waveform/testdata`` -- the Rust checkout that owned the corpus before the C++ port
   replaced it. Kept as a fallback so a checkout that has not moved the corpus yet still runs;
   drop this once the Rust repo is gone.

A run that reports *skipped* found no corpus. That is not a pass -- it means nothing was checked.
"""

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _find_testdata() -> Path:
    env = os.environ.get("S2W_TESTDATA")
    candidates = []
    if env:
        candidates.append(_resolve(env))
    candidates.append(PROJECT_ROOT / "testdata")
    candidates.append(_resolve("../seq2waveform/testdata"))
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    return candidates[0]


TESTDATA = _find_testdata()


def datasets(*required_suffixes: str) -> list[str]:
    """Dataset names under the corpus that have every one of `required_suffixes`.

    Parameters
    ----------
    *required_suffixes : str
        Extensions a dataset directory must contain as `<name><suffix>`, e.g. ``".seq"``,
        ``".h5"``. With none given, any directory holding `<name>.seq` counts.

    Returns
    -------
    list of str
        Sorted dataset names, empty if the corpus is missing.
    """
    if not TESTDATA.is_dir():
        return []
    want = required_suffixes or (".seq",)
    return sorted(
        d.name
        for d in TESTDATA.iterdir()
        if d.is_dir() and all((d / f"{d.name}{sfx}").exists() for sfx in want)
    )


def seq_path(name: str) -> str:
    """The `.seq` path for a dataset name."""
    return str(TESTDATA / name / f"{name}.seq")


def dsv_paths(name: str):
    """The five DSV/INF paths a `DsvSet` needs, or None if the dataset lacks any of them.

    Returns
    -------
    tuple of str, or None
        ``(grx, gry, grz, rfd, inf)``.
    """
    d = TESTDATA / name / "dsv"
    if not d.is_dir():
        return None
    out = []
    for suffix in ("GRX", "GRY", "GRZ", "RFD", "INF"):
        hits = sorted(d.glob(f"*_{suffix}.dsv"))
        if not hits:
            return None
        out.append(str(hits[0]))
    return tuple(out)


def requires_corpus(names) -> pytest.MarkDecorator:
    """Skip marker for a module whose datasets list came back empty."""
    return pytest.mark.skipif(
        not names, reason=f"no matching datasets under {TESTDATA} (set S2W_TESTDATA)"
    )
