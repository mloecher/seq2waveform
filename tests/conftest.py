"""Shared test-dataset discovery, over two sources.

A dataset is a directory holding `<name>.seq` and, depending on the source, `<name>.h5` and a
`dsv/` export.

**The corpus** is the full set, ~1.6 GB, and lives outside the repo, so where it is has to be
discovered rather than assumed. Resolution order, first hit wins:

1. ``S2W_TESTDATA``, if set (relative values resolve against the project root, not the cwd).
2. ``testdata/`` in this repo.
3. ``../seq2waveform/testdata`` -- the Rust checkout that owned the corpus before the C++ port
   replaced it. Kept as a fallback so a checkout that has not moved the corpus yet still runs;
   drop this once the Rust repo is gone.

**The bundled set** is `tests/data/`: two datasets committed to the repo, so a fresh clone and a
CI runner check real numbers instead of skipping everything. It carries the `.seq` and the Pulseq
`.h5` -- which is what makes `test_h5.py`, the only outside answer in the suite, run anywhere --
but no DSV export. Those files are a vendor simulator's output and stay out of a public repo;
`test_dsv.cpp` and `test_inf.cpp` cover those readers from minimal files written to the format.

The corpus wins for a name both sources hold, since it also has the DSV export. So the bundled
files change nothing about a run that has the corpus.

A run that reports *skipped* checked less than it looks like, which is what `test_datasets.py`
is for: it asserts the bundled set was found, so a skip there is a resolution bug and not a
missing corpus.
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

#: The datasets committed to the repo. Searched after the corpus; see the module docstring.
BUNDLED = PROJECT_ROOT / "tests" / "data"

#: Both sources, in the order a dataset name is looked up.
SOURCES = (TESTDATA, BUNDLED)


def _holds(root: Path, name: str, suffixes) -> bool:
    d = root / name
    return d.is_dir() and all((d / f"{name}{sfx}").exists() for sfx in suffixes)


def dataset_root(name: str, *required_suffixes: str) -> Path | None:
    """Which source holds `name`, corpus first, or None if neither does.

    Parameters
    ----------
    name : str
        Dataset name.
    *required_suffixes : str
        Extensions the directory must contain as `<name><suffix>`. Defaults to ``".seq"``.

    Returns
    -------
    Path or None
    """
    want = required_suffixes or (".seq",)
    return next((root for root in SOURCES if _holds(root, name, want)), None)


def datasets(*required_suffixes: str) -> list[str]:
    """Dataset names, from either source, that have every one of `required_suffixes`.

    Parameters
    ----------
    *required_suffixes : str
        Extensions a dataset directory must contain as `<name><suffix>`, e.g. ``".seq"``,
        ``".h5"``. With none given, any directory holding `<name>.seq` counts.

    Returns
    -------
    list of str
        Sorted dataset names, empty if neither source has a match.
    """
    want = required_suffixes or (".seq",)
    found: set[str] = set()
    for root in SOURCES:
        if root.is_dir():
            found |= {d.name for d in root.iterdir() if _holds(root, d.name, want)}
    return sorted(found)


def seq_path(name: str) -> str:
    """The `.seq` path for a dataset name."""
    return str((dataset_root(name) or TESTDATA) / name / f"{name}.seq")


def h5_path(name: str) -> Path:
    """The Pulseq reference `.h5` path for a dataset name."""
    return (dataset_root(name, ".seq", ".h5") or TESTDATA) / name / f"{name}.h5"


def dsv_paths(name: str):
    """The five DSV/INF paths a `DsvSet` needs, or None if the dataset lacks any of them.

    A bundled dataset never has them; see the module docstring.

    Returns
    -------
    tuple of str, or None
        ``(grx, gry, grz, rfd, inf)``.
    """
    root = dataset_root(name)
    if root is None:
        return None
    d = root / name / "dsv"
    if not d.is_dir():
        return None
    out = []
    for suffix in ("GRX", "GRY", "GRZ", "RFD", "INF"):
        hits = sorted(d.glob(f"*_{suffix}.dsv"))
        if not hits:
            return None
        out.append(str(hits[0]))
    return tuple(out)


def dsv_dir(name: str) -> Path | None:
    """The dataset's `dsv/` directory, or None if it has no export."""
    paths = dsv_paths(name)
    return None if paths is None else Path(paths[0]).parent


def requires_corpus(names) -> pytest.MarkDecorator:
    """Skip marker for a module whose datasets list came back empty."""
    return pytest.mark.skipif(
        not names,
        reason=f"no matching datasets in {TESTDATA} (set S2W_TESTDATA) or {BUNDLED}",
    )
