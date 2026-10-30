"""Golden-file tests: the output on the real corpus, pinned to committed snapshots.

These exist to replace what the cut-over removed. The differential suite asserted bit-exact
equality against an independent implementation; nothing can bring that back, but it is not the
only thing that suite was doing. Most of its value in practice was *change detection* -- noticing
that a refactor moved a number -- and a snapshot does that just as well, because the reference is
the output everything else was validated against at the moment parity was proven.

What it does not do is tell you the numbers are right. `test_h5.py` is the outside answer.
A snapshot only says "the same as before", so the snapshots themselves must only ever be
regenerated deliberately: see `scripts/gen_golden.py`.

The columns deliberately cover what nothing else checks on real data -- triggers, RF event
metadata, ADC grouping after the `MaxAdcSegmentLength` merge, the INF and DSV readers, and the
`centers` / `dsv` gradient modes -- plus the gradient corners and flat k-space, which are the
primary outputs and the code the allocation fixes rewrote. See PORTING.md, "What the cut-over
gave up".

A run that reports *skipped* checked nothing; see `conftest.py`.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

from conftest import TESTDATA, datasets, dsv_paths, seq_path
from golden import compare, digest, snapshot

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"

#: Accept a hash mismatch whose values agree this closely. Unset by default: the snapshots are
#: bit-exact on the platform that wrote them. Set it when first running snapshots from one
#: compiler against another (the Windows set against a macOS build, say) -- a pass then says the
#: difference is float noise, and the snapshots can be regenerated there with confidence.
ACCEPT_RTOL = float(os.environ["S2W_GOLDEN_RTOL"]) if "S2W_GOLDEN_RTOL" in os.environ else None

COMMITTED = sorted(p.stem for p in GOLDEN_DIR.glob("*.json")) if GOLDEN_DIR.is_dir() else []
CORPUS = datasets(".seq")
ALL_NAMES = sorted(set(COMMITTED) | set(CORPUS))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_output_matches_the_committed_snapshot(name):
    """Every digested column for one dataset, against its snapshot.

    Reports every mismatching column at once rather than stopping at the first: when a shared
    helper changes, the set of columns that moved is the diagnosis.
    """
    if name not in CORPUS:
        pytest.skip(f"{name} has a snapshot but is not in the corpus at {TESTDATA}")
    path = GOLDEN_DIR / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"{name} is in the corpus but has no snapshot; run `pixi run golden-update`")

    want = json.loads(path.read_text(encoding="utf-8"))
    got = snapshot(seq_path(name), dsv_paths(name))

    problems = []
    for key in sorted(set(want) - set(got)):
        problems.append(f"{key}: in the snapshot but no longer produced")
    for key in sorted(set(got) - set(want)):
        problems.append(f"{key}: produced but not in the snapshot (run `pixi run golden-update`)")
    for key in sorted(set(got) & set(want)):
        bad = compare(key, got[key], want[key], accept_rtol=ACCEPT_RTOL)
        if bad:
            problems.append(bad)

    assert not problems, f"{len(problems)} of {len(want)} columns differ for {name}:\n" + "\n".join(
        "  " + p for p in problems
    )


@pytest.mark.skipif(not COMMITTED, reason="no snapshots committed")
@pytest.mark.parametrize("mode", ["centers", "dsv"])
def test_each_gradient_mode_differs_from_pulseq_somewhere(mode):
    """The snapshots carry all three gradient modes; this checks that is worth the bytes.

    `centers` and `dsv` only diverge from `pulseq` where a sequence actually has centers-sampled
    arbitrary shapes -- on an all-trapezoid sequence the modes agree exactly, by design. So most
    datasets duplicate, and if *every* one did, 12 of the 18 gradient columns would be pinning
    nothing and the two alternate modes would be untested. Reads the committed files only, so it
    needs no corpus.
    """
    same = []
    for name in COMMITTED:
        snap = json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))
        for axis in "xyz":
            base = snap[f"seq.grad.pulseq.{axis}.v_hz_per_m"]
            other = snap[f"seq.grad.{mode}.{axis}.v_hz_per_m"]
            if base["n"] == 0:
                continue  # an unused axis is empty in every mode; nothing to compare
            same.append(base["hash"] == other["hash"])
    assert same, "no non-empty gradient axis in any snapshot"
    assert not all(same), (
        f"'{mode}' mode is identical to 'pulseq' on every axis of every dataset, so nothing in "
        f"the corpus exercises it. Add a dataset with centers-sampled arbitrary gradients, or "
        f"stop snapshotting the mode and say so."
    )


@pytest.mark.skipif(not CORPUS, reason=f"no corpus under {TESTDATA}")
def test_every_corpus_dataset_has_a_snapshot():
    """A dataset with no snapshot is unchecked, and a skip is easy to miss in a long run."""
    missing = [n for n in CORPUS if n not in COMMITTED]
    assert not missing, (
        f"no committed snapshot for {missing}; run `pixi run golden-update` and commit "
        f"tests/golden/*.json, or these datasets are checked by nothing here"
    )


# -------------------------------------------------------------------------------------------
# Negative controls for the digest machinery itself.
#
# A snapshot test that cannot fail is worse than no test, and unlike the comparisons above these
# need no corpus -- so they run everywhere, including a fresh clone, and prove the digest is
# actually sensitive before anyone trusts a green run.
# -------------------------------------------------------------------------------------------


def _differs(a, b, **kw):
    return compare("probe", digest(a), digest(b), **kw)


def test_a_one_ulp_change_is_caught():
    a = np.linspace(0.0, 1.0, 1000)
    b = a.copy()
    b[500] = np.nextafter(b[500], np.inf)
    assert _differs(a, b) is not None


def test_a_reordering_is_caught():
    """Sum, min and max all survive a swap; the hash is what notices."""
    a = np.linspace(0.0, 1.0, 1000)
    b = a.copy()
    b[[100, 900]] = b[[900, 100]]
    assert _differs(a, b) is not None


def test_a_length_change_is_caught():
    a = np.arange(10, dtype=np.int64)
    assert _differs(a, a[:-1]) is not None


def test_a_moved_nan_is_caught():
    """NaN payload bits are canonicalised away, so only the positions can be compared."""
    a = np.array([np.nan, 1.0, 2.0])
    b = np.array([1.0, np.nan, 2.0])
    assert _differs(a, b) is not None


def test_nan_becoming_a_number_is_caught():
    a = np.array([np.nan, 1.0])
    b = np.array([0.0, 1.0])
    assert _differs(a, b) is not None


def test_a_string_change_is_caught():
    assert _differs(["excitation", "refocusing"], ["excitation", "excitation"]) is not None


def test_identical_input_passes():
    a = np.linspace(-3.0, 7.0, 500)
    a[7] = np.nan
    assert _differs(a, a.copy()) is None
    assert _differs(["a", "b"], ["a", "b"]) is None


def test_the_rtol_escape_hatch_accepts_float_noise_but_not_a_real_change():
    """What a macOS run would use: 1-ULP compiler differences pass, a real change still fails."""
    a = np.linspace(1.0, 2.0, 1000)
    noisy = np.nextafter(a, np.inf)
    assert _differs(a, noisy) is not None
    assert _differs(a, noisy, accept_rtol=1e-12) is None

    shifted = a * 1.0001
    assert _differs(a, shifted, accept_rtol=1e-12) is not None
