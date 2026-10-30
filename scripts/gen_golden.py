"""Regenerate the committed golden snapshots under `tests/golden/`.

Run this only when a change to the output is *intended*, and say in the commit message why the
numbers moved. A snapshot that gets regenerated to make a red test green is worse than no snapshot
at all, so the diff is the review: a one-line change to a reader should move one or two columns,
not ninety.

Usage
-----
    pixi run golden-update
    pixi run python scripts/gen_golden.py --only cones
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# The corpus lives outside the repo and conftest.py owns finding it.
TESTS = pathlib.Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS))
from conftest import TESTDATA, datasets, dsv_paths, seq_path  # noqa: E402
from golden import snapshot  # noqa: E402

GOLDEN_DIR = TESTS / "golden"


def write(name: str) -> tuple[int, int, float]:
    """Snapshot one dataset and write its JSON. Returns (columns, bytes, seconds)."""
    t0 = time.perf_counter()
    snap = snapshot(seq_path(name), dsv_paths(name))
    elapsed = time.perf_counter() - t0
    GOLDEN_DIR.mkdir(exist_ok=True)
    path = GOLDEN_DIR / f"{name}.json"
    # sort_keys and a trailing newline so a regenerated file diffs line-by-line against the old.
    text = json.dumps(snap, indent=1, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(snap), len(text), elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, metavar="DATASET")
    args = ap.parse_args()

    names = args.only or datasets(".seq")
    if not names:
        print(f"no datasets found under {TESTDATA} (set S2W_TESTDATA)", file=sys.stderr)
        return 1

    print(f"corpus: {TESTDATA}")
    for name in names:
        if not dsv_paths(name):
            print(f"  {name}: no DSV set, .seq columns only")
        cols, size, elapsed = write(name)
        print(f"  {name:14s} {cols:4d} columns  {size / 1024:6.1f} KiB  {elapsed:6.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
