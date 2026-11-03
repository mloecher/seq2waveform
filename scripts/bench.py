"""Time the public API on real datasets. Each call recomputes from parsed data.

Usage:
    pixi run bench
    pixi run python scripts/bench.py --only cones girf --repeats 8
    pixi run python scripts/bench.py --dsv

For native C++ timings, use tests/bench_expand.cpp.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# The corpus lives outside the repo and conftest.py owns finding it; reuse that rather than
# duplicating the fallback chain here.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from conftest import TESTDATA, datasets, dsv_paths, seq_path  # noqa: E402

import seq2waveform as s2w  # noqa: E402

DENSE_RASTER = 1e-4


def timed(fn, repeats):
    """Best-of-`repeats` wall time in ms, plus whatever the call returned.

    Minimum rather than mean: the computation is deterministic, so spread here is scheduler noise
    and the fastest run is the closest to the cost of the work.
    """
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best, out


def size_of(out):
    """A one-number summary of a result, so a run can be seen to have done the expected work."""
    if out is None:
        return ""
    if hasattr(out, "size"):
        return f"{out.size:,}"
    if isinstance(out, tuple):
        return f"{sum(getattr(a, 'size', 0) for a in out):,}"
    if isinstance(out, list):
        return f"{len(out):,} ev"
    return ""


def seq_cases(path, repeats):
    """Parse is timed on its own; the rest reuse one Sequence, which caches nothing."""
    yield "Sequence(path)", timed(lambda: s2w.Sequence(path), repeats)
    seq = s2w.Sequence(path)
    yield "gradient('x')", timed(lambda: seq.gradient("x"), repeats)
    yield "adc_times()", timed(lambda: seq.adc_times(), repeats)
    yield "adc_kspace_flat()", timed(lambda: seq.adc_kspace_flat(), max(1, repeats // 2))
    yield f"gradients_dense({DENSE_RASTER:g})", timed(
        lambda: seq.gradients_dense(DENSE_RASTER), max(1, repeats // 2)
    )


def dsv_cases(paths, repeats):
    yield "DsvSet(...)", timed(lambda: s2w.DsvSet(*paths), repeats)
    s = s2w.DsvSet(*paths)
    yield "  .gradient('x')", timed(lambda: s.gradient("x"), repeats)
    yield "  .adc_kspace_flat()", timed(lambda: s.adc_kspace_flat(), max(1, repeats // 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, metavar="DATASET")
    ap.add_argument("--repeats", type=int, default=3, help="heavy calls get half this many")
    ap.add_argument("--dsv", action="store_true", help="include the DsvSet cases (slow)")
    args = ap.parse_args()

    names = args.only or datasets(".seq")
    if not names:
        print(f"no datasets found under {TESTDATA} (set S2W_TESTDATA)")
        return 1

    for name in names:
        path = pathlib.Path(seq_path(name))
        if not path.exists():
            print(f"\n{name}: no .seq, skipped")
            continue
        print(f"\n=== {name}  ({path.stat().st_size / 1e6:.1f} MB) ===")
        print(f"{'':34s} {'ms':>10s}   result")

        cases = list(seq_cases(str(path), args.repeats))
        if args.dsv:
            paths = dsv_paths(name)
            if paths:
                cases += list(dsv_cases(paths, 1))

        for label, (ms, out) in cases:
            print(f"{label:34s} {ms:10.1f}   {size_of(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
