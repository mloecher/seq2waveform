"""Time the public API on the real datasets, optionally against the Rust implementation.

Neither implementation caches anything, so every call below recomputes from the parsed blocks --
which is what makes the numbers comparable and also what makes repeated calls honest.

Usage
-----
    pixi run python scripts/bench.py                 # C++ only
    pixi run python scripts/bench.py --compare-rust  # alongside the Rust oracle
    pixi run python scripts/bench.py --only cones girf
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import seq2waveform_cpp as cpp

TESTDATA = pathlib.Path(os.environ.get("S2W_RUST_REPO", "../seq2waveform")) / "testdata"
DATASETS = ("cones", "epi", "epi_fatsat", "girf", "sync_osc")
DENSE_RASTER = 1e-4


def timed(fn, repeats):
    """Best-of-`repeats` wall time in ms, plus whatever the call returned.

    Minimum rather than mean: both implementations are deterministic, so spread here is scheduler
    noise, and the fastest run is the closest to the cost of the work itself.
    """
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best, out


def size_of(out):
    """A one-number summary of a result, so the two sides can be shown to have done equal work."""
    if out is None:
        return ""
    if hasattr(out, "size"):
        return f"{out.size:,}"
    if isinstance(out, tuple):
        return f"{sum(getattr(a, 'size', 0) for a in out):,}"
    if isinstance(out, list):
        return f"{len(out):,} ev"
    return ""


def seq_cases(mod, path, repeats):
    """Parse is timed on its own; the rest reuse one Sequence, which caches nothing."""
    yield "Sequence(path)", timed(lambda: mod.Sequence(path), repeats)
    seq = mod.Sequence(path)
    yield "gradient('x')", timed(lambda: seq.gradient("x"), repeats)
    yield "adc_times()", timed(lambda: seq.adc_times(), repeats)
    yield "adc_kspace_flat()", timed(lambda: seq.adc_kspace_flat(), max(1, repeats // 2))
    yield f"gradients_dense({DENSE_RASTER:g})", timed(
        lambda: seq.gradients_dense(DENSE_RASTER), max(1, repeats // 2)
    )


def dsv_paths(name):
    d = TESTDATA / name / "dsv"
    out = []
    for suffix in ("GRX", "GRY", "GRZ", "RFD", "INF"):
        hits = sorted(d.glob(f"*_{suffix}.dsv"))
        if not hits:
            return None
        out.append(str(hits[0]))
    return tuple(out)


def dsv_cases(mod, paths, repeats):
    yield "DsvSet(...)", timed(lambda: mod.DsvSet(*paths), repeats)
    s = mod.DsvSet(*paths)
    yield "  .gradient('x')", timed(lambda: s.gradient("x"), repeats)
    yield "  .adc_kspace_flat()", timed(lambda: s.adc_kspace_flat(), max(1, repeats // 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-rust", action="store_true", help="also time the Rust implementation")
    ap.add_argument("--only", nargs="*", default=None, metavar="DATASET")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dsv", action="store_true", help="include the DsvSet cases (slow)")
    args = ap.parse_args()

    rust = None
    if args.compare_rust:
        import seq2waveform as rust  # noqa: PLC0415  -- optional, and gone after the cut-over

    names = args.only or DATASETS
    hdr = f"{'':34s} {'C++ ms':>10s}"
    if rust:
        hdr += f" {'Rust ms':>10s} {'C++/Rust':>9s}"
    hdr += "   result"

    for name in names:
        seq = TESTDATA / name / f"{name}.seq"
        if not seq.exists():
            print(f"\n{name}: no .seq, skipped")
            continue
        print(f"\n=== {name}  ({seq.stat().st_size / 1e6:.1f} MB) ===")
        print(hdr)

        cases = [list(seq_cases(cpp, str(seq), args.repeats))]
        if rust:
            cases.append(list(seq_cases(rust, str(seq), args.repeats)))

        if args.dsv:
            paths = dsv_paths(name)
            if paths:
                cases[0] += list(dsv_cases(cpp, paths, 1))
                if rust:
                    cases[1] += list(dsv_cases(rust, paths, 1))

        for i, (label, (c_ms, c_out)) in enumerate(cases[0]):
            line = f"{label:34s} {c_ms:10.1f}"
            if rust:
                _, (r_ms, r_out) = cases[1][i]
                line += f" {r_ms:10.1f} {c_ms / r_ms:8.2f}x"
                if size_of(c_out) != size_of(r_out):
                    line += f"  !! sizes differ: {size_of(c_out)} vs {size_of(r_out)}"
            print(f"{line}   {size_of(c_out)}")


if __name__ == "__main__":
    main()
