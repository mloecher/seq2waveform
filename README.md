# seq2waveform-cpp

C++ port of [seq2waveform](../seq2waveform): read Pulseq `.seq` files and Siemens `.dsv`/`.inf`
files into waveform arrays, with k-space sampling at arbitrary times.

Built with nanobind + scikit-build-core, so the Python type stubs (docstrings and `NDArray`
annotations included) are regenerated from the compiled module on every build.

**Status: feature-complete; cut-over pending.** Every module is ported — the time type, the
piecewise-linear core, the Pulseq `.seq` reader, waveform expansion, the sampler/k-space, the
Siemens `.dsv` and INF readers, and `DsvSet` behind the same `WaveformSource` interface as
`Sequence`, so a Siemens measurement set is a drop-in for a `.seq`.

All of it is verified bit-exact against the Rust implementation by 284 differential tests, and
independently against Pulseq's own reference `.h5` at a 1e-4 bar (worst actual margin 4.3e-6).
Parsing runs at parity with the Rust version.

What remains is the cut-over: renaming the package to `seq2waveform`, dropping the oracle and the
parity suite, and moving the dataset tests and corpus over. See
[PORTING.md](PORTING.md).

## Build and test

```bash
pixi run build          # editable install, regenerates seq2waveform_cpp/_core.pyi
pixi run test-cpp       # C++ core tests via ctest
pixi run test-parity    # C++ vs the Rust implementation on the real datasets, exactly
pixi run test-h5        # C++ vs Pulseq's own reference h5 at 1e-4 (no oracle needed)
pixi run test           # all three
```

`test-parity` only proves the two implementations agree; `test-h5` is the only check that does not
depend on the Rust code being right, so it is the one to run after changing a waveform.

Requires a system Visual Studio (MSVC) install on Windows; `cmake` and `nanobind` come from the
pixi environment. The one third-party dependency, [fast_float](extern/fast_float/README.md), is
vendored so that wheels, the sdist and a downstream conda recipe all build without network access.

## The Rust repo is still required

During the port, the Rust implementation is the oracle: every ported module is checked for exact
equality against it on real data, rather than against re-derived ground truth. The Rust checkout
also owns the test corpus — `testdata/` is gitignored, so it lives only there.

Both are found via `S2W_RUST_REPO`, which defaults to `../seq2waveform`. Set it in
`pyproject.toml` under `[tool.pixi.activation.env]` if your checkouts sit elsewhere.

A parity run that reports **skipped** found neither the oracle nor the corpus — treat it as a
failure, not a pass.

## Porting

[PORTING.md](PORTING.md) is the reference: project layout, the test corpus, validation
thresholds, every edge case the Rust version earned the hard way, the known gaps, the port order,
and a kickoff prompt to paste into a fresh session.
