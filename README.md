# seq2waveform-cpp

C++ port of [seq2waveform](../seq2waveform): read Pulseq `.seq` files and Siemens `.dsv`/`.inf`
files into waveform arrays, with k-space sampling at arbitrary times.

Built with nanobind + scikit-build-core, so the Python type stubs (docstrings and `NDArray`
annotations included) are regenerated from the compiled module on every build.

**Status: in progress.** The time type, the piecewise-linear core, the Pulseq `.seq` reader and
waveform expansion are ported and verified bit-exact against the Rust implementation — 76
differential tests covering every event library, every shape, the parse-error messages, and 24.8 M
gradient corners across all three gradient modes. Parsing runs at parity with the Rust version.
The sampler/k-space and the DSV/INF readers are not started, and the reference-h5 comparison waits
on the sampler.

## Build and test

```bash
pixi run build          # editable install, regenerates seq2waveform_cpp/_core.pyi
pixi run test-cpp       # C++ core tests via ctest
pixi run test-parity    # C++ vs the Rust implementation on the real datasets
pixi run test           # both
```

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
