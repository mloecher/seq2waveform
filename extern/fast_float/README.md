# Vendored: fast_float 8.1.0

Daniel Lemire's header-only `from_chars` for `float`/`double`.
Upstream: <https://github.com/fastfloat/fast_float> — note **no hyphen**; `lemire/fast_float` is a
fork, and its `LICENSE-MIT` carries no copyright line.

## Why it is vendored rather than fetched

The main build target is cibuildwheel wheels for PyPI, and a vendored copy is the only option that
needs no network at build time. That matters in three places, in increasing order of importance:

1. CI does ~20 builds per release (5 Python versions x 3 platforms, plus a second macOS arch). With
   `FetchContent` each one is a download that can fail, and a half-failed matrix is a release with
   some wheels missing.
2. **The sdist has to build offline.** Anyone on a platform or Python version we do not ship a wheel
   for falls back to building from source, and plenty of those builds have no network: corporate
   mirrors, `--no-build-isolation`, Docker steps with networking off.
3. **conda-forge builds are offline by design.** A `FetchContent` build simply cannot pass there, so
   the downstream conda package would be blocked on removing it again.

There is also a reason specific to this project: the port's whole test strategy is bit-exact
differential comparison against the Rust implementation. A float parser that can change underneath
us is a parity hazard, so pinning it by copying it in is a feature, not a cost. Any change arrives
as a reviewable diff.

## Provenance of these files, exactly

- **The nine `.h` files** are byte-identical copies (verified with `cmp`) of the conda-forge package
  `fast_float 8.1.0 h49e36cd_0`, taken from `.pixi/envs/default/Library/include/fast_float/`.
- **`LICENSE-MIT` was transcribed, not byte-copied.** The conda-forge package ships no licence file
  and the headers carry no licence preamble, so the text here was taken from the canonical repo's
  `LICENSE-MIT` on `main` — read from the rendered GitHub page, so its line wrapping is standard MIT
  rather than upstream's narrower wrapping. The wording and the copyright line
  (`Copyright (c) 2021 The fast_float authors`, corroborated by the Apache file's appendix) are what
  the licence requires; if you want the bytes, replace it:

  ```bash
  curl -L -o extern/fast_float/LICENSE-MIT \
    https://github.com/fastfloat/fast_float/raw/refs/heads/main/LICENSE-MIT
  ```

- Upstream offers the library under **Apache-2.0 OR MIT OR BOOST**. We take it under MIT, so only
  that file is kept here.
- Upstream credits code adapted from **Google Wuffs** (Nigel Tao), originally published under
  Apache-2.0.

## Updating

Upstream was at v8.3.2 when this was vendored; 8.1.0 is simply what conda-forge had. To move:

```bash
pixi add fast_float                       # or download the release's single-header fast_float.h
cp .pixi/envs/default/Library/include/fast_float/*.h extern/fast_float/
pixi remove fast_float                    # the vendored copy is what builds; keep the env lean
pixi run test-cpp && pixi run test-parity # the parity suite is the real check

# then confirm the new version is still bit-identical:
pixi run -- cmake --build build-tests --config Release --target bench_scan
./build-tests/tests/Release/bench_scan ../seq2waveform/testdata/cones/cones.seq 9
```

`tests/bench_scan.cpp` is the guard: it asserts fast_float and this toolchain's `std::from_chars`
return identical bit patterns for every `[SHAPES]` token in a dataset. It found 1,289,657 of 1,289,657
identical across the five datasets, which is what makes the swap safe for parity.

To build against a system copy instead of this one, configure with
`-DS2W_USE_SYSTEM_FAST_FLOAT=ON`; CMake then takes it from `find_package(FastFloat)`. Packagers who
unbundle will want that. Correct rounding is unique, so a different version returns the same doubles.
