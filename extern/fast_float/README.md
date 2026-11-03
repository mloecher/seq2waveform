# Vendored fast_float 8.1.0

Header-only, locale-independent `from_chars` for floating-point numbers.
Upstream: [fastfloat/fast_float](https://github.com/fastfloat/fast_float).

The headers are vendored so source builds do not need to download C++ dependencies.
The nine `.h` files were copied from conda-forge package `fast_float 8.1.0 h49e36cd_0`.
`LICENSE-MIT` was transcribed from upstream, retaining its wording and copyright notice.
Upstream also offers Apache-2.0 and Boost licenses; this project uses MIT. Upstream
credits code adapted from Google Wuffs by Nigel Tao.

## Updating

Replace all nine headers together from one upstream release or conda package. Keep
`LICENSE-MIT` with them and update the version and source recorded above.

```bash
pixi run test
pixi run python scripts/bench.py --only cones
```

The C++ test build also produces `bench_scan`. Run it on a `.seq` dataset to check
parsed shape values against `std::from_chars` and compare timings:

```bash
./build-tests/tests/Release/bench_scan ../seq2waveform/testdata/cones/cones.seq 9
```

Packagers can use an installed copy with `-DS2W_USE_SYSTEM_FAST_FLOAT=ON`, which
selects `find_package(FastFloat)` instead of these headers.
