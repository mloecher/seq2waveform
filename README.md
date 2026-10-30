# seq2waveform

Read Pulseq `.seq` files and Siemens `.dsv`/`.inf` files into waveform arrays, and sample k-space
at arbitrary times.

Two sources, one interface. `Sequence` reads a Pulseq `.seq` and expands its blocks; `DsvSet` reads
a Siemens measurement's gradient channels plus its `.inf`. Both then offer the same sampling and
k-space API, so a Siemens set is a drop-in for a `.seq` — which is what lets you ask whether what
the scanner played matches what Pulseq asked for.

Times are int64 picoseconds internally. That is what makes a delay sweep bit-exact no matter where
in the sequence it lands: absolute times reach tens of seconds, where a double resolves only a few
femtoseconds, so interpolating with large operands would make the same block sample differently
early and late. Values stay float64.

```python
import seq2waveform as s2w

seq = s2w.Sequence("scan.seq")
t, gx, gy, gz = seq.gradients_dense(10e-6)        # uniform raster, Hz/m
t, kx, ky, kz, offsets = seq.adc_kspace_flat()    # every ADC sample, 1/m

dsv = s2w.DsvSet("..._GRX.dsv", "..._GRY.dsv", "..._GRZ.dsv", "..._RFD.dsv", "..._INF.dsv")
t, kx, ky, kz, offsets = dsv.adc_kspace_flat()    # the same call, the Siemens side
```

Built with nanobind + scikit-build-core, so the Python type stubs — docstrings and `NDArray`
annotations included — are regenerated from the compiled module on every build and committed, which
is why an editor can show the full API without building first.

## Install

```bash
pixi run build          # editable install; regenerates seq2waveform/_core.pyi
```

Requires a system Visual Studio (MSVC) install on Windows; `cmake` and `nanobind` come from the
pixi environment. The one third-party dependency, [fast_float](extern/fast_float/README.md), is
vendored, so wheels, the sdist and the conda recipe all build with no network access.

For a redistributable build:

```bash
pixi run build-wheel    # dist/*.whl
pixi run conda-build    # wraps that wheel into a local conda channel under output/
```

## Test

```bash
pixi run test-cpp         # the C++ core, via ctest
pixi run test-h5          # against Pulseq's own reference .h5, at 1e-4
pixi run test-invariants  # internal invariants on the real datasets
pixi run test             # all three
pixi run bench            # timings; see PORTING.md
```

`test-h5` is the one that does not depend on this library's own arithmetic being right — the `.h5`
is written by Pulseq itself — so it is the one to run after changing a waveform.
`test-invariants` is its complement: it pins this library's paths to *each other* (sorted vs
unsorted sampling, flat vs per-event k-space, the `.seq` route vs the DSV route), which is what
catches a change that moves one half of a pair. Neither sees what the other does.

### The test corpus

Matched `.seq` / `.h5` / `.dsv` files, one directory per dataset, ~1.6 GB — so it is gitignored and
lives outside the repo. `tests/conftest.py` finds it via `S2W_TESTDATA`, then `./testdata`, then
`../seq2waveform/testdata` (the retired Rust checkout that used to own it).

A run that reports **skipped** found no corpus. Treat that as a failure, not a pass: it means
nothing was checked.

## History

This started as a C++ port of a Rust implementation, which was the reference until the port
reached full parity — 284 differential tests asserting bit-exact equality on real data. The
cut-over retired the Rust crate and with it those tests, since they compared against something that
no longer exists; the oracle-free half of them became `tests/test_invariants.py`.

[PORTING.md](PORTING.md) is what that left behind, and it is still the reference worth reading
before changing anything: the edge cases each implementation earned the hard way, the validation
thresholds and why they are what they are, the measured performance work, and the remaining known
gaps.
