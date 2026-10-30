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

wf = s2w.load("scan.seq")                         # or a Siemens DSV export directory
t, gx, gy, gz = wf.gradients_dense(10e-6)         # uniform raster, Hz/m
t, kx, ky, kz, offsets = wf.adc_kspace_flat()     # every ADC sample, 1/m
```

`load` takes either source and reads a `.seq` in `gradient_mode="dsv"`, the one representation the
two share, so code on top of it does not know which it was handed. The constructors are still
there when you want the other modes — `Sequence("scan.seq", gradient_mode="pulseq")` is what
matches Pulseq's own reference `.h5`.

Built with nanobind + scikit-build-core, so the Python type stubs — docstrings and `NDArray`
annotations included — are regenerated from the compiled module on every build and committed, which
is why an editor can show the full API without building first.

**[USAGE.md](USAGE.md) is the map of the API**: which method to call for which quantity at which
sampling site, what `gradient_mode` changes (measured), and why GIRF work wants `"dsv"` so that a
Pulseq `.seq` and a Siemens DSV export land in the same representation.

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
pixi run test-golden      # every output column against a committed digest
pixi run test             # all four
pixi run bench            # timings; see PORTING.md
```

Three different questions, and none of them answers another:

- **`test-h5` — is it right?** The only check that does not depend on this library's own
  arithmetic, since the `.h5` is written by Pulseq itself. Run it after changing a waveform. It is
  a tolerance check (1e-4) on gradients, ADC times and excitation centers.
- **`test-invariants` — do the paths agree?** Pins this library's routes to *each other*: sorted vs
  unsorted sampling, batched vs pointwise moments, flat vs per-event k-space, the `.seq` route vs
  the DSV route. Exact, and what catches a change that moves one half of a pair.
- **`test-golden` — did anything move?** 97 digested output columns per dataset against
  `tests/golden/*.json`: triggers, RF event metadata, ADC grouping, the INF and DSV readers, all
  three gradient modes, and the flat k-space. Bit-exact, and the only check that covers the first
  four of those on real data at all.

The snapshots hold hashes rather than arrays (the corpus is too big to commit), plus counts,
statistics and sampled values so a failure says *what* moved. Regenerate them with
`pixi run golden-update` only when the change in output is intended — a snapshot updated to make a
red test green is worth less than no snapshot — and review the diff: a one-line change to a reader
should move one or two columns, not ninety.

### The test corpus

Matched `.seq` / `.h5` / `.dsv` files, one directory per dataset, ~1.6 GB — so it is gitignored and
lives outside the repo. `tests/conftest.py` finds it via `S2W_TESTDATA`, then `./testdata`, then
`../seq2waveform/testdata` (the retired Rust checkout that used to own it).

A run that reports **skipped** found no corpus. Treat that as a failure, not a pass: it means
nothing was checked.

### Continuous integration

`.github/workflows/ci.yml` runs on every push: the C++ core and its ctest suites on gcc, clang
(both macOS architectures) and MSVC, the bindings compiled and installed on all three platforms,
and the sdist checked to build from source alone. `wheels.yml` builds the full matrix — CPython
3.10-3.14 plus 3.14 free-threaded, on Linux x86_64, macOS x86_64 and arm64, Windows x64, each
architecture on its own native runner — on a `v*` tag or on demand from the Actions tab.

The extension declares that it does not need the GIL, so a free-threaded interpreter keeps it
disabled; PORTING.md's *Free-threaded Python* records what was audited to support that.

CI has no corpus, so only the tests marked `no_corpus` run there; everything that reads a dataset
skips. **A green CI run does not stand in for `pixi run test` on a machine that has the corpus.**
See PORTING.md's *Continuous integration* for what each job does and does not prove.

## History

This started as a C++ port of a Rust implementation, which was the reference until the port
reached full parity — 284 differential tests asserting bit-exact equality on real data. The
cut-over retired the Rust crate and with it those tests, since they compared against something that
no longer exists; the oracle-free half of them became `tests/test_invariants.py`.

[PORTING.md](PORTING.md) is what that left behind, and it is still the reference worth reading
before changing anything: the edge cases each implementation earned the hard way, the validation
thresholds and why they are what they are, the measured performance work, and the remaining known
gaps.
