# seq2waveform

Read Pulseq `.seq` files and Siemens `.dsv`/`.inf` files into waveform arrays, and sample k-space
at arbitrary times.

Two sources, one interface: a Siemens export is a drop-in for a `.seq`, which is what lets you ask
whether what the scanner played matches what Pulseq asked for.

```python
import seq2waveform as s2w

wf = s2w.load("scan.seq")                         # or a Siemens DSV export directory
t, gx, gy, gz = wf.gradients_dense(10e-6)         # uniform raster, Hz/m
t, kx, ky, kz, offsets = wf.adc_kspace_flat()     # every ADC sample, 1/m
```

Per readout, with k-space referenced to the excitation:

```python
sm = wf.sampler()
for times, ref in wf.adc_specs():
    t, kx, ky, kz = sm.kspace(times, ref)
```

`load` reads a `.seq` in `gradient_mode="dsv"`, the representation a Siemens export also holds, so
code on top of it does not need to know which source it got. `Sequence("scan.seq",
gradient_mode="pulseq")` is the mode that matches Pulseq's own reference `.h5`.

**[USAGE.md](USAGE.md) is the map of the API** — which call gives which quantity at which sampling
site, and what `gradient_mode` changes.

Times are int64 picoseconds internally, so a delay sweep is bit-exact wherever in the sequence it
lands. Values stay float64.

## Install

```bash
pixi run build          # editable install; regenerates seq2waveform/_core.pyi
```

Needs a system Visual Studio (MSVC) on Windows; `cmake` and `nanobind` come from the pixi
environment. The one third-party dependency, [fast_float](extern/fast_float/README.md), is
vendored, so wheels, the sdist and the conda recipe build with no network access.

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
pixi run test-load        # load(): dispatch and DSV discovery
pixi run test             # all of the above
pixi run bench            # timings on the real datasets
```

`test-h5` is the one that does not depend on this library's own arithmetic — the `.h5` is written
by Pulseq itself — so run it after changing a waveform. `test-invariants` pins this library's
routes against each other (sorted vs unsorted sampling, flat vs per-event k-space, the `.seq`
route vs the DSV route). `test-golden` is change detection: 97 digested columns per dataset against
`tests/golden/*.json`. Regenerate those with `pixi run golden-update` only when the change in
output is intended, and read the diff.

The test corpus is matched `.seq` / `.h5` / `.dsv` files, ~1.6 GB, so it lives outside the repo.
`tests/conftest.py` finds it via `S2W_TESTDATA`, then `./testdata`, then `../seq2waveform/testdata`.
**A run that reports *skipped* found no corpus — that is a failure, not a pass.**

CI (`.github/workflows/ci.yml`) builds the core and the bindings on gcc, clang and MSVC on every
push; `wheels.yml` builds the wheel matrix on a `v*` tag or on demand. CI has no corpus, so only
the `no_corpus` tests run there and a green CI run does not stand in for `pixi run test`.
