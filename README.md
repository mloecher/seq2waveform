# seq2waveform

Read Pulseq `.seq` files and DSV exports into waveform arrays, and sample k-space at arbitrary
times.

```python
import seq2waveform as s2w

wf = s2w.load("scan.seq")                         # or a DSV export directory
t, gx, gy, gz = wf.gradients_dense(10e-6)         # uniform raster, Hz/m
t, kx, ky, kz, offsets = wf.adc_kspace_flat()     # every ADC sample, 1/m
```

Per readout, with k-space referenced to the excitation:

```python
sm = wf.sampler()
for times, ref in wf.adc_specs():
    t, kx, ky, kz = sm.kspace(times, ref)
```

A DSV export answers the same calls as a `.seq`, so you can compare what the scanner played
against what Pulseq asked for. `load` reads a `.seq` in `gradient_mode="dsv"`, the representation
a DSV export also holds; `Sequence("scan.seq", gradient_mode="pulseq")` is the mode that matches
Pulseq's own `waveforms_and_times()`.

Times are int64 picoseconds internally, so a delay sweep is bit-exact wherever it lands in the
sequence. Values are float64.

[USAGE.md](https://github.com/mloecher/seq2waveform/blob/main/USAGE.md) covers the rest of the API.

## Install

```bash
pixi run build          # editable install; regenerates seq2waveform/_core.pyi
pixi run build-wheel    # dist/*.whl
pixi run conda-build    # wraps that wheel into a local conda channel under output/
```

Needs Visual Studio on Windows; `cmake` and `nanobind` come from pixi. The one dependency,
[fast_float](https://github.com/mloecher/seq2waveform/blob/main/extern/fast_float/README.md), is vendored, so wheels, the sdist and the conda recipe
build offline.

## Test

```bash
pixi run test           # all of it; pyproject.toml has the individual tasks
```

`test-h5` compares against a dump of Pulseq's own `waveforms_and_times()`, so it does not rest on
this library's arithmetic — run it after changing a waveform. `test-golden` digests 97 output
columns per dataset against `tests/golden/*.json`; regenerate those with `pixi run golden-update`
only when the change is intended, and read the diff.

Two datasets live in `tests/data` — a `.seq` and its Pulseq `.h5` — so a fresh clone checks real
numbers instead of skipping. They carry no DSV export: those files are a vendor simulator's output,
and `test_dsv.cpp` covers that reader from a minimal file written to the format instead.

The full corpus is matched `.seq` / `.h5` / `.dsv` files, ~1.6 GB, so it lives outside the repo.
`tests/conftest.py` looks for `S2W_TESTDATA`, then `./testdata`, then `../seq2waveform/testdata`,
and prefers it over the bundled copy for a dataset in both.

CI builds the core and the bindings on gcc, clang and MSVC, and runs everything the bundled data
answers for — including the `.h5` comparison. The DSV half still skips there, so a green run is no
substitute for `pixi run test`.
