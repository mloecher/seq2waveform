# seq2waveform

Read Pulseq `.seq` files and DSV exports into gradients and k-space
trajectories, using Python with a C++17 core.

## Install

The easiest way to install is from [PyPI](https://pypi.org/project/seq2waveform/):

```bash
pip install seq2waveform
```

## Usage

```python
import seq2waveform as s2w

wf = s2w.load("scan.seq")                      # or a DSV export directory
t, gx, gy, gz = wf.gradients_dense(10e-6)      # seconds, Hz/m
t, kx, ky, kz, offsets = wf.adc_kspace_flat()  # seconds, 1/m
```

`load()` defaults to DSV gradient mode. Pass `mode="pulseq"` to match Pulseq's
`waveforms_and_times()`. See [USAGE.md](https://github.com/mloecher/seq2waveform/blob/main/USAGE.md)
for the full API.

## Build and test

```bash
pixi run build          # editable install
pixi run test           # C++ and Python tests
pixi run build-wheel    # wheel in dist/
pixi run conda-build    # conda package in output/
```

Windows needs Visual Studio with C++ support; pixi supplies the other build tools.

Tests include bundled `.seq`/`.h5` data. Set `S2W_TESTDATA` for the full corpus;
DSV comparisons skip when exports are absent.

## License

[MIT](https://github.com/mloecher/seq2waveform/blob/main/LICENSE), including the
vendored [fast_float](https://github.com/mloecher/seq2waveform/blob/main/extern/fast_float/README.md)
dependency.
