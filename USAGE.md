# seq2waveform usage

## Loading files

`load()` returns a `Sequence` for a `.seq` file or a `DsvSet` for a DSV export.

```python
import seq2waveform as s2w

wf = s2w.load("scan.seq")
wf = s2w.load("export_dir/")
wf = s2w.load("export_dir/SimulationProtocol_GRX.dsv")
```

DSV exports require matching `*_GRX.dsv`, `*_GRY.dsv`, `*_GRZ.dsv`, `*_RFD.dsv`
and `*_INF.dsv` files, matched case-insensitively. For directories with several
exports, specify a member file. `.seq` files use Pulseq 1.5 event columns and must
be named directly.

`load(..., gamma=...)` sets the DSV conversion from mT/m to Hz/m. `gamma` is in
Hz/T and defaults to `s2w.SCAN_GAMMA`. It is rejected for `.seq` files, already in Hz/m.

### Gradient modes

| Mode | `.seq` reconstruction |
|---|---|
| `"pulseq"` | Raster-edge reconstruction matching Pulseq's `waveforms_and_times()`. |
| `"centers"` | Interpolation between arbitrary shape center samples; trapezoids unchanged. |
| `"dsv"` | Whole-channel sampling at raster centers, including rounded trapezoid corners. |

`load()` defaults to `mode="dsv"`; `Sequence()` defaults to
`gradient_mode="pulseq"`. DSV exports accept only `mode="dsv"`.

## Gradients

```python
corners, values = wf.gradient("x")
t, gx = wf.gradient_axis_dense("x", 10e-6)
t, gx, gy, gz = wf.gradients_dense(10e-6)
```

Axes are `"x"`, `"y"` and `"z"`. `gradient()` returns piecewise-linear corners;
unused axes return empty arrays. Dense methods sample from zero to at least
`wf.duration`, with zeros for unused axes.

Outputs use seconds, Hz/m and 1/m, as float64 arrays. Internal times are integer picoseconds.

## Sampling

Both source types provide these methods.

| Output | ADC samples | Trigger-relative samples |
|---|---|---|
| Gradients | `adc_gradient_samples()` | `trigger_gradient_samples(delay, dt, n)` |
| k-space | `adc_kspace()` | `trigger_kspace(delay, dt, n)` |
| Flat k-space | `adc_kspace_flat()` | `trigger_kspace_flat(delay, dt, n)` |
| Times and references | `adc_specs()` | `trigger_specs(delay, dt, n)` |

Trigger parameters are delay after the trigger (s), sample spacing `dt` (s), and
sample count `n`. k-space is integrated from the trigger time. ADC k-space is
integrated from the most recent excitation at or before the event, or from the
waveform start if none precedes it. Gamma is already included in the gradients.

Gradient and k-space event methods return a list of `(t, x, y, z)` array tuples.
Flat methods return `(t, kx, ky, kz, offsets)`; event `i` is the slice
`offsets[i]:offsets[i + 1]`.

### Custom times

`sampler()` precomputes gradients and cumulative moments for repeated queries.

```python
sm = wf.sampler()
for times, reference in wf.adc_specs():
    t, gx, gy, gz = sm.gradients(times)
    t, kx, ky, kz = sm.kspace(times, reference)
```

Sampler outputs keep the query shape. Float64 input times are seconds, rounded
to picoseconds; returned times are seconds on that grid. Int64 inputs and their
reference are picoseconds. `s2w.ps_from_secs(times)` converts seconds; adding
`1000` shifts the result by 1 ns.

`sm.kspace_flat(times, offsets, references)` takes concatenated float64 times and
per-event references in seconds, and returns `(kx, ky, kz)`.

## Metadata

| Method or property | Result |
|---|---|
| `adc_events()` | ADC time arrays, one per event. |
| `adc_events_meta()` | `(times_s, freq_hz, phase_rad)` per event. |
| `adc_excitations()` | Matched `(center_s, freq_hz, phase_rad)`; NaNs if no excitation precedes the event. |
| `triggers()` | `(channel, time_s, duration_s)` per trigger. |
| `duration` | Duration in seconds. |

`Sequence` also has `num_blocks`, `gradient_mode`, `adc_times()` (concatenated)
and `rf_events()` (`center_s, freq_hz, phase_rad, use`). Undefined RF uses are
not treated as excitations.

DSV times include the scanner lead-in. `DsvSet` merges contiguous ADC segments
with matching dwell times. The INF lacks RF use, ADC frequency/phase and trigger
channel fields: all RF events are treated as excitations, and those offsets and
channels are zero. RF centers use the window midpoint for symmetric pulses or an
RFD peak estimate for asymmetric pulses.
Excitation/ADC count mismatches produce a warning on stderr.

## Other readers

`read_dsv(path)` returns dense values, raster time and unit. `read_dsv_pwl(path)`
returns corner times, values, raster time and unit. INF metadata readers are
`read_inf_adc`, `read_inf_adc_events`, `read_inf_rf`, `read_inf_sync` and `inf_lead_in`.
`parse_file(path)` returns the `.seq` block count.

`PiecewiseLinear` provides interpolation and integration methods. Method
docstrings describe the arguments and units.
