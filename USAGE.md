# Using seq2waveform

A map of the API, written because the method names are deliberately parallel and that makes them
easy to confuse. The short version: there is **one primitive and a lot of named shortcuts**.

```python
import seq2waveform as s2w
```

## The mental model

Every call is some combination of three choices.

1. **A source.** A Pulseq `.seq` file (`Sequence`) or a Siemens DSV export (`DsvSet`). They expose
   the same 17 methods, so everything below works the same on either.
2. **A quantity.** Gradients (Hz/m), or k-space (1/m, the time integral of the gradient from some
   reference time).
3. **Where it is sampled.** At the ADC samples, at trigger-relative samples, on a uniform raster,
   or at times you supply.

Pick one from each and the method name follows. Nothing in the API does anything else.

For the first of those three, reach for `load`:

```python
wf = s2w.load("scan.seq")             # Sequence, gradient_mode="dsv"
wf = s2w.load("export_dir/")          # DsvSet, five files found by their suffixes
wf = s2w.load("export_dir/SimulationProtocol_GRX.dsv")   # or name any one of them
```

It defaults to the representation both sources share, which is the subject of the
`gradient_mode` section below and the reason this function exists.

## The one primitive

`source.sampler()` builds a `Sampler`: the three gradient channels plus their cumulative moments,
precomputed once. Everything else that samples is this, with the times filled in for you:

```python
sm = seq.sampler()
sm.gradients(times)              # -> (t, gx, gy, gz)
sm.kspace(times, reference)      # -> (t, kx, ky, kz)
```

These hold bit-for-bit, so you can always drop down a level without changing your answer:

| the shortcut | is exactly |
|---|---|
| `adc_kspace()` | `[sampler().kspace(t, ref) for t, ref in adc_specs()]` |
| `adc_gradient_samples()` | `[sampler().gradients(t) for t in adc_events()]` |
| `adc_events()` | the `times` half of `adc_specs()` |
| `adc_times()` | `np.concatenate(adc_events())` |
| the reference in `adc_specs()` | the excitation center from `adc_excitations()` |

So if a shortcut does not give you quite the sampling you want, take `adc_specs()` (or
`trigger_specs(...)`), adjust the times, and call the sampler yourself.

## Which method do I call?

Rows are the quantity, columns are where it lands.

| | at every ADC sample | at trigger-relative samples | at times you supply |
|---|---|---|---|
| **gradients** | `adc_gradient_samples()` | `trigger_gradient_samples(delay, dt, n)` | `sampler().gradients(t)` |
| **k-space** | `adc_kspace()` | `trigger_kspace(delay, dt, n)` | `sampler().kspace(t, ref)` |
| **k-space, one flat array** | `adc_kspace_flat()` | `trigger_kspace_flat(delay, dt, n)` | `sampler().kspace_flat(t, offsets, refs)` |
| **the sample times alone** | `adc_events()`, `adc_specs()` | `trigger_specs(delay, dt, n)` | — |

**Per-event or flat.** The first two rows return a list, one tuple per readout. The `_flat`
versions return one long array plus an `offsets` array to slice it apart. Use flat when you were
going to concatenate anyway.

**The trigger columns** take `(delay, dt, n)`: the delay after each trigger, the sample spacing,
and the number of samples. That is the Skope-style measurement — a fixed grid hung off each
trigger, rather than the scanner's own ADC grid.

## The gradient on its own

| | |
|---|---|
| `gradient(axis)` | `(t, v)` at the **corners** — the exact piecewise-linear description, no resampling, fewest possible points |
| `gradient_axis_dense(axis, raster)` | `(t, v)` for one axis on a uniform raster |
| `gradients_dense(raster)` | `(t, gx, gy, gz)` for all three on a uniform raster |

`axis` is `"x"`, `"y"` or `"z"`. Prefer `gradient(axis)` when you want the truth with no raster
choice baked in; prefer the dense forms when you need a fixed grid to compare against something
else.

**An unused axis comes back empty, not zero.** On a single-axis GIRF sequence, `gradient("x")` is
a pair of length-0 arrays rather than a flat line — there are no corners to report. The dense
forms do give you zeros, because they are defined on `[0, duration]` regardless. So
`gradient(axis)` is the one to guard with a length check, and the dense forms are the ones that
always line up across axes.

## Return shapes

Every trace is `(t, x, y, z)` — four arrays, time first:

```python
t, gx, gy, gz = seq.gradients_dense(10e-6)

tr = sm.gradients(times)
tr[0]          # times (s)
tr[1:]         # the three axes
tr[1 + axis]   # one axis, axis in 0..2
```

The `_flat` variants return `(t, kx, ky, kz, offsets)`; event `i` is `slice(offsets[i], offsets[i+1])`.
Scalar-ish metadata is a list of tuples: `rf_events()` is `(center_s, freq_hz, phase_rad, use)`,
`triggers()` is `(channel, time_s, duration_s)`, `adc_excitations()` is `(center_s, freq_hz, phase_rad)`.

`t` is the query grid *after* the picosecond quantisation every time in this library goes through,
so it is what was actually sampled rather than what you asked for. For raster-aligned times it
comes back identical to the input.

## `gradient_mode`

Only on `Sequence`, because it is about reconstructing a `.seq`'s shape lists.

| mode | what it does |
|---|---|
| `"pulseq"` (current default) | edge reconstruction; matches Pulseq's own reference `.h5`. Can staircase on fast waveforms. |
| `"centers"` | connects the center samples of *arbitrary* shapes; trapezoids stay sharp. |
| `"dsv"` | resamples the whole channel onto raster centers: rounds trapezoid corners **and** smooths arbitrary shapes. This is the Siemens DSV / commanded waveform. |

Measured across the test corpus, comparing each mode's gradient against `pulseq`:

| dataset | `dsv` vs `pulseq` | `centers` vs `pulseq` |
|---|---|---|
| cones | 3/3 axes differ | 0/3 |
| epi | 3/3 axes differ | 0/3 |
| epi_fatsat | 3/3 axes differ | 0/3 |
| girf | 3/3 axes differ | 3/3 |
| sync_osc | 1/1 axes differ | 1/1 |

`centers` only diverges where a sequence actually has centers-sampled arbitrary shapes. **`dsv`
diverges everywhere**, because rounding trapezoid corners affects every sequence.

### For GIRF work: use `"dsv"`, always

The point of `dsv` mode is that it is *the representation the two sources share*. A Siemens
sequence gives you only a DSV export; a Pulseq sequence gives you a `.seq` (and you could simulate
a DSV, but reading the `.seq` is far easier). Loading the `.seq` in `dsv` mode puts both routes
into the same commanded-waveform representation, so the measurement you build on top does not
depend on which kind of sequence it came from.

Reading a `.seq` in the default `pulseq` mode and comparing it against a real DSV compares two
different representations of the same waveform, and the table above is the size of that difference.

```python
seq = s2w.Sequence("scan.seq", gradient_mode="dsv")   # always, for this workflow
```

The invariant is checked on real data: `tests/test_invariants.py` asserts that the `.seq` route in
`dsv` mode and the `.dsv` route reach the same k-space.

## The two sources

```python
wf = s2w.load(src)                             # the usual way: either source, dsv by default

seq = s2w.Sequence("scan.seq", gradient_mode="dsv")   # or name the class yourself
dsv = s2w.DsvSet(grx, gry, grz, rfd, inf)             # five paths; gamma defaults to SCAN_GAMMA
```

A Siemens export is five files per scan, conventionally `*_GRX.dsv`, `*_GRY.dsv`, `*_GRZ.dsv`,
`*_RFD.dsv` and `*_INF.dsv` in one directory, alongside a dozen other channels (`_ADC`, `_CON`,
`_M0X`, ...) that are ignored. `load` finds the five by suffix, case-insensitively; point it at
the directory, or at any one of the files when a directory holds more than one export. A `.seq`
has to be named directly, since finding one file is not a problem worth solving.

Arguments that do not apply to the source named are rejected rather than ignored: `gamma` with a
`.seq`, or `mode="pulseq"` with a DSV export (which *is* the dsv waveform already). Both fail
before anything is parsed.

**Shared by both (17):** `gradient`, `gradient_axis_dense`, `gradients_dense`, `sampler`,
`duration`, `adc_events`, `adc_events_meta`, `adc_specs`, `adc_excitations`,
`adc_gradient_samples`, `adc_kspace`, `adc_kspace_flat`, `triggers`, `trigger_specs`,
`trigger_gradient_samples`, `trigger_kspace`, `trigger_kspace_flat`.

**`Sequence` only:** `num_blocks`, `gradient_mode`, `adc_times`, `rf_events`.
**`DsvSet` only:** `gamma`.

So anything written against the shared 17 is source-agnostic. The notable gap is `rf_events` — for
a DSV set the RF comes from the INF instead, via `s2w.read_inf_rf(inf_path, rfd_path)`.

### Lower-level readers

For when you want one piece without building a whole source: `read_dsv(path)` (dense channel),
`read_dsv_pwl(path)` (the piecewise-linear form), `read_inf_adc`, `read_inf_adc_events`,
`read_inf_rf`, `read_inf_sync`, `inf_lead_in`, and `parse_file(path)` for a raw `.seq` parse.
`PiecewiseLinear` is the waveform type itself, with `value_at`, `sample_at`, `integral` and
`cumulative_moment`.

## Recipes

Dense gradient on a fixed raster — the whole sequence, one array per axis:

```python
t, gx, gy, gz = seq.gradients_dense(10e-6)
```

Per-readout k-space, then the gradient it implies:

```python
for t in seq.adc_events():
    k = seq.sampler().kspace(t, t[0])        # referenced to the window start
    g = np.diff(np.stack(k[1:]), axis=-1) / (t[1] - t[0])
```

The same gradient sampled directly at the bin midpoints, which is the cleaner comparison:

```python
dt = t[1] - t[0]
g = np.stack(sm.gradients(t[:-1] + dt / 2)[1:])
```

Trigger-based (Skope) sampling, and the k-space from integrating it:

```python
specs = seq.trigger_specs(delay + time_shift, dt, n_samp)
for times, ref in specs:
    g = np.stack(sm.gradients(times)[1:])
    k = np.cumsum(g, axis=-1) * dt
```

Every readout's k-space in one array, no Python loop:

```python
t, kx, ky, kz, offsets = seq.adc_kspace_flat()
first = slice(offsets[0], offsets[1])
```

Exact times, for a delay sweep that must not drift: pass int64 picoseconds instead of float
seconds and they are taken verbatim.

```python
ps = s2w.ps_from_secs(times)
k = sm.kspace(ps + 1000, ps[0])      # shifted by exactly 1 ns
```
