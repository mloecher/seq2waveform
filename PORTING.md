# Porting seq2waveform from Rust to C++

Reference for the port, and the kickoff prompt to start it.

This is its own project. The Rust implementation lives in a separate checkout — `$S2W_RUST_REPO`,
defaulting to the sibling `../seq2waveform` — and stays working and installed throughout: it is the
oracle every ported module is checked against, and it owns the test corpus. Paths below of the form
`crates/...`, `python/...`, `scripts/...` or `testdata/...` are relative to *that* repo; everything
else is relative to this one.

## Why

- **Stub generation is a build step.** `nanobind_add_stub` regenerates `seq2waveform_cpp/_core.pyi`
  from the built module on every build, docstrings and `NDArray` annotations included. The Rust
  side needs PyO3 ≥ 0.29 plus an inline-module restructure to get the same thing, and even then the
  numpy annotations depend on rust-numpy implementing the introspection traits.
- **One toolchain.** gropt is already scikit-build-core + nanobind + pixi + cibuildwheel. Same
  build system, same stub story, and `seq2waveform_core` is a CMake target gropt could link.
- **The time-safety guarantee survives.** The `Ps` newtype is the reason delay sweeps are bit-exact.
  C++ keeps it as a real type with explicit conversions; a numpy rewrite would reduce it to a
  convention. This is why the port target is C++ and not Python.
- **No Python version ceiling.** nanobind tracks new CPython releases; `cibuildwheel` here already
  lists `cp314`, while the Rust package is pinned `<3.14` by PyO3 0.23.

## Current state of the scaffold

Built and verified:

- `src/include/seq2waveform/time.hpp` — `Ps`, integer picoseconds, explicit conversions only.
- `src/include/seq2waveform/pwl.hpp`, `src/pwl.cpp` — `PiecewiseLinear`: `value_at`,
  `sample_at`, `sample_at_sorted`, `integral`, `cumulative_moment`, `moment_to`,
  `moments_at_sorted`. Ported line-for-line from `crates/seq2waveform/src/pwl.rs`.
- `bindings/seq2waveform_wrapper.cpp` — nanobind module `_core`, with the zero-copy `to_numpy`
  capsule helper that every array return must go through.
- `tests/test_pwl.cpp` — the Rust unit tests from `time.rs` and `pwl.rs`, ported.
- `tests/test_parity.py` — differential tests against the Rust extension on the real datasets.

Not started: parser, model, waveform expansion, sampler, DSV, INF. See **Port order**.

## Layout

```
CMakeLists.txt                 nanobind_add_module + nanobind_add_stub + tests option
pyproject.toml                 scikit-build-core backend, cibuildwheel and pixi config
src/
  seq2waveform_core.cmake      the shared static-library target — add each new .cpp here
  include/seq2waveform/*.hpp   public headers
  *.cpp                        core implementation
bindings/
  seq2waveform_wrapper.cpp     the only file that may mention nanobind
seq2waveform_cpp/
  __init__.py                  re-exports from ._core
  _core.pyi                    GENERATED on every build — never edit by hand
  py.typed
tests/
  test_pwl.cpp                 C++ core tests (ctest)
  test_parity.py               C++ vs Rust on the Rust repo's datasets
```

Environment and tasks live in `pyproject.toml` under `[tool.pixi.*]` rather than a separate
`pixi.toml`, so the build requirements sit next to the environment that satisfies them — the same
layout as gropt.

Keep the core free of nanobind: only `bindings/` includes it, so the core stays linkable from
plain C++ (tests, gropt, a future CLI).

## Build and test

```bash
pixi run build          # editable install + regenerate _core.pyi
pixi run test-cpp       # C++ core tests via ctest
pixi run build-oracle   # install the Rust implementation into this environment
pixi run test-parity    # C++ vs Rust on every dataset (builds both first)
pixi run test           # test-cpp + test-parity
```

And in the Rust repo, which must keep passing untouched:

```bash
pixi run test           # the Rust suite vs the h5 and the DSV
```

`cmake` and `nanobind` come from the pixi environment; MSVC comes from the system Visual Studio
install, as in gropt. `ninja` is deliberately **not** a dependency — without it CMake uses the
Visual Studio generator, which finds MSVC without `vcvars` on `PATH`.

`maturin` and `h5py` are in this environment only for the port: the first builds the oracle from
the Rust checkout, the second is for the dataset suite once it moves here. Both go at cut-over.

## The test corpus: matched `.seq` / `.h5` / `.dsv`

The corpus lives in the Rust checkout at `$S2W_RUST_REPO/testdata` (override with `S2W_TESTDATA`).
It is gitignored there, so it exists on this machine only — it is not in either repository, and a
fresh clone has no data.

Five datasets, each a directory named after itself:

```
testdata/<name>/
  <name>.seq                        Pulseq source — Pulseq time frame
  <name>.h5                         reference waveforms — Pulseq frame, Hz/m
  dsv/
    SimulationProtocol_GRX.dsv      gradient x — Siemens frame, mT/m
    SimulationProtocol_GRY.dsv
    SimulationProtocol_GRZ.dsv
    SimulationProtocol_RFD.dsv      RF envelope (gives RF centre times)
    SimulationProtocol_INF.dsv      event-block table, ADC headers, SYNC column
    SimulationProtocol_ADC.dsv      } also emitted by the simulator;
    SimulationProtocol_M0X/Y/Z.dsv  } not currently read
    SimulationProtocol_CON/ERR/NC1/RF2/RF3/SLL.dsv
    SimulationProtocol.pro          Siemens protocol
    SequenceSimulation.log
```

Datasets: `cones`, `epi`, `epi_fatsat`, `girf`, `sync_osc`.

A directory counts as a dataset when both `<name>.seq` and `<name>.h5` exist. Because the corpus
can be absent, the suites skip rather than fail on an empty dataset list — keep that, but
understand the trap: **a parity run reporting `skipped` is a failed parity run.** It means the
oracle or the corpus was not found, and the port lost its entire safety net silently. Check the
summary line, not just the exit code.

**The two frames are the crux of this project.** The `.seq` and `.h5` share the Pulseq frame. The
DSV lives in the Siemens frame, later by the INF-encoded lead-in. Everything that compares the two
must shift by that lead-in, and must read it from the INF rather than fitting it.

## Validation thresholds

From the Rust repo's `scripts/test_datasets.py`, which must keep passing against the Rust build and
be mirrored here as modules land:

| Check | Comparison | Bar |
|---|---|---|
| `test_gradients` | our corners interpolated onto h5 times | rel max diff < `1e-4`, normalised by `max(abs(h5), 1.0)` |
| `test_excitation_times` | RF centres vs h5 `/excitation` | count equal, abs < `1e-9` |
| `test_adc_times` | all ADC times vs h5 `/adc` | count equal, abs < `1e-9` |
| `test_dsv_gradients_match_h5` | DSV vs h5 on the DSV raster | rel max diff < `1e-3`, normalised by `max(abs(h5), 1e-3)` |
| lead-in sanity | `inf_lead_in` | `0.9 ms < lead < 1.2 ms` |

For the port itself the bar is **higher**: exact equality against Rust. Both implementations do the
same arithmetic in the same order on int64 times, so any difference is a porting bug rather than
float noise. `tests/test_parity.py` asserts `np.array_equal`. If a check genuinely cannot be exact,
document why in that test rather than loosening it quietly.

## Edge cases and conventions that must survive the port

Each of these was found the hard way. Getting one wrong produces output that is plausible and wrong.

### Time

- **Integer picoseconds, not float seconds.** At 24 s an `f64` resolves only a few femtoseconds, so
  interpolating `(t-t0)/(t1-t0)` with large operands loses precision that depends on *where* in the
  sequence you are — the same block played early vs late would sample differently. Only time is
  integer; values (Hz/m, 1/m, Hz, rad) stay `double`.
- **`from_secs` rounds half away from zero** (`std::llround`), matching Rust's `(s*1e12).round() as
  i64`. Parity depends on it.
- **`secs()` multiplies by `1e-12`**, it does not divide by `1e12`. Same result to ~1 ulp, much
  cheaper, and the choice is baked into the parity expectations.
- **`frac_between` subtracts as integers, then divides once.** Converting to seconds first defeats
  the entire design.
- *Known difference:* Rust's `as i64` saturates on overflow; `std::llround` is undefined there.
  Unreachable below ~106 days of sequence, but it is a real divergence — note it if you ever accept
  untrusted times.

### Piecewise-linear waveforms

- **Values clamp outside the corner range; moments do not.** `value_at` returns the first/last
  corner value; `moment_to` returns `0` before the range and saturates at `cum[n-1]` after.
- **Zero-width segments are legal.** Duplicate corner times occur at block boundaries; return `v0`
  rather than dividing by zero.
- **`partition_point(|x| x <= t)` is `std::upper_bound`.** Off by one here shifts every sample that
  lands exactly on a corner — which, on a raster-aligned sequence, is most of them.
- **The sorted sweep's cursor starts at `max(upper_bound(times[0]), 1)`** and is never rewound by a
  clamp branch. Both ports rely on `times` being non-decreasing for that path.

### Pulseq `.seq` parsing and expansion

- Format **v1.4/1.5**. v1.3 and unknown versions are untested — add a version guard rather than
  inheriting the silence.
- **Event id `0` means "no event"**, per block, per event type.
- **Units are canonicalised at parse time**: seconds, Hz, Hz/m, radians. Per-table scales are
  applied in the parser so nothing downstream rescales.
- **`restoreAdditionalShapeSamples`** reconstructs centres-raster arbitrary shapes. Invariant worth
  its own test: the per-interval mean of the reconstructed segment equals the stored centre sample
  (moment conservation). This is listed as a missing test in `TODO.md` — add it in C++.
- **Cross-block stitch**: insert a forced ramp to and from zero across gaps, and drop duplicate
  samples at shared boundaries. A non-zero value at a gap boundary is worth a warning.
- **Deliberate divergence from Pulseq:** RF keeps its real `use`; undefined pulses are *not* folded
  into excitation. Pulseq's own h5 routes both `excitation` and `undefined` into `/excitation`, so
  the h5 comparison must union the two — see `test_excitation_times`.
- **Three gradient modes**, all three needed:
  - `pulseq` (default) — edge reconstruction; matches the h5; can staircase on fast waveforms.
  - `centers` — connect centre samples for *arbitrary* shapes (smooth); trapezoids stay sharp.
  - `dsv` — resample the whole channel onto raster centres; rounds trapezoid corners *and* smooths
    arbitrary shapes, i.e. the Siemens commanded waveform, one consistent representation.

### Siemens `.dsv`

- `[DEFINITIONS]` gives `HORIDELTA` (sample spacing, **microseconds**), `VERTFACTOR` (divide the
  cumulative sum by it for physical units) and `VERTUNITNAME`.
- **`[VALUES]` is the derivative, run-length encoded**: `d, d, k` means `d` repeated `k+2` times.
  Decode straight to piecewise-linear corners — a run of equal deltas is one constant-slope
  segment, so keep only slope changes plus the endpoints. Never materialise the dense array.
- **Sample times are interval-centred at `(k+0.5)*raster`**, not at `k*raster`. Sampling the decoded
  PWL at those instants must reproduce `cumsum(deltas)/VERTFACTOR` exactly; that equivalence is the
  decoder's own test.

### Siemens INF

- Two block shapes: `adc MeasHeader <start_us> <dur_us>` followed by `key : value` lines, and
  `* EventBlock <start_us> <dur_us>` followed by a `|`-delimited table.
- **Skip rows whose first cell is not a number** — headers, separators, the rotation matrix,
  `RTAdjData`.
- **Lead-in:** event block index 0 is the Siemens `GLOBzSet` setup; index **1** is the first Pulseq
  block, and *its* start time is the lead-in (~1020 µs). Add it to a Pulseq-frame time to reach the
  Siemens frame. Never fit it.
- **Undo `MaxAdcSegmentLength` chunking:** Siemens splits a long acquisition into chunks (e.g. 5000
  samples). Merge consecutive segments that are contiguous in time *and* share a dwell — tolerance
  `1e-6 * max(dwell, 1e-12)` — to recover the logical ADC events that match the `.seq`. The merge is
  exact. A gap, or a dwell change, starts a new event.
- SYNC triggers report **channel 0** (the SYNC column has no channel).
- INF ADC events carry **freq = phase = 0**; the INF does not record them.

### Sampling and k-space

- **k-space is `∫G·dt` in 1/m directly.** Gamma is already baked into the Hz/m gradients — do not
  multiply by it again. This is the easiest way to be wrong by a factor of 42.6 million.
- **References differ by source:** ADC integrates from the preceding excitation; trigger/Skope
  integrates from the trigger itself.
- `adc_specs` warns when the RF and ADC event counts disagree; `adc_excitations` returns
  `(nan, nan, nan)` when an ADC has no preceding excitation. Keep both — they are how a mismatched
  sequence announces itself.
- **Flat layout:** `offsets[0] == 0`, `offsets[n_events] == len(times)`, event `e` is
  `[offsets[e], offsets[e+1])`.

### Gamma

`SCAN_GAMMA = 42_575_575` Hz/T — the value the **Siemens interpreter** uses, not the textbook
42.577e6. `mT/m = (Hz/m) / (gamma * 1e-3)`. The Rust version hardcodes it; the C++ port should take
the chance to add the units/gamma layer that `TODO.md` §3 asks for.

### Reading the reference `.h5`

- **Orientation varies.** Datasets arrive as `(N,)`, `(N, ncol)` or `(ncol, N)`; normalise before
  comparing (`_orient` in `scripts/test_datasets.py`).
- **The h5 has corners the DSV never had.** `restoreAdditionalShapeSamples` adds 5 µs corners that
  are Pulseq-side reconstruction detail; Siemens only ever played the 10 µs raster. Compare
  DSV-vs-h5 on the DSV grid, or the comparison is unfair and will fail at ~1e-2.

## Known gaps inherited from the Rust version

Decide on each rather than porting the silence (from the Rust repo's `TODO.md`):

1. **Unhandled `.seq` extensions.** Only `extension TRIGGERS` is handled. `ROTATIONS`, `LABELS`,
   soft-delays and anything else are recognised as section headers and their rows dropped **with no
   warning**. The five datasets are unaffected — `cones` bakes its rotations into the per-interleaf
   waveforms, and matches both the h5 and the physical DSV — but a `.seq` using runtime `ROTATIONS`
   would produce wrong physical gradients silently. Minimum: warn. Better: apply the rotation.
2. **Parse errors report line 0.** No line tracking. Worth fixing while writing a new parser.
3. **No units layer.** See Gamma above.
4. **Complex RF (B1 re/im) is built but never exposed.** The expander has it; the bindings do not.

## Port order

Each step: port the module, add its C++ tests, extend `tests/test_parity.py`, and keep
`pixi run test` green.

1. ~~`time.hpp`, `pwl.cpp`~~ — done.
2. **`model.hpp` + `parser.cpp`** — plain structs plus the `[SECTION]` reader. Oracle:
   `parse_file(path)` block count, then every event library compared field by field.
3. **`waveform.cpp`** — the hard one. Block walk, trapezoid corners,
   `restoreAdditionalShapeSamples`, the stitch, all three gradient modes. Oracle: `seq.gradient(ax)`
   corner times and values, exactly; then the h5 at 1e-4.
4. **`sample.cpp`** — `Sampler`, the spec builders, k-space, the flat batched path. Oracle:
   `sampler().gradients`, `.kspace`, `.kspace_flat`, `adc_kspace_flat`.
   Rust's `WaveformSource` trait with default methods maps to an abstract base class whose
   non-virtual methods are written in terms of pure-virtual primitives (`gradient`, `adc_events`,
   `excitations`, `trigger_events`, `duration`).
5. **`dsv.cpp`** — RLE-delta decode to PWL corners. Oracle: `read_dsv` dense values and
   `read_dsv_pwl` corners.
6. **`inf.cpp`** — INF parse, lead-in, ADC merge, triggers, RF centres. Oracle: `inf_lead_in`,
   `read_inf_adc_events` grouping, `read_inf_sync`, `read_inf_rf`.
7. **`DsvSet`** behind the same interface as `Sequence`, so it stays a drop-in source.
8. **Cut over.** Rename the Python package from `seq2waveform_cpp` to `seq2waveform`, drop `maturin`
   and the oracle task from `pyproject.toml`, move the dataset suite and the test corpus over from
   the Rust repo, port its `recipe.yaml` to build from the new wheel, add the cibuildwheel GitHub
   Actions workflow (copy gropt's `build-and-publish.yml`), and archive the Rust repo.

## Conventions for the C++ side

- **C++17**, matching gropt's `cxx_std_17`, so the cores can share a build.
- **Core stays nanobind-free.** Only `bindings/` includes it.
- **Every array return goes through `to_numpy`** — move the vector to the heap, hand ownership to a
  capsule. Returning a view into a local is the one memory bug this layer can have.
- **Times across the boundary:** float64 seconds, or int64 picoseconds taken verbatim. Bind both as
  overloads so the generated stub shows `@overload`.
- **Docstrings go in the `def(...)` call**, in numpydoc form. They are the only source the generated
  stub has — nothing else reaches the editor.
- **Add each new `.cpp` to `src/seq2waveform_core.cmake`.** Tests and bindings then pick it up with
  no further wiring.
- **Never hand-edit `seq2waveform_cpp/_core.pyi`.** It is regenerated on every build.

---

## Kickoff prompt

Start the session in **this** project folder, and add the Rust repo (`../seq2waveform`) as a second
working directory — the port needs to read the Rust source, and the datasets live there.

```text
Continue the C++ port of seq2waveform. This project is the C++ port; ../seq2waveform is the Rust
implementation it is being ported from, which is also the test oracle and holds the datasets.

Read PORTING.md here first — it has the layout, the test corpus, the validation bars, every edge
case the Rust version earned the hard way, the known gaps and the port order. Then read the Rust
module you are porting under ../seq2waveform/crates/seq2waveform/src/.

Scope for this session: step 2 of the port order — model.hpp and parser.cpp, the Pulseq .seq
reader (../seq2waveform/crates/seq2waveform/src/model.rs and .../parser/mod.rs).

Rules:
- The Rust crate is the oracle, not a sketch. Port semantics line-for-line; where you deviate, say
  so and justify it. Canonical units are applied in the parser (seconds, Hz, Hz/m, radians), and
  event id 0 means "no event".
- C++17. The core stays nanobind-free: only bindings/ may include nanobind. Add every new .cpp to
  src/seq2waveform_core.cmake.
- Tests are not optional and not an afterthought:
  * port the Rust unit tests for the module into tests/ and register them with ctest;
  * extend tests/test_parity.py with differential checks against the Rust extension over every
    dataset, asserting exact equality (np.array_equal), not a tolerance;
  * a parity run that reports "skipped" has found no oracle or no corpus — that is a failure, not
    a pass. Check the summary line.
- Verify before reporting: `pixi run test-cpp` and `pixi run test-parity` here, and `pixi run test`
  in ../seq2waveform (the Rust suite vs the h5 and the DSV must stay green). Paste the real output.
  If something fails, say so with the numbers rather than narrowing the test.
- The generated stub seq2waveform_cpp/_core.pyi is build output — never edit it. Put docstrings in
  the nanobind def(...) calls, numpydoc style, and check they came through in the regenerated stub.
- Do not change anything in ../seq2waveform — it must keep working until the final cut-over.

Flag anything in PORTING.md's "Known gaps" list that this module touches, and tell me what you
decided rather than silently inheriting the behaviour.
```

Swap the "Scope for this session" paragraph for the next step each time; everything else stays.
