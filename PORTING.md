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
- `src/include/seq2waveform/scan.hpp`, `src/scan.cpp` — `trim`, `fields`, `split_first_space`,
  `parse_f64`, `parse_u32`, `parse_usize`, `saturating_i64`. Reproduces Rust's `str::parse`
  semantics; see **Scanning numbers**.
- `extern/fast_float/` — vendored fast_float 8.1.0 (MIT), supplying `from_chars` for doubles.
  Build with `-DS2W_USE_SYSTEM_FAST_FLOAT=ON` to unbundle. Its README explains why it is a copy
  rather than a `FetchContent`.
- `src/include/seq2waveform/error.hpp` — `ParseError`, carrying the 1-based source line and
  formatting `what()` exactly as Rust's `Display` does.
- `src/include/seq2waveform/model.hpp`, `src/model.cpp` — the structs, `RfUse`, `GradMode`,
  `Shape::decompressed`, `Sequence::definition_f64`.
- `src/include/seq2waveform/parser.hpp`, `src/parser.cpp` — the `[SECTION]` reader.
- `src/include/seq2waveform/dsv.hpp`, `src/dsv.cpp` — the Siemens `.dsv` channel reader: the
  RLE-delta stream decoded straight to piecewise-linear corners, never materialising the dense
  array. `rle_to_piecewise` is also declared in `detail/dsv_detail.hpp` for the tests. `DsvSet`
  (the whole measurement set as a `WaveformSource`) needs the INF reader and arrives at step 7.
- `src/include/seq2waveform/sample.hpp`, `src/sample.cpp` — the `Sampler` (gradients, k-space,
  the flat batched path), the spec builders and the dense uniform-raster sampling. Rust's
  `WaveformSource` trait with default methods becomes an abstract base class whose non-virtual
  helpers are written in terms of five pure virtuals, and `SequenceSource` is the adapter that
  makes a parsed `Sequence` one — keeping `Sequence` a plain aggregate, since it is parsed data
  rather than a polymorphic object. `DsvSet` becomes the second subclass at step 7.
- `src/include/seq2waveform/waveform.hpp`, `src/waveform.cpp` — the block walk and expansion:
  `waveforms`, `gradient`, `adc_times`, `adc_events`, `rf_events_meta`, `trigger_events`, plus
  `stitch`, `restore_additional_shape_samples` and `resample_at_centers`. Rust puts these on
  `Sequence` via an `impl` block in the module; C++ cannot, so they are free functions over
  `const Sequence&` and `model.hpp` stays free of waveform concerns. The last two are also
  declared in `detail/waveform_detail.hpp` so the tests can reach them directly.
- `bindings/seq2waveform_wrapper.cpp` — nanobind module `_core`, with the zero-copy `to_numpy`
  capsule helper that every array return must go through.
- `tests/test_pwl.cpp` — the Rust unit tests from `time.rs` and `pwl.rs`, ported.
- `tests/test_parser.cpp` — the Rust unit tests from `model.rs` and `parser/mod.rs`, ported, plus
  scanning tests that have no Rust counterpart (Rust gets that behaviour free from `str::parse`).
- `tests/test_parity.py` — differential tests against the Rust extension on the real datasets.
  221 of them as of step 5. Step 2 compared 25,300 blocks, 2,051 shapes and 1.36 M decompressed
  shape samples; step 3 added 24.8 M gradient corners across three modes and three axes, 27.3 M
  ADC sample times, 4,601 RF events and 4,261 triggers; step 4 added 27.9 M flat k-space points.
  All exactly equal. A one-ULP perturbation of a single corner value does fail the suite, and
  shifting a k-space reference by 10 µs does change the result -- worth re-checking whenever a new
  comparison is added, because an assertion that cannot fail is worse than no assertion.
- `tests/test_h5.py` — **the only check that does not depend on the Rust implementation being
  right.** Everything above compares the two ports, so an error they *share* is invisible to all
  of it; the `.h5` is written by Pulseq itself. 15 tests at a 1e-4 relative bar, and the worst
  actual margin is 4.3e-6 — a factor of 23 of headroom, so it is agreement rather than a near
  miss. It needs no oracle, so `pixi run test-h5` works on a bare checkout.
- `tests/test_waveform.cpp` — the Rust unit tests from `waveform.rs`, ported. Two matter most:
  `resample_jump_is_exact_vs_bruteforce` re-derives the DSV resampling the slow way (`value_at` at
  every center, then the same collinear drop) and demands the fast path's segment-skipping jump
  agree exactly; and `dsv_mode_reads_zero_outside_the_channel` is a regression test for the
  lead-in/lead-out centers.
- `tests/bench_scan.cpp` — guards the vendored fast_float: asserts it returns identical *bit
  patterns* to this toolchain's `std::from_chars` for every `[SHAPES]` token (1,289,657 of
  1,289,657 across the datasets), and times both. Run it after bumping the vendored version:
  `cmake --build build-tests --config Release --target bench_scan`, then pass it a `.seq` path.

Not started: the DSV and INF readers, and `DsvSet` as a second `WaveformSource`. See **Port order**.

Two branches that no dataset reaches, so they are covered by synthetic tests only — check them by
hand if you touch the matching code. An **ADC with no preceding excitation** (every ADC in all five
corpora has one) takes a far-past sentinel reference and reports `(nan, nan, nan)`:
`adc_with_no_preceding_excitation` in `tests/test_sample.cpp` and
`test_unreferenced_adc_matches_rust` in the parity suite build a synthetic `.seq` for it. The
**excitation/ADC count mismatch warning** is the opposite case — it fires on four of the five
datasets (cones 4236 vs 4216, epi 20 vs 1080), so the stderr noise during a parity run is expected,
not a defect.

> **Step 3 was ported from an uncommitted change, confirmed as intended.**
> `crates/seq2waveform/src/waveform.rs` had an uncommitted fix to `resample_at_centers` (the
> lead-in/lead-out centers, so a DSV channel reads zero outside its range) when this port was
> written, and `build-oracle` builds the oracle from the working tree — so that is the behaviour
> the C++ side matches and the parity suite pins. The author has confirmed it is the wanted
> behaviour. If it is ever reverted or reworked in the Rust repo, DSV-mode parity will break here
> and the failure will look like a C++ bug.

### Parsing performance, measured

Worth knowing before optimising anything else, and worth re-checking when the reader changes.
`parse_text` on `cones` (1.25 M tokens, 21,140 blocks), best of 9, release:

| | cones parse | vs Rust (end to end) |
|---|---|---|
| `std::getline` + `std::from_chars` | 156.8 ms | 0.45x — **2.2x slower** |
| fast_float only | ~137 ms | ~0.55x |
| fast_float + in-place line split | **62.8 ms** | **1.08x — at parity** |

Two lessons. **Line reading dominated, not number parsing**: `std::getline` allocates and copies a
`std::string` per line, which on 1.25 M lines cost more than all the float scanning put together.
`parse_text` now walks the buffer with `string_view` and `find('\n')`, and `parse(std::istream&)`
slurps in 64 KiB blocks and delegates to it. Float scanning went from 30% of parse time to 45%
*because the rest got so much faster*. **And always check the oracle's build profile**: an early
measurement put the port 2.5x *ahead* of Rust, which was nonsense — `maturin develop` had built a
debug oracle. Release Rust is a genuinely fast baseline.

The remaining ~35 ms on `cones` is `trim`/`starts_with`/dispatch per line plus the `std::map`
inserts. Not obviously worth chasing; the h5-comparison work in step 3 will dwarf it.

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

## The `.dsv` channel kinds

Each dataset's `dsv/` folder holds 14 channel kinds plus the INF, and they are not interchangeable
— worth knowing before writing a test that sweeps them, or before assuming `.dsv` means
"gradient".

| kind | raster | unit | notes |
|---|---|---|---|
| `GRX` `GRY` `GRZ` | 10 µs | `mT/m` | the gradients |
| `M0X` `M0Y` `M0Z` | 10 µs | `mT/m * ms` | zeroth moments |
| `SLL` | 10 µs | `T/s` | slew; the largest channel (89 MB on `cones`) |
| `RFD` `RF2` `RF3` | 5 µs | `Volt` | RF envelopes; `RFD` carries the excitation peaks |
| `ADC` | 1 µs | `-` | |
| `NC1` | 1 µs | `Degree` | |
| `CON` `ERR` | — | — | **no `VERTFACTOR`: both readers reject these** |
| `*_INF` | — | — | **not a waveform channel at all** — `[SYNCTYPE]`/`[INFO]` and an ASCII event table. The reader runs to EOF and fails on the missing `VERTFACTOR`. This is step 6's format. |

Two practical consequences for the parity suite. Agreeing on what to *refuse* matters as much as
agreeing on what to decode, so the `CON`/`ERR`/INF rejections are compared by message, not skipped.
And the sizes are brutal: 70 non-INF channels over ~260 MB, and one `cones` channel decodes to
**63.5 million** samples, so a single dense reconstruction is a 508 MB array. Sweeping all of them
densely on both implementations exhausts memory, not patience — the suite therefore checks corners
(which stream) on every channel and the dense reconstruction on one channel per kind, which still
covers every raster and unit above.

## Which dataset covers which gradient branch

`build_grad_piece` has four branches and `restore_additional_shape_samples` has two, and no single
dataset exercises them all. Worth knowing before you change one, because the parity suite can stay
green while a branch goes untested. `time_shape = 0` means a centers-sampled shape.

| dataset | traps | arbs | centers-sampled | branch exercised |
|---|---|---|---|---|
| cones | 1 | 3531 | 3531 | restore's **spiral fallback** |
| girf | 22 | 20 | 20 | restore's **main reconstruction** |
| sync_osc | 2 | 1 | 1 | restore's main reconstruction |
| epi | 6 | 3 | 0 | the non-centers branches (padded / extended trapezoid) |
| epi_fatsat | 7 | 3 | 0 | the non-centers branches |

`cones` is a spiral, so every one of its shapes fails the `edge_rest[n] ≈ last` check and takes the
fallback. That fallback emits exactly what `GradMode::Centers` emits, which is why `pulseq` and
`centers` modes produce *identical* output on `cones`, `epi` and `epi_fatsat` — only `girf` makes
the two modes differ. So the main reconstruction path rests on `girf` plus the
`restore_recovers_a_triangle` unit test. Add a centers-sampled non-spiral dataset if you touch it.

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

### Scanning numbers

C++-specific, and learned the hard way in step 2: Rust gets all of this from `str::parse`, so none
of it is visible in the Rust source. `src/scan.cpp` reproduces it, and `tests/test_parser.cpp`
pins it.

- **Never `strtod`.** It honours the C locale's decimal separator, so under a comma locale it reads
  `1e-05` as `1`. `from_chars` and fast_float are both locale-independent and correctly rounded.
- **Doubles go through vendored fast_float, integers through `std::from_chars`.** Integer
  `from_chars` is everywhere (libc++ since LLVM 7); the floating-point overload only reached libc++
  in 2025, and Xcode's copy lags, which would break the macOS wheels. fast_float needs only C++11,
  is what libstdc++ already uses for `from_chars`, and is the same Eisel-Lemire algorithm Rust's
  `dec2flt` was rewritten to use — so the two readers parse floats identically by construction.
  `extern/fast_float/README.md` has the provenance; `tests/bench_scan.cpp` is the guard.
- **The whole token must be consumed.** Rust's `parse()` rejects trailing characters, so `1.5x` is
  a failure, not `1.5`. Both parsers report where they stopped — compare that against the end.
- **A leading `+` has to be stripped by hand.** Rust accepts `+1.5`; both parsers reject it.
- **`result_out_of_range` is accepted, not rejected.** fast_float has already saturated the value
  to ±inf or ±0, which is what Rust returns for `1e999` / `1e-999`. Treating it as an error would
  reintroduce a divergence for no gain.
- **Float-to-int casts need `saturating_i64`.** Rust's `as i64` saturates and maps NaN to `0`; the
  C++ cast is undefined outside the range. The shape repeat count is the one place it is reachable
  from file data.

### Pulseq `.seq` parsing and expansion

- Format **v1.4/1.5**. v1.3 and unknown versions are untested — add a version guard rather than
  inheriting the silence.
- **Event id `0` means "no event"**, per block, per event type.
- **Units are canonicalised at parse time**: seconds, Hz, Hz/m, radians. Per-table scales are
  applied in the parser so nothing downstream rescales.
- **Copy each unit scale's exact spelling.** `240.0 * 1e-6` and `240.0 / 1e6` are *different*
  doubles (`0.00023999999999999998` vs `0.00024`), and the parity test compares exactly, so
  `* 1e-6` has to stay `* 1e-6`. RF/trap/trigger times and `adc delay` are µs; `adc dwell` is ns.
- **Error lines count blank and comment lines.** Rust enumerates lines *before* skipping them, so
  the reported number is the real file line. Counting only content lines is the easy way to get a
  parser that is right about everything except its error messages.
- **Block durations resolve after the whole file is read.** They are stored in
  `BlockDurationRaster` units and `[DEFINITIONS]` may come *after* `[BLOCKS]`, so rows are held
  raw and scaled at the end. Default raster `10e-6`.
- **`[SHAPES]` is a state machine across lines.** `shape_id` closes the previous shape and opens a
  new one; EOF closes the last. Three quirks to preserve: the `shape_id`/`num_samples` prefixes are
  matched *without* requiring a following space (`shape_id1` parses); a `num_samples` line outside
  any shape is silently dropped; and a repeated `[SHAPES]` header does *not* close the shape in
  progress, while any other section header does.
- **Shape decompression:** `data.size() == num_samples` means stored uncompressed — return it with
  *no* cumulative sum. Otherwise the data are a run-length-encoded derivative (`v, v, k` = `v`
  repeated `k+2` times, detected only when `i+2 < n`), expanded and then cumulatively summed in
  order. The result's length is whatever the runs expand to; it is *not* forced to `num_samples`.
- **A missing `use` column is normal, not an error** — v1.4 files have no such field, and it falls
  back to `undefined`.
- **Unknown sections and unknown `extension` names are skipped, not rejected**, and their rows are
  dropped silently. See **Known gaps** 1.
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
2. ~~**Parse errors report line 0.**~~ Stale — the Rust parser does track lines (`Error::at_line`
   stamps the real number in the parse loop, and its own unit test asserts line 4). The C++ port
   reproduces it, and `test_parse_error_matches_rust` compares the two messages verbatim.
3. **No units layer.** See Gamma above.
4. **Complex RF (B1 re/im) is built but never exposed.** The expander has it; the bindings do not.

C++-specific, added by the port:

5. ~~**Floating-point `std::from_chars` is required.**~~ Closed by vendoring fast_float, which
   needs only C++11. The gap was real: libc++ shipped the floating-point overload in 2025 and
   Xcode's copy trails upstream, so the macOS wheels would have failed on their first CI run.
6. ~~**An out-of-range numeric token fails to parse.**~~ Closed too: fast_float saturates the value
   and only flags `result_out_of_range`, so accepting that code matches Rust exactly.
7. **`parse_model` / `parse_shapes` exist on both sides purely for parity testing** — the Rust
   bindings had no model-level oracle beyond a block count. Neither is re-exported from its
   package. Delete both at cut-over.
8. **`maturin develop` builds the oracle in debug** (`build-oracle` passes no `--release`). Float
   results are identical either way — Rust applies no fast-math — but the parity suite runs in
   ~4.7 s against a release oracle versus ~17 s against a debug one. Debug buys only integer
   overflow checks. Worth switching if the test loop ever feels slow.

## Port order

Each step: port the module, add its C++ tests, extend `tests/test_parity.py`, and keep
`pixi run test` green.

1. ~~`time.hpp`, `pwl.cpp`~~ — done.
2. ~~`model.hpp` + `parser.cpp`~~ — done, plus `scan.cpp` and `error.hpp`, which the Rust version
   gets from the standard library. Oracle: `parse_file` block count, then every event library and
   every shape compared field by field via `parse_model` / `parse_shapes`, and the parse-error
   message compared verbatim.
3. ~~`waveform.cpp`~~ — done, against the `seq.gradient(ax)` oracle: 24.8 M gradient corners over
   three modes and three axes, 27.3 M ADC sample times, 4,601 RF events and 4,261 triggers, all
   exactly equal. **The h5 comparison at 1e-4 is still outstanding** — it needs the dense sampler
   from step 4, and it is what would catch an error both ports share.
4. ~~`sample.cpp`~~ — done: `Sampler`, the spec builders, k-space, the flat batched path, the
   dense sampling, and the N-D time-array support in the bindings. 27.9 M flat k-space points
   compared exactly. **The reference-h5 comparison is done too** (`pixi run test-h5`), which
   closes the one thing step 3 left outstanding.
5. ~~`dsv.cpp`~~ — done, against both `read_dsv_pwl` corners and `read_dsv` dense values, over
   every channel in the corpus rather than just the gradients. See **The `.dsv` channel kinds**.
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
- **All number and token scanning goes through `seq2waveform/scan.hpp`.** One place that matches
  Rust's parsing rules is the only way the two readers agree; a stray `std::stod` or `atof` would
  reintroduce locale dependence and partial parses. The `.dsv` and `.inf` readers reuse it.
- **Read text with `parse_text`-style in-place splitting, never `std::getline` per line.** See
  **Parsing performance** — it was a 2.5x difference on the largest dataset. The `.dsv` and `.inf`
  readers should take the same shape.
- **Any new third-party dependency gets vendored under `extern/`**, with a README recording its
  version, where the bytes came from and its licence. The wheels, the sdist and the conda recipe
  all have to build with no network.
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

Scope for this session: step 6 of the port order — inf.cpp, the Siemens INF reader
(../seq2waveform/crates/seq2waveform/src/dsv/inf.rs).

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
- Verify before reporting: `pixi run test-cpp`, `pixi run test-parity` and `pixi run test-h5`
  here, and `pixi run test` in ../seq2waveform (the Rust suite vs the h5 and the DSV must stay
  green). Paste the real output. If something fails, say so with the numbers rather than narrowing
  the test.
- The parity suite only proves the two ports agree. `pixi run test-h5` is the only independent
  ground truth in the project, so run it on anything that changes a waveform.
- The generated stub seq2waveform_cpp/_core.pyi is build output — never edit it. Put docstrings in
  the nanobind def(...) calls, numpydoc style, and check they came through in the regenerated stub.
- ../seq2waveform must keep working until the final cut-over, so do not change its behaviour. You
  MAY add read-only introspection to its bindings when parity needs an oracle that does not exist
  yet (step 2 had to, for the event libraries): additive pyfunctions only, not re-exported from the
  package, documented as port scaffolding, added to python/seq2waveform/_seq2waveform.pyi so the
  stub stays an exact match for the module, and listed in PORTING.md's known gaps for deletion at
  cut-over. The Rust suite staying green is the test that you only added.

Flag anything in PORTING.md's "Known gaps" list that this module touches, and tell me what you
decided rather than silently inheriting the behaviour.
```

Swap the "Scope for this session" paragraph for the next step each time; everything else stays.
