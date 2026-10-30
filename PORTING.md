# seq2waveform: design notes, hard-won edge cases, and the port that produced them

What the port left behind, and the reference worth reading before changing anything here.

This library began as a C++ port of a Rust implementation. The Rust crate was the oracle: every
module was checked for bit-exact equality against it on real data rather than against re-derived
ground truth, which is how 284 differential tests came to exist. At cut-over the crate was retired
and those tests went with it — they compared against something that no longer runs — so what
remains is this document plus the suites described under **Build and test**.

Read it for: the edge cases each implementation earned the hard way (**Edge cases and conventions**
is the long section, and the valuable one), why the validation thresholds are the numbers they are,
the measured performance work, and the known gaps that are still open.

Paths of the form `crates/...`, `python/...` or `scripts/test_datasets.py` refer to the retired Rust
checkout, kept in the text where the history explains a decision. Everything else is relative to
this repo.

## Why C++ at all

- **Stub generation is a build step.** `nanobind_add_stub` regenerates `seq2waveform/_core.pyi`
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
- `src/include/seq2waveform/dsvset.hpp`, `src/dsvset.cpp` — `DsvSet`: three gradient
  channels, the RFD envelope and the INF, behind the same `WaveformSource` interface as
  `Sequence`. Its own header rather than part of `dsv.hpp` because it needs `sample.hpp` and
  `inf.hpp`, which the channel reader does not.
- `src/include/seq2waveform/constants.hpp` — `kScanGamma`, which the Rust crate keeps at its root.
- `src/include/seq2waveform/inf.hpp`, `src/inf.cpp` — the Siemens INF reader: the block/table
  parse, the `MaxAdcSegmentLength` merge, the lead-in, SYNC triggers and RF events with their
  RFD-refined centers. `merge_contiguous` and `split_cell` are also in `detail/inf_detail.hpp` for
  the tests. Reads with the in-place line split, not `std::getline`: these are the largest files in
  the corpus.
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
- `tests/test_invariants.py` — internal invariants on the real datasets: the two sampling paths
  against each other, the flat and per-event k-space against each other, the corner form against
  the dense form, and the `.seq` route against the DSV route. Exact where both sides are our own
  arithmetic; a tolerance only for the last, which crosses two file formats.
  **This is what is left of the differential suite** — the tests in it that never needed the
  oracle. For the record, that suite ran 284 comparisons: 25,300 blocks, 2,051 shapes and 1.36 M
  decompressed shape samples; 24.8 M gradient corners across three modes and three axes; 27.3 M ADC
  sample times, 4,601 RF events, 4,261 triggers; 27.9 M flat k-space points. All exactly equal.
  A one-ULP perturbation of a single corner value did fail it, and shifting a k-space reference by
  10 µs did change the result — worth re-checking whenever a new comparison is added anywhere,
  because an assertion that cannot fail is worse than no assertion.
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
> written, and the oracle was built from that working tree — so that is the behaviour
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
inserts. Not obviously worth chasing -- and it is now a small share of any real pipeline, as the
next section shows.

### Whole-pipeline performance, measured

Done once the port hit full parity, with **both sides in release** and the parity suite as the
check that nothing moved. The ratios below are therefore history: they cannot be reproduced
now that the oracle is gone (`scripts/bench.py` lost its `--compare-rust` at cut-over), and
`adc_kspace_flat()` has since improved again — see the next section. Use
`tests/bench_expand.cpp` for a native breakdown with no Python in the way; it mirrors the Rust
repo's `examples/bench_expand.rs` + `examples/time_kspace.rs`, phase for phase.

`cones`, the worst case at 14.9 MB and 21,140 blocks, C++/Rust ratios from Python:

| call | before | after |
|---|---|---|
| `Sequence(path)` | 0.95x | 0.96x |
| `gradient('x')` | 1.67x | 1.17x |
| `adc_times()` | **419x** | **0.86x** |
| `adc_kspace_flat()` | 1.43x | 1.21x |
| `gradients_dense(1e-4)` | 1.85x | 1.20x |

Five fixes, all of them allocation rather than arithmetic. The native breakdown is what found
them: the sampling *kernel* was already at parity (93 M samples/s against Rust's 89 M), so every
gap had to be in a construction phase.

1. **`adc_times` reserved inside its loop.** `out.reserve(out.size() + adc.num)` per ADC event
   asks for the *exact* new size, which defeats the geometric growth `push_back` gives for free and
   recopies the whole array every event. 4216 events over 21.5 M samples = 47 **seconds** of pure
   memcpy, against Rust's 96 ms. Counting first and reserving once makes it 62 ms. A per-iteration
   `reserve` is worse than no `reserve` at all; `grep -rn "reserve(" src/ | grep "size() +"` keeps
   it from coming back.
2. **`finish_channel` returned a `const` local.** A const local cannot be implicitly moved on
   return, so `return pwl` deep-copied both corner arrays — 43 MB per `gradient()` call. Dropping
   the `const` was the whole fix. `scripts/` has no detector for this; the pattern to look for is a
   `const` non-scalar local that is later `return`ed.
3. **`stitch` grew its output unreserved**, as did **`flatten_specs`**. Both concatenate thousands
   of pieces into one array, and MSVC's `std::vector` grows at 1.5x (Rust's `Vec` doubles), so the
   recopying is worse here than the equivalent Rust. Both now total the pieces first.
4. **`kspace_flat` allocated a temporary per event per axis.** `moments_at_sorted` returned a fresh
   40 KB vector that was then copied into the result slice so a constant could be subtracted:
   12,648 allocations and 64 M redundant element copies on `cones`. The new
   `moments_at_sorted_into` writes straight into the slice, and the vector-returning form delegates
   to it, so there is still one implementation.
5. **`gradients_dense` copied its own time grid** into the result instead of moving a dying local.

What is left is 1.2–1.4x on the calls that build large arrays, and a chunk of that is the
nanobind-vs-pyo3 conversion rather than the core — native `gradient('x')` is 45.2 ms against Rust's
45.8, while through Python it reads 1.17x. Closing the rest would mean an interface change
(`WaveformSource` exposes `gradient(axis)` one axis at a time, so `gradients_dense` walks the
blocks three times where `waveforms()` walks once) for a win worth tens of ms. Not taken.

Two general lessons worth keeping. **Measure native before blaming the bindings** — the opposite
was true here every time. And **a 1.2x gap and a 400x gap look the same from Python**: the 47 s
`adc_times` had been in the tree since step 3 and the parity suite never minded, because it was
correct, only slow.

### The second pass: the whole codebase, once the goldens could see it

Step 8 was driven by a native breakdown of one pipeline, so it found what that pipeline touched.
With the snapshots in place the rest of the library became safe to change, and reading all 4,700
lines for cost rather than for parity turned up four things bigger than anything on the list
below. Two classes, and both are worth recognising by shape:

**Wrong complexity.** `RfEvent::rfd_peak` scanned the entire RFD corner array twice, to find the
maximum inside one pulse's window, once per RF event — 4,236 events over 309,230 corners on
`cones`, 1.3 G iterations, and it sits under every DSV k-space call because `excitations()` goes
through it. The corners are sorted, so the window is two binary searches. `DsvChannel::dense()`
had the same shape one level down: `value_at` per sample, a binary search each, over 63.5 M
strictly increasing sample times where a forward sweep is O(n + m). Neither was visible from the
`.seq` benchmarks, because neither is on the `.seq` path.

**Redundant passes.** `adc_excitations()` needed the first sample time of each ADC event and got
it by building every sample time in the run: 172 MB to read 4,216 numbers. The `.dsv` reader still
used `std::getline` — one allocation per line, on files that are one integer per line and millions
of lines long — although the `.seq` parser and the INF reader had both already been converted to
slurp-and-split for exactly that reason. And in the bindings, `trace_to_py` and `flat_to_py` took
their result by `const&`, so they had nothing movable to hand `to_numpy`, whose entire purpose is
to move a vector to the heap and let numpy view it: they copied each axis into a local first,
which on a whole-run k-space is 517 MB duplicated to defeat the helper written to avoid exactly
that.

`cones` at the Python level, best of n, the two builds run back to back so a slow moment on the
machine cannot flatter either:

| call | before | after | |
|---|---|---|---|
| `adc_excitations()` | 96.1 ms | 0.7 ms | **137x** |
| `DsvSet.adc_excitations()` | 849.1 ms | 7.9 ms | **108x** |
| `DsvSet.adc_kspace_flat()` | 1252.5 ms | 445.5 ms | **2.8x** |
| `read_dsv()` (dense) | 1287.6 ms | 578.2 ms | **2.2x** |
| `read_inf_rf()` | 1664.3 ms | 866.3 ms | **1.9x** |
| `read_dsv_pwl()` | 489.1 ms | 359.8 ms | 1.36x |
| `DsvSet(...)` | 2271.1 ms | 1912.3 ms | 1.19x |
| `read_inf_adc()` | 486.1 ms | 427.1 ms | 1.14x |
| `adc_kspace_flat()` | 612.6 ms | 549.6 ms | 1.11x |
| `gradients_dense(1e-4)` | 219.0 ms | 198.8 ms | 1.10x |

Unchanged, within noise: the `.seq` parse, `gradient()`, `adc_times()` and `gradient_axis_dense()`
— nothing in this pass touches them. Judge the noise floor from the parse, which is the canary:
it reads 80.6 and 76.1 ms in the two runs with no change to the parser at all, so treat anything
under about 1.1x here as unproven.

Two mistakes, both caught only by measuring, both worth repeating because the first kind is easy
to make while tidying:

- Routing the vector-returning `sample_at_sorted` through the new write-into-buffer form made it
  zero-fill an output it immediately overwrites. The sweep rule now lives in one helper that both
  forms call, so one can reserve and the other write in place; `moments_at_sorted`, which already
  had that defect, is fixed the same way. **A refactor toward "one implementation" is a
  performance change, and needs measuring like one.**
- Replacing `find('\n')` with an inline scan, on the theory that a memchr call per four-byte line
  was the `.dsv` reader's cost, did nothing at all (364.7 against 359.6 ms). It is not in the tree.
  The reader's remaining cost is the per-line integer parse, which is the actual work.

### What is left on the table, measured

The three below were quantified at step 8 and are still **not** taken. They were declined then
because each risks *silently wrong output* rather than a crash, with the differential suite just
deleted; `test-golden` has since removed that objection, so what is left is simply that none of
them is worth much.

| opportunity | worth | why still not taken |
|---|---|---|
| `adc_events()` builds one vector per event | ~22 ms (5%) | needs an interface change |
| `kspace_flat` zero-fills 517 MB it then overwrites | 15-25 ms (4%) | needs uninitialised storage |
| nothing is cached between calls | up to ~130 ms per repeat call | changes the API's contract |

**Per-event vectors.** `adc_events()` takes 84.3 ms to produce 21.5 M sample times as 4216
separate vectors; `adc_times()` produces the *same* values into one vector in 62.1 ms. The
difference is 4216 allocations at roughly 5 µs each, and the flat k-space path concatenates them
all into one array anyway, so for that path the per-event form is pure overhead. Building the flat
array directly needs a streaming variant of the `adc_events()` virtual, i.e. `WaveformSource` and
both implementations. Note that `adc_event_starts()` is the cheap half of this done already: where
a caller needs one number per event rather than all the samples, it no longer pays for them.

**The zero-fill.** `kspace_flat` starts with three `std::vector<double>(n, 0.0)` and overwrites
every element, because `flatten_specs` builds offsets covering `[0, n)` contiguously. Allocating
517 MB zeroed costs ~37 ms, but part of that is the page faults from first touching the memory,
which an uninitialised allocation still pays on first *write* — the genuinely avoidable share is
the zeroing alone. Avoiding it means storage `std::vector` cannot give, so `FlatK` would have to
hold something else, and that type is what the bindings hand to numpy.

**Caching.** Nothing is memoised: `sampler()` expands all three gradient channels every call, and
`gradients_dense()` expands them again. A caller doing both pays for the same work twice. This is
the largest single win available and still the one to avoid: `Sequence` has a mutable
`gradient_mode` for a cache to go stale behind, and nothing being memoised is what makes repeated
calls honest and these numbers mean what they say.

Two more, found in this pass and left alone deliberately:

- **The `.dsv` per-line integer parse.** A channel reads at ~64 MB/s against the `.seq` parser's
  ~198 MB/s on much longer rows, so roughly 55 ns goes on each four-byte line. It is not the
  newline scan (measured above). Beating `std::from_chars` here means a hand-rolled digit loop
  that re-implements `parse_i64`'s sign handling, overflow rejection and error message, in the one
  reader where a wrong value is a wrong waveform. Worth about 1 s on a `DsvSet(...)` of `cones`.
- **The event libraries are `std::map`.** A red-black tree lookup per block per axis, chosen so
  iteration order was reproducible while the oracle existed. `unordered_map` would be faster, but
  parse is already at 0.96x of Rust, which uses a `HashMap` — so this is not where the time is.

The right trigger for any of these is a profile of a real workload, not this table.

## Compile and repository options

Few flags, and the reasons they are the ones they are.

**`-ffp-contract=off` on GCC, Clang and AppleClang — a correctness setting, not a tuning knob.**
GCC defaults to `-ffp-contract=fast` and Clang to `=on`, both of which let the compiler fuse
`a + b * c` into one FMA with a single rounding instead of two. That changes the last bit, and
tests/golden/ pins every output bit-for-bit. The exposure is uneven: x86-64's base ISA has no FMA
instruction to fuse into, so the Linux and Intel macOS wheels were never at risk, but **FMA is in
the arm64 baseline**, so the Apple Silicon wheel was. The expressions it would have changed are
precisely the ones that matter — `cum[i-1] + 0.5 * (v0 + v1) * dt` in `pwl.cpp`, the k-space sums
in `waveform.cpp`.

Pinning it off makes the GNU-like compilers reproduce what MSVC already produces; it does not
change what MSVC produces, so the committed snapshots stay valid. MSVC is deliberately left alone:
its x64 default targets SSE2, emits no FMA, and so already behaves this way, and touching its FP
flags would risk moving the very numbers the snapshots pin. The cost is losing FMA in those loops
on arm64 and on any AVX2 build — unmeasured, because there is no arm64 machine here, and the hot
paths are dominated by parsing and memory traffic. If an arm64 benchmark ever argues otherwise,
the alternative is a second snapshot set for arm64, which is a considerably worse trade.

With contraction pinned, the remaining threat to cross-platform bit-exactness is `libm`: `std::sin`
and `std::cos` are not bit-identical between glibc, Apple's libm and the MSVC CRT. There are
exactly two such calls in the library, both in `rf_pulse` building the complex RF waveform, and
**no golden column carries that waveform** — the RF columns are scalars (`center_s`, `freq_hz`,
`phase_rad`, `use`). So the snapshots should hold across compilers, rather than being expected to
drift. Everything else numeric goes through `std::llround`, `floor`, `ceil`, `fabs` and `round`,
all of which are exactly specified.

**`-Wall -Wextra` on the GNU-like compilers, not `-Werror`, and MSVC left at `/W3`.** CI is the
only reader of those warnings, since development happens on MSVC; a warning from the first ever
gcc or clang build of this code should start a conversation, not block the matrix. `/W4` locally
is a one-word change in `src/seq2waveform_core.cmake` if it is ever wanted.

**`CMAKE_BUILD_TYPE` defaults to Release in `CMakeLists.txt`**, but only when the generator is
single-config and nothing was asked for. An empty build type on Make or Ninja means no
optimisation at all, so a plain `cmake -S . -B build && cmake --build build` would quietly produce
an unoptimised library and meaningless `bench_*` numbers. Multi-config generators (Visual Studio,
Xcode) ignore the variable and take `--config` at build time, which is why it must not be forced
there — doing so is what makes CMake warn that a manually-specified variable went unused. The
wheel build states `cmake.build-type = "Release"` in `pyproject.toml`; that is already
scikit-build-core's default, and it is written down because a silent change to it would ship an
unoptimised wheel without anything failing.

**What nanobind already handles, so it is absent here on purpose.** `nanobind_add_module` sets
`CXX_VISIBILITY_PRESET hidden`, and defaults to linking the nanobind runtime **statically** into
`_core` (`NB_STATIC` is implied unless `NB_SHARED` or split mode). The second point settles an
open question from the cibuildwheel v4 bump: with no non-system shared library to bundle,
`delvewheel` on Windows and `auditwheel` on Linux have nothing to do. `STABLE_ABI` is deliberately
unused — it would forgo per-version optimisations and cannot be combined with free-threading
before Python 3.15.

**LTO is available and deliberately not taken yet.** `nanobind_add_module` accepts an `LTO`
keyword, and it would shrink the extension. It is also a classic source of toolchain-specific
link failures, and no non-Windows compiler has built this code at all yet. Adding it now would
mean debugging two new variables at once; it is worth revisiting once the matrix is green.

**`cxx_std_17` via `target_compile_features`** rather than a hardcoded `/std:` or `-std=` flag, so
CMake picks the right spelling per compiler. The standard matches gropt's core target, so the two
libraries can be linked into one build if a shared waveform core is ever wanted.

**Repository options.** `.gitattributes` decides line endings rather than each clone's
`core.autocrlf`: `* text=auto` stores text as LF, and `*.pyi` and `*.yml` are pinned to LF
specifically. The `.pyi` because it is generated, committed and byte-compared by
`pixi run check-stub`, which has to mean the same thing on every platform; the workflow YAML
because it carries shell heredocs that a stray CR would break on a Linux runner. No tracked blob
held a CR when this was added, so it is a no-op for history. What is committed and why: `pixi.lock`
(so the nanobind that writes the stub is pinned), `seq2waveform/_core.pyi` (so editors have the
API without a build first), and `extern/fast_float` vendored (so the sdist and conda's offline
build phase need no network).

## Continuous integration

Two workflows under `.github/workflows/`.

`ci.yml`, on every push and pull request, in about five minutes:

| job | what it proves |
|---|---|
| `core` (linux-gcc, macos-clang arm64 and x86_64, windows-msvc) | the core library compiles and its 7 ctest suites pass on four toolchains, with no Python involved (`-DS2W_BUILD_MODULE=OFF`) |
| `package` (3 platforms, 3.10 and 3.13) | the bindings compile, nanobind's stub generation runs, the wheel installs, and it imports and computes from outside the source tree |
| `sdist` | the sdist carries `extern/fast_float`, passes `twine check`, and builds and installs from source alone |

The `package` matrix includes `3.14t`, the free-threaded interpreter, which is the only place the
free-threaded module build is exercised at all — see *Free-threaded Python* below.

`wheels.yml`, on a `v*` tag, from the Actions tab, or on a pull request that edits the workflow or
`pyproject.toml`: the full cibuildwheel matrix, 24 wheels — CPython 3.10 to 3.14 plus 3.14
free-threaded, on Linux x86_64, macOS x86_64 and arm64, and Windows x64, six per runner. It is not
on every push because it takes tens of minutes and `core` plus `package` already catch a toolchain
break; run it by hand the first time, since **no non-Windows compiler has ever built this code**.

On a tag, a `version` job runs first and fails if the tag does not match `project.version` in
`pyproject.toml`. The wheel build triggers on the tag but takes its version from the file, so
tagging `v0.2.0` without bumping it produces a full matrix of wheels labelled 0.1.0, and nothing
downstream notices, because a wheel's version is simply whatever the metadata said. Taken from
gropt's `build-and-publish.yml`, which had this check when these workflows did not.

**Both macOS architectures are built natively**, one runner each (`macos-latest` for arm64,
`macos-15-intel` for x86_64), rather than cross-compiling x86_64 from the Apple Silicon runner.
That is the difference between a wheel that is checked to build and one that is checked to run:
cibuildwheel's `test-command` and nanobind's stub generation both need to import the module they
just built, so neither happens for a cross-compiled wheel. `archs = ["auto"]` is what selects
this; listing both architectures on one runner is what would cross-compile.

**Hardening, which is mostly about what a workflow is allowed to do.** Both files declare
`permissions: {}` at the top and re-grant `contents: read` per job, so a compromised step cannot
write to the repository; `actions/checkout` runs with `persist-credentials: false`, so the token
is not left in `.git/config` where a later step or a build dependency could use it; and every job
has a `timeout-minutes`, which on a private repository is the difference between a hung build
costing six minutes and six hours of quota.

**`.github/dependabot.yml` exists because the pins went stale silently.** The first version of
these workflows pinned `actions/checkout@v4`, `setup-python@v5` and `upload-artifact@v4`; current
is v7, v7 and v7, with `download-artifact` at v8. Nothing was broken — GitHub still ships patches
to the old major lines, and released one for every line from v2 to v7 on the same day in July
2026 — but a pinned tag keeps working indefinitely while quietly ceasing to receive fixes, and
eventually runs on a Node runtime being retired. One grouped weekly PR turns "stale forever" into
"a diff to review". `pypa/cibuildwheel` stays pinned to an exact patch deliberately: a bump
changes which interpreters and platform tags come out, and v4 made `delvewheel` the default
Windows repair step and `abi3audit` a default post-repair audit. Neither should matter here —
nothing links a non-system DLL and nothing claims the stable ABI — but they are defaults
inherited by pinning v4, and so the first place to look if a Windows wheel build fails in the
repair step rather than in the compile.

**What CI cannot check, which is most of the suite.** The corpus is ~1.6 GB and lives nowhere near
a runner, so `test-h5`, `test-invariants` and the per-dataset half of `test-golden` all skip there
— roughly seventy tests, including every comparison against Pulseq's reference `.h5` and every
golden snapshot. A green CI run is **not** equivalent to a green `pixi run test` on a machine that
has the corpus, and it is the corpus-bearing suites that gate a release. What does run is the
`no_corpus` marker: the digest machinery's own negative controls and the check that the gradient
modes differ somewhere, which need only the committed snapshots. `pytest` exits 5 on a selection
that matches nothing, so if those tests are renamed or lose the marker the job fails rather than
passing vacuously — the one guard that stops "everything skipped" from reading as success.

### macOS specifics, since that is the untested platform

- **Deployment targets are per-architecture**, via two `[[tool.cibuildwheel.overrides]]` blocks:
  11.0 for arm64, because that is when the architecture appeared and a lower value tags a wheel
  for a system that cannot run it, and 10.14 for x86_64. The second is about the C++17 library
  exceptions that live in the platform's libc++ dylib rather than in a header —
  `bad_variant_access` and its siblings are availability-annotated, and below 10.14 a throwing
  `std::get` on a variant either fails to compile or aborts instead of throwing.
- **Nothing throws one any more.** `build_grad_piece` was the only call that could, and it now
  uses `std::get_if`, which is header-only and unconditionally available. The variant has two
  alternatives and the `Trap` one returns earlier, so the null branch is unreachable.
- **x86_64 is built natively**, on `macos-15-intel`, so both macOS wheels are imported and run.
  The cross-compilation path still exists for anyone who wants it locally — `CMakeLists.txt`
  skips stub generation when `CMAKE_OSX_ARCHITECTURES` differs from `uname -m`, because it cannot
  import a foreign-architecture module — but CI no longer takes it. (`macos-13` was the old Intel
  runner label and is retired; `macos-15-intel` replaces it.)
- **The golden snapshots should hold, and arm64 is the reason they might not have.** They are
  bit-exact and were written by MSVC on x86-64. FMA is in the arm64 baseline, so Clang would have
  fused the trapezoid accumulations and moved the last bit — which is why
  `-ffp-contract=off` is pinned for the GNU-like compilers (see *Compile and repository options*).
  With that in place, and with the library's only `libm` calls not reaching any snapshot column,
  a mismatch here is a real difference rather than expected noise and is worth reading before
  regenerating anything. `S2W_GOLDEN_RTOL=1e-12` on a machine with the corpus remains the way to
  tell the two apart: a pass says float noise. That fallback only sees the statistics and the
  probes, so it is for bring-up, not everyday use.

### Free-threaded Python

The wheels include `cp314t`, and the module declares that it does not need the GIL. Without that
declaration a free-threaded interpreter re-enables the GIL for the whole process when it imports
the module, and warns — a silent performance cliff rather than an error, which is why CI asserts
`sys._is_gil_enabled()` is false after the import rather than trusting a green run.

The mechanism is `FREE_THREADED` on `nanobind_add_module`, not a C++ declaration: nanobind owns
the `Py_mod_gil` slot. (`nb::mod_gil_not_used()` does not exist — `NB_MODULE` takes two
arguments.) nanobind sets the option to FALSE itself when the interpreter is not free-threaded, so
it is unconditional here and compiles to nothing on a normal build. Which also means `3.14t` in
the `package` matrix is not a nice-to-have: it is the only configuration where the flag does
anything, so without it the free-threaded build is shipped untested.

What makes the claim true, audited rather than assumed:

| checked | found |
|---|---|
| namespace-scope mutable state in `src/`, `bindings/` | none |
| `mutable` members | none |
| lazily-computed or cached fields | none |
| Python objects held in C++ | none |
| `rv_policy` / `keep_alive` views into object-owned buffers | none; every result is freshly allocated and handed to numpy with its own capsule owner |
| public methods on the shared types | all `const`, so two threads calling into one `Sequence`, `DsvSet` or `PiecewiseLinear` are both reading |
| function-local statics | one, `static const DsvChannel empty` in `dsvset.cpp`; `const`, and C++11 makes the initialisation thread-safe |
| global C state (`setlocale`, `rand`, `localtime`, `strtod`) | none — `scan.hpp` exists partly because `strtod` honours the locale |
| `std::fprintf(stderr)` warnings | two; the C runtime locks the stream, so concurrent warnings can interleave in the text but cannot race |

If any of that changes — a cache, a lazily filled field, a non-const accessor — the flag has to be
reconsidered, and `nb::ft_mutex` is the tool for it. The audit is restated next to the flag in
`CMakeLists.txt`, because that is where someone will be standing when it stops being true.

### What CI deliberately does not do

- **No lint or format gate.** There is no `ruff`, `black` or `pre-commit` configuration in the
  repository, so there is nothing for a lint job to run. Adding one is a policy decision with a
  one-time cleanup attached, not a CI fix.
- **No byte comparison of the committed stub.** `seq2waveform/_core.pyi` is generated by nanobind
  and committed, so it can go stale, but the check belongs locally rather than in CI:
  `pip install .` resolves `nanobind>=3.0,<4` to whatever is newest on PyPI, so a byte comparison
  there would flap on a nanobind release instead of on a real change. `pixi run check-stub`
  (wired into `pixi run test`) does it where `pixi.lock` pins the generator, and `.gitattributes`
  pins the file to LF so the comparison means the same thing on every platform. `wheels.yml`
  keeps the broader `git diff --exit-code` after a wheel build, which is about a build dirtying
  the checkout rather than about the stub specifically.
- **No publishing.** This project installs from a local conda channel built by
  `pixi run conda-build`, not from PyPI, so there is no upload job. `wheels.yml` records what one
  would need — `environment: pypi`, `id-token: write`, `attestations: write`, `actions/attest`
  and trusted publishing — rather than a long-lived token.
- **No ARM Linux or ARM Windows wheels.** `ubuntu-24.04-arm` and `windows-11-arm` are free for
  public repositories and nothing in the code is architecture-specific; it is a question of what
  is worth shipping, not of whether it would work.

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
seq2waveform/
  __init__.py                  re-exports from ._core
  _core.pyi                    GENERATED on every build — never edit by hand
  py.typed
tests/
  conftest.py                  finds the test corpus; shared by both Python suites
  test_pwl.cpp                 C++ core tests (ctest), one per module
  test_h5.py                   against Pulseq's own reference files
  test_invariants.py           our own paths against each other
  test_golden.py               every output column vs a committed digest
  golden.py                    the digests, shared by the test and the generator
  golden/<dataset>.json        the snapshots — regenerate deliberately, review the diff
  bench_expand.cpp             native timings, no Python boundary; also the allocation probes
scripts/
  bench.py                     timings through the Python API
  gen_golden.py                rewrites tests/golden/*.json
.github/
  workflows/ci.yml             core + bindings on 4 toolchains, every push
  workflows/wheels.yml         the full cibuildwheel matrix, on a tag or by hand
  dependabot.yml               one grouped weekly PR for the action pins
.gitattributes                 LF for the generated, committed .pyi
recipe.yaml                    wraps a built wheel into a conda package
```

Environment and tasks live in `pyproject.toml` under `[tool.pixi.*]` rather than a separate
`pixi.toml`, so the build requirements sit next to the environment that satisfies them — the same
layout as gropt.

Keep the core free of nanobind: only `bindings/` includes it, so the core stays linkable from
plain C++ (tests, gropt, a future CLI).

## Build and test

```bash
pixi run build            # editable install + regenerate _core.pyi
pixi run test-cpp         # C++ core tests via ctest
pixi run test-h5          # against Pulseq's own reference .h5, at 1e-4
pixi run test-invariants  # our own paths against each other, on the real datasets
pixi run test-golden      # every output column against tests/golden/*.json
pixi run golden-update    # rewrite those snapshots (deliberately — review the diff)
pixi run test             # all four
pixi run bench            # timings on the real datasets
pixi run build-wheel      # dist/*.whl
pixi run conda-build      # wrap that wheel into a local conda channel under output/
```

These answer three different questions and none subsumes another: `test-h5` is an outside answer
but a tolerance one on three quantities, `test-invariants` is exact but grades this code against
itself, and `test-golden` is exact on real data but only detects change. Run all of them.

`cmake` and `nanobind` come from the pixi environment; MSVC comes from the system Visual Studio
install, as in gropt. `ninja` is deliberately **not** a dependency — without it CMake uses the
Visual Studio generator, which finds MSVC without `vcvars` on `PATH`.

`h5py` is a dev dependency, for `test-h5` only — the package itself needs nothing but numpy.

## The test corpus: matched `.seq` / `.h5` / `.dsv`

`tests/conftest.py` finds the corpus: `S2W_TESTDATA` if set, then `./testdata`, then
`../seq2waveform/testdata` — the retired Rust checkout that owned it originally, kept as a fallback
so a machine that has not moved the ~1.6 GB yet still runs. It is gitignored wherever it sits, so
it exists on this machine only: a fresh clone has no data, and every data-driven test skips.

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

## The two sources agree physically

Once `DsvSet` landed, the same k-space could be computed two independent ways — from the `.seq`
through the Pulseq expansion, and from the Siemens DSV channels — and the peak magnitudes match:

| dataset | max abs kz from `.seq` | max abs kz from `DsvSet` |
|---|---|---|
| cones | 399.6 1/m | 399.6 1/m |
| girf | 503.4 1/m | 503.4 1/m |

That is not a tested assertion (the two paths are different arithmetic on different files, so they
agree to physics rather than to the bit), but it is a useful sanity check that the DSV frame, the
~1020 µs lead-in and the mT/m → Hz/m conversion are all right at once. If a future change makes
these diverge, something structural broke.

## What the corpus does and does not exercise

Three branches are reachable only from synthetic input, because the five datasets simply never hit
them. Each has a hand-built test, and each is somewhere a port drifts quietly — so check them by
hand if you touch the matching code.

| branch | why the corpus misses it | covered by |
|---|---|---|
| **Asymmetric RF center** (sub-raster RFD peak: plateau midpoint, or parabolic refinement) | measured: **zero** off-centre pulses across all five datasets — every pulse is symmetric, so the center is always the exact window middle | `test_asymmetric_rf_center_matches_rust` (a synthetic INF + RFD pair, with the expected centers derived by hand so a *shared* drift is caught too), plus two unit tests in `test_inf.cpp` |
| **ADC with no preceding excitation** (far-past sentinel reference, `(nan, nan, nan)`) | every ADC in every dataset has an excitation before it | `test_unreferenced_adc_matches_rust`, `adc_with_no_preceding_excitation` |
| **restore's main reconstruction** | `cones` is a spiral, so all 3531 of its shapes take the fallback | `girf` (20 shapes) and `restore_recovers_a_triangle` |

What the corpus *does* exercise, worth knowing so a change there is not mistaken for noise: the
`MaxAdcSegmentLength` merge fires on two datasets (`sync_osc` merges 20 raw segments into 5 logical
events, `girf` 1200 into 300) and not on the other three; the lead-in is 1020 µs on all five; and
the excitation/ADC count mismatch warning fires on four of five.

**A trap when hand-building a `.dsv` fixture:** the `[VALUES]` body is itself run-length encoded,
so `d, d, k` means "`d` repeated `k+2` times". A delta list with a repeated *adjacent* value
decodes to something quite different from its `cumsum`. The synthetic RFD in the parity suite
deliberately has no two adjacent deltas equal; the first version of that test did not, and the
resulting envelope was nothing like the intended one.

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
float noise. `tests/test_invariants.py` asserts `np.array_equal` wherever both sides are our own
arithmetic, and the differential suite did so throughout. If a check genuinely cannot be exact,
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
7. ~~**`parse_model` / `parse_shapes` exist purely for parity testing.**~~ Closed: both were
   deleted at cut-over, along with the row type aliases they needed (128 lines of bindings).
8. ~~**`-DS2W_BUILD_TESTS=ON` required Python and nanobind**~~ although the pure-C++ tests link
   neither — `find_package(Python)` and the module target were unconditional. Closed by
   `S2W_BUILD_MODULE`, which defaults to ON (every `pip install` builds the module) and OFF builds
   the core library and its ctest suites alone. That is what `pixi run test-cpp` and the CI `core`
   job now use, so a compiler error in the core arrives without a Python stack in the way.

## How the port went, step by step

Kept as a record of what each stage was actually checked against — useful if a bug ever turns up in
one of these areas, because it says what was and was not proven at the time.

Each step ported a module, added its C++ tests, extended the differential suite, and kept
`pixi run test` green.

1. `time.hpp`, `pwl.cpp` — the `Ps` time type and the piecewise-linear core.
2. `model.hpp` + `parser.cpp`, plus `scan.cpp` and `error.hpp`, which Rust gets from its standard
   library. Checked on a block count, then every event library and every shape field by field, and
   the parse-error message verbatim.
3. `waveform.cpp` — 24.8 M gradient corners over three modes and three axes, 27.3 M ADC sample
   times, 4,601 RF events and 4,261 triggers, all exactly equal.
4. `sample.cpp` — `Sampler`, the spec builders, k-space, the flat batched path, the dense sampling,
   and N-D time arrays in the bindings. 27.9 M flat k-space points compared exactly. This step also
   added `test-h5`, the first check that did not depend on the Rust code being right.
5. `dsv.cpp` — checked against both the corner and the dense form, over every channel in the corpus
   rather than just the gradients. See **The `.dsv` channel kinds**.
6. `inf.cpp` — the lead-in, the ADC times and grouping, the SYNC events and the RF events on all
   five datasets, plus a synthetic pair for the asymmetric RF center (see below).
7. `DsvSet` — the second `WaveformSource`. `Sequence` and `DsvSet` bind their shared surface from
   **one template** in the bindings, so they cannot drift apart;
   `test_dsvset_and_sequence_expose_the_same_interface` still asserts it. The surfaces were checked
   method for method against the Rust classes: 21, 18 and 3 names, none missing, none extra.
8. Performance. With parity reached and both sides in release, five allocation bugs — one of them a
   47-second `memcpy` — came out. See **Whole-pipeline performance, measured**.
9. Cut-over. Renamed the package to `seq2waveform`, retired the oracle and the differential suite,
   kept its oracle-free half as `tests/test_invariants.py`, folded the Rust repo's dataset suite
   into `test-h5` (three of its four tests were already there; the fourth, DSV-vs-h5, was not),
   deleted the `parse_model` / `parse_shapes` scaffolding, and ported `recipe.yaml` to wrap a
   scikit-build-core wheel.

### What the cut-over gave up

Worth being explicit, because it is the main risk this document exists to flag. The differential
suite asserted *bit-exact* equality against an independent implementation across 284 cases. Nothing
left is that strong. In its place:

- `test-h5` — an outside answer, but a tolerance one (1e-4), and only on gradients, ADC times and
  excitation centers. It says nothing about triggers, RF metadata, ADC grouping, the INF, or the
  DSV decoder's edge cases.
- `test-invariants` — exact, but self-referential: it pins paths to each other, not to truth.
- the C++ unit tests (`tests/test_*.cpp`, 7 ctest suites) — ported from the Rust unit tests, so they
  carry that coverage forward, but they are small fixtures rather than real data.
- `test-golden` — exact, and on real data, but a snapshot of this code's own output: it detects
  change, not error. Added right after the cut-over; see below.

The gap was therefore **anything on real data that only the oracle covered**: triggers, RF event
metadata, ADC event grouping after the `MaxAdcSegmentLength` merge, and the INF readers, all still
checked on fixtures but no longer on the corpus.

That gap is now closed by `test-golden`, which is the fourth entry in the list above and the reason
this section is history rather than a warning:

- `test-golden` — 97 digested output columns per dataset in `tests/golden/*.json`, bit-exact
  against the corpus. It covers exactly what nothing else did, and it was generated while the code
  was still the code that had just been proven at parity, so the reference is not re-derived
  ground truth — it is the output the oracle signed off on.

What a snapshot *cannot* do is tell you a number is right; it only says it has not changed. So the
shape of the safety net is now: `test-h5` asks whether the waveforms are correct, `test-invariants`
asks whether the internal routes agree, and `test-golden` asks whether anything moved. Losing the
oracle cost the ability to answer the first question about triggers, RF metadata, ADC grouping and
the INF on real data — that part is genuinely gone, and only a new external reference would bring
it back. What was recoverable has been recovered.

Two details worth knowing before trusting a green `test-golden` run:

- **It is sensitive, and that is checked.** `tests/test_golden.py` carries negative controls that
  need no corpus: a one-ULP change, a reordering that leaves every statistic identical, a moved
  NaN, a length change and a string change all have to fail the digest. A further control asserts
  that `centers` and `dsv` gradient modes differ from `pulseq` *somewhere* in the corpus — they
  agree exactly on all-trapezoid sequences by design, so without that check 12 of the 18 gradient
  columns could quietly be pinning nothing.
- **It is bit-exact, which is a platform claim.** The snapshots were written by MSVC on x86-64. A
  different compiler may contract a multiply-add differently and shift the last bit. Run with
  `S2W_GOLDEN_RTOL=1e-12` on first contact with a new platform: a pass says the difference is
  float noise, and the snapshots can be regenerated there. That fallback only sees the statistics
  and the probes, so it is for platform bring-up, not for everyday use.

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
- **Never hand-edit `seq2waveform/_core.pyi`.** It is regenerated on every build.

---

## If you are picking this up cold

Read **Edge cases and conventions that must survive the port** first — it is most of the real
knowledge here, and almost every line of it is a bug someone already paid for.

Then: `pixi run test` must be green before and after whatever you change, and `pixi run test-h5`
is the one that matters most, since it is the only check that does not grade this code against
itself. If you touch anything in the hot paths, read **Whole-pipeline performance, measured**
before optimising — four of the five bugs it describes were introduced by code that looked
obviously correct, and one of them survived five rounds of testing because it was correct, just
500x too slow.
