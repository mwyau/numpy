# DUCC-backed `numpy.fft` optimization findings

Status: final local implementation selected. This report covers only the
existing one-dimensional `numpy.fft` gufunc backend. No native multi-axis FFT
work is part of this change.

## Executive conclusion

The final implementation makes four targeted changes:

1. The default/backward normalization factor is created in the input real
   precision, so public `float32`/`complex64` calls select DUCC's matching
   single-precision loop.
2. Direct c2c calls with one transform are presented to DUCC as a true
   one-dimensional shape with axis 0. The real r2c/c2r calls retain their
   existing generalized wrapper shape.
3. Direct contiguous batches retain DUCC's generalized one-axis path. A
   low-level `pocketfft_c<T>` plan and scratch buffer are reused only for
   non-direct/staged c2c rows, where the old wrapper was reconstructing a plan
   once per row.
4. Overlapping c2c views remain on the existing staged path so all input rows
   are snapshotted before output writes. DUCC's global FFT cache and worker
   threading remain disabled.

This is the smallest evidence-supported design. The true 1-D path removes the
known `n=4096` single-transform cliff. The staged local plan removes repeated
plan construction. A full contiguous-batch comparison did not show a stable,
architecture-independent crossover in favor of a row loop; keeping DUCC's
generalized batch path avoids adding a size/dtype heuristic.

The final validation results are:

* 2,034 old-versus-final numerical cases passed, including all requested
  lengths, representative batches, layouts, padding/truncation, norms,
  output variants, overlap cases, and long-double variants.
* The NumPy FFT test module passed 172/172 tests.
* In the final direct API run, 896 cases comparing final to unchanged c222 had
  a median time ratio of `c222/final = 1.001` and geometric mean `1.072`.
  The separate c2c grid was `1.036` median / `1.143` geometric mean, and the
  secondary storage grid was `1.025` / `1.108`.
* The direct final-versus-parent pocketfft run was `old/final = 0.911` median
  / `0.890` geometric mean. The remaining old-relative regression is
  concentrated in large complex128 batches, where DUCC's generalized
  algorithm is slower than the former pocketfft algorithm. It is not caused
  by the final wrapper changes: final and c222 use the same generalized batch
  path for those cases. The formerly severe single c2c case is fixed; for
  example, final `fft(complex64, n=4096, batch=1)` measured 1.21x old and
  `ifft(complex64, n=4096, batch=1)` 1.42x old in the final direct run.

Ratios in this report are always `previous_time / candidate_time`; values over
one mean the candidate is faster. Very small sub-millisecond cases and very
large single-sample cases can have noisy extrema, so medians, geometric means,
and repeated focused cases carry more weight than isolated minimum/maximum
ratios.

The option-3 DUCC-side follow-up is a separate local proposal. It does not
change the production conclusion or the source in this commit.

## Scope and invariants

* Required starting commit: `c222665be45cbd361f2b0e96f6ff81f12b390ebd`,
  subject `ENH: migrate numpy.fft to duccfft`.
* Previous backend comparison: parent commit
  `1819fd4beec3d15cc24cd6237b7d55e3a2debda4`.
* Vendored DUCC: 0.41.1, commit
  `64f42ba531f609ba7029c82207a063b17f9d5275`.
* Only the one-dimensional gufunc backend is covered. The benchmark scripts
  never call a multi-axis FFT API.
* Production policy remains single-threaded internally and cache-off.
* The implementation contains no size-specific 4096 special case and no
  speculative cache or blocking threshold.

## Baselines and reproducibility

The parent commit was confirmed as the pre-migration pocketfft backend and was
built separately from unchanged c222. Each candidate was installed into its
own staging directory and tested in a fresh worker process. The benchmark
workers use `python -S`, run from `/tmp`, and set all relevant external thread
counts to one. Full environment details are in
[`ENVIRONMENT.md`](ENVIRONMENT.md), and exact build/test/benchmark commands
are in [`COMMANDS.md`](COMMANDS.md).

The authoritative matrix contains these lengths:

`64, 256, 1024, 2048, 3072, 3584, 4095, 4096, 4097, 4608, 8192, 16384,
32768, 65536`.

It covers batches `1, 2, 4, 8, 16, 32, 64, 256` and:

* `fft`/`ifft` with `complex64` and `complex128`;
* `rfft` with `float32` and `float64`;
* `irfft` with `complex64` and `complex128`.

Each case is warmed, timed with adaptive inner loops, sampled repeatedly, and
summarized by median, best, and median absolute deviation. Candidates are
never timed simultaneously. The raw timing files are under
[`results/`](results/).

## Root-cause analysis

### One transform was disguised as a 2-D generalized transform

At c222, even a single direct transform was passed as shape `{1, n}` and
axis `{1}`. DUCC consequently used its generalized path. Diagnostic output
shows, for the public double c2c loop at `n=4096`, `ndim=2`, `axis=1`,
effective complex SIMD width 2, `n_simul=2`, `n_bunch=2`, and 132,096 bytes of
temporary storage. A true 1-D call reports DUCC's `simple` c2c path, axis 0,
and zero temporary storage. The complete diagnostic summary is in
[`DIAGNOSTICS.md`](DIAGNOSTICS.md).

This explains the reproducible c2c `n=4096` cliff. It is not explained by a
simple 512-KiB transition: the measured generalized working set at 4096 is
328,192 bytes for the vector estimate and the generalized path is still using
two simultaneous transforms. At `n=8192`, DUCC does change to
`n_simul=1`/`n_bunch=1`, but that is a distinct transition.

The final c2c-only shape change reaches DUCC's simple path for `n_outer == 1`
while preserving the existing alignment, stride, exact-in-place, and overlap
checks. r2c/c2r were deliberately not changed to use the c2c specialization.

### Staged rows reconstructed plans

With `DUCC0_NO_FFT_CACHE`, c222's staged wrapper calls a high-level transform
once per row. A diagnostic zero-padding call at `n=4096` requests one simple
plan per row. A focused probe measured these cache-off warm times:

| `n=4096`, batch | c222 staged | local c2c plan | improvement |
|---:|---:|---:|---:|
| 1 | 0.033 ms | 0.029 ms | 12% |
| 8 | 0.217 ms | 0.168 ms | 23% |
| 64 | 1.714 ms | 1.235 ms | 28% |

The local path constructs one `ducc0::pocketfft_c<T>` and one aligned scratch
buffer per gufunc invocation. It copies each row into the reusable work area,
executes the plan, and copies the result out. It is used only after the direct
view path declines. If the conservative input/output range check finds an
overlap, the code returns to `run_staged()` so all rows are copied before any
output is written.

### Public single-precision calls selected the double loop

The ufunc signatures contain both float and double loops. At c222, the Python
wrapper passed integer `fct=1`; the public ufunc resolver therefore commonly
selected the double loop even for `complex64`/`float32` inputs. A directly
typed factor selects DUCC's float loop, whose V2 effective width is four
complex scalars rather than two double complex scalars.

The final wrapper uses `a.real.dtype.type(1)` only for the exact-one
backward/default normalization case. Non-default normalization still uses the
existing result-precision calculation. This is semantically the same factor
but preserves the input-precision loop where it exists.

The matched typed-factor experiment is in
[`api_matrix_fct.csv`](results/api_matrix_fct.csv). On its 896 direct cases,
post-change/pre-change was `1.002` median / `0.945` geometric mean when
expressed as `final_with_typed_factor / final_with_integer_factor` (lower is
faster). The useful operation-specific effect was `fft complex64`: the
pre/post ratio `pre/post` was `1.265` median / `1.537` geometric mean in the
same run. `complex128`, inverse transforms, and real double precision stayed
near parity. The awkward-length outliers are retained in the raw CSV rather
than hidden.

### Ordinary contiguous batches

The first local-plan experiments are retained in
[`c2c_batch_compare.csv`](results/c2c_batch_compare.csv),
[`c2c_batch_compare_novec.csv`](results/c2c_batch_compare_novec.csv), and
[`c2c_batch_compare_final.csv`](results/c2c_batch_compare_final.csv). They
were staged/local-plan controls, not a clean ordinary-direct row-loop
comparison, so they should not be read as evidence for a direct contiguous
batch policy. The scalar control did expose a useful failure mode:
`ifft(complex64, n=4097, batch=4..256)` regressed by roughly 2.4--2.5x versus
c222.

A fresh direct-only candidate was then built from the current production
commit in `/home/albert/numpy-rowloop`. Before `run_direct()`, it recognizes
only ordinary contiguous batches (`n_outer > 1`, no factor stride, equal
input/output length, and unit element strides), constructs one local
`pocketfft_c<T>` plan, and loops over rows. It leaves the true 1-D single
transform, generalized fallback, real transforms, and overlap handling
unchanged. The exact source patch is
[`ducc-direct-rowloop.patch`](experimental_patches/ducc-direct-rowloop.patch),
and the full raw run is
[`c2c_batch_compare_rowloop.csv`](results/c2c_batch_compare_rowloop.csv)
(1,792 successful rows).

Ratios below are previous/candidate, so values over one favor the row loop:

| comparison | all 448 median / gmean | large focus median / gmean |
|---|---:|---:|
| c222 / direct row loop | 1.066 / 1.083 | 1.016 / 1.085 |
| current final / direct row loop | 0.997 / 0.937 | 0.871 / 0.839 |
| old pocketfft / direct row loop | 0.868 / 0.885 | 0.528 / 0.599 |

The aggregate is not a portable win: in the large focus, row-loop
`fft(complex64)` was `1.762`/`1.783` versus c222 by median/geometric mean,
while `ifft(complex64)` was `0.732`/`0.781`. Complex128 was mixed (`fft`:
`1.065`/`1.065`; `ifft`: `0.941`/`0.936`). The worst c64 regressions were
the awkward-length inverse cases, including the known `n=4097` cluster.
The focused RSS run is
[`rss_rowloop_complex128.csv`](results/rss_rowloop_complex128.csv): the row
loop added `+252`, `+596`, `+308`, and `+312 KiB` at batches 8, 32, 64, and
256, respectively, with no stable speed benefit in that probe.

This clean control confirms that a direct local-plan row loop can help some
complex128 batches, but it also creates stable complex64 regressions and does
not recover the parent pocketfft behavior. It is rejected as a generalized
policy. The DUCC-side heuristic/policy experiments below are the upstreamable
alternatives; the rejected designs are summarized in
[`experimental_patches/README.md`](experimental_patches/README.md).

The final direct-batch result is therefore the simpler generalized DUCC path.
The new local plan is limited to staged/fallback rows, where its benefit is
clear and it does not require a speculative crossover rule.

### Phase 10: DUCC-side automatic and explicit batch policies

The generalized-batch question was revisited in a separate worktree rooted at
the preserved production commit, with the vendored DUCC tree at
`64f42ba531f609ba7029c82207a063b17f9d5275`. The experiment stayed inside the
existing one-dimensional NumPy c2c wrapper; `_ducc_nd_umath`, native N-D
dispatch, and all N-D work remained untouched.

Two upstreamable designs were tested:

* The automatic DUCC candidate adds an internal policy abstraction and, only
  for contiguous c2c double batches with compiled SIMD width at most two,
  selects `n_simul=n_bunch=vlen` after the existing working-set heuristic.
* The explicit candidate adds a named internal
  `batch_policy::vectorize_contiguous` entry point. The temporary adapter
  calls it only for batched complex128 c2c; it exposes no raw `n_simul` or
  policy argument to Python.

The focused automatic comparison uses `default / automatic` ratios, where
values above one mean the candidate is faster:

| build / dtype | cases | median | geometric mean |
|---|---:|---:|---:|
| V2 / complex64 | 32 | 1.004 | 1.005 |
| V2 / complex128 | 32 | 1.668 | 1.515 |
| V3 / complex64 | 32 | 0.999 | 0.999 |
| V3 / complex128 | 32 | 0.999 | 1.006 |

This is materially better than forcing vectorization for every dtype/ISA:
the V2 complex128 cluster improves, while complex64 remains at parity and
V3 does not pay the forced-vectorization cost. In the same-build public API
control, automatic/explicit ratios for unaffected c2c complex64 rows were
`1.012 / 1.011` on V2 and `1.001 / 1.000` on V3 by median/geometric mean.
The real-transform rows were `0.997 / 1.002` and `1.002 / 1.004`, respectively.
These are neutral noise bands; neither candidate changes the real-transform
source path.

The exact scoped explicit-policy run is
[`c2c_batch_compare_ducc_policy.csv`](results/c2c_batch_compare_ducc_policy.csv).
It contains 1,792 successful rows over the full required c2c matrix and
compares old pocketfft, c222, current final, and the DUCC policy candidate:

| comparison | all 448 median / gmean | large focus median / gmean |
|---|---:|---:|
| c222 / current final | 1.033 / 1.148 | 1.033 / 1.288 |
| current final / DUCC policy | 1.001 / 1.053 | 1.028 / 1.235 |
| c222 / DUCC policy | 1.056 / 1.209 | 1.640 / 1.591 |
| old pocketfft / DUCC policy | 0.807 / 0.883 | 0.786 / 0.901 |

The large focus is `n >= 8192` with batches 8, 32, 64, and 256. The policy
does not erase DUCC's remaining algorithm-level large complex128 gap to old
pocketfft, but it improves the generalized DUCC path without a stable
complex64 regression relative to current final.

Peak-RSS measurements were taken in a fresh process for each case. At
`fft(complex128, n=65536)`, explicit minus automatic RSS was approximately
`+0.1--0.3 MiB` on V2 and `+5.7--7.2 MiB` on V3 across batches 8, 32, 64, and
256. Complex64 deltas stayed within the measurement noise band. This makes
the explicit policy a reasonable fallback hook but not a default choice for
wider SIMD builds.

The final scoped policy stage passed all 2,034 old-versus-candidate numerical
cases and `numpy.fft.tests` passed `172/172` in both V2 and V3 builds. The
builds retained `DUCC0_NO_FFT_CACHE` and `DUCC0_NO_LOWLEVEL_THREADING`, and
all workers used one external thread plus DUCC `nthreads=1`.

Decision: retain the current NumPy implementation and do not vendor the DUCC
patch locally yet. The automatic rule is the preferred upstream proposal and
the explicit internal enum is the fallback, but this evidence is still from
one Ryzen host and two x86 ISA builds. DUCC-level cross-architecture,
cache-size, memory, and caller-compatibility validation plus upstream review
are required before changing the vendored 0.41.1 source. Full design detail,
patches, commands, and raw results are in
[`DUCC_PROPOSAL.md`](DUCC_PROPOSAL.md).

### Generalized blocking and the 8-KiB transition

DUCC's generalized path estimates working set and chooses `n_simul` and
`n_bunch`. The diagnostic build measured:

| c2c call | SIMD width | `wss(1)` | `wss(vlen)` | `n_simul` | `n_bunch` |
|---|---:|---:|---:|---:|---:|
| public double, `n=4096` | 2 | 197,120 | 328,192 | 2 | 2 |
| public double, `n=4097` | 2 | 212,144 | 343,248 | 2 | 2 |
| public double, `n=8192` | 2 | 393,728 | 655,872 | 1 | 1 |
| directly typed float, `n=4096` | 4 | 98,560 | 295,168 | 4 | 4 |
| directly typed float, `n=8192` | 4 | — | — | 1 | 1 |

The known 4096 cliff occurs before the `n_simul` drop, so changing DUCC's
working-set heuristic would not be a causal fix. The final patch does not
modify vendored DUCC or introduce a host-specific L2 threshold. The remaining
large-batch behavior is documented as an algorithm-level difference between
DUCC and the former pocketfft implementation, not treated with a benchmark-
specific heuristic.

### Real transforms

r2c and c2r use separate DUCC generalized implementations; they do not have
the c2c `exec_simple` branch. A generic true-1-D change was tested previously
and sometimes regressed rfft. Low-level real-plan reuse would require handling
DUCC's real/halfcomplex packing and did not have a broad measured advantage.
The final source leaves real transforms on the existing direct/staged high-
level paths. In the final c222 comparison, direct-operation geometric means
were close to parity for `rfft float64` (1.011) and `irfft complex128` (1.001),
with `rfft float32` benefiting from the typed-factor change (0.939 final/c222
time ratio, lower is faster).

### Cache and threading

The production build continues to define both:

```text
DUCC0_NO_FFT_CACHE
DUCC0_NO_LOWLEVEL_THREADING
```

Removing only `DUCC0_NO_FFT_CACHE` made repeated same-shape calls faster in a
diagnostic experiment, but `DUCC0_NO_LOWLEVEL_THREADING` turns DUCC's mutex and
lock-guard implementations into no-ops while the global FFT cache mutates
static LRU state. That combination is not thread-safe. No cache-on result is
used as a production performance claim, and no worker threading was enabled.
Per-call local plan reuse provides the useful staged benefit without global
state, fork questions, or retained plans.

## SIMD and CPU-dispatch analysis

The normal default build was configured with:

```text
-Dcpu-baseline=min -Dcpu-dispatch=max
```

On this host `min` resolves to `X86_V2`. `_duccfft_umath.cpp` is a normal
extension-module source, not a NumPy multi-target source, so DUCC is compiled
at the baseline ISA in the default build. DUCC diagnostics report effective
complex widths of four floats and two doubles at V2. Explicit equivalent
baseline-only builds were also made with `X86_V2` and `X86_V3`.

The matched final-vs-parent ISA results are:

| build | cases | old/final median | old/final geometric mean | final V2/final V3 geometric mean |
|---|---:|---:|---:|---:|
| X86_V2 | 480 | 0.943 | 0.925 | — |
| X86_V3 | 480 | 0.975 | 0.956 | 1.126 |

The V3 final build is about 12.6% faster than the V2 final build by geometric
mean across this matrix, and the gain is broadest for large transforms. V3
also narrows the old-relative aggregate gap. The old pocketfft build benefits
from V3 as well, so the result is not a DUCC-only comparison.

The final patch does not add runtime NumPy CPU dispatch. The extension module
contains module initialization, gufunc registration, exception translation,
and computational templates in one translation unit. Standalone V2/V3 DUCC
objects were inspected with `nm -C --defined-only` and `readelf -Ws`: the DUCC
header instantiations observed in the shared objects are local (`t`) symbols,
with no global/weak `ducc0::` definitions.

To test the remaining dispatch question rather than infer it from symbols, an
isolated prototype in `/home/albert/numpy-cpudispatch` split only the two c2c
loop entry points through `mod_features.multi_targets()`. It built a V2
baseline and a V2-plus-V3-dispatch extension, linked its own CPU-feature
runtime, and selected the function pointers once at module initialization.
The full raw run is
[`c2c_batch_compare_cpudispatch.csv`](results/c2c_batch_compare_cpudispatch.csv)
(448 cases per candidate); the repeat large-focus run is
[`c2c_batch_compare_cpudispatch_focus.csv`](results/c2c_batch_compare_cpudispatch_focus.csv).
Ratios are previous/candidate:

| comparison | all 448 median / gmean | repeat large focus median / gmean |
|---|---:|---:|
| current / dispatch V2 | 1.002 / 1.009 | 0.998 / 0.997 |
| current / dispatch V3 | 0.995 / 0.986 | 0.954 / 0.965 |
| dispatch V2 / dispatch V3 | 0.990 / 0.978 | 0.966 / 0.968 |

The prototype therefore establishes a working link/runtime topology but not a
positive performance case: the V3 target was slower in the repeat focus, and
the result is not a fix for the contiguous-batch regression. Its focused RSS
probe is [`rss_cpudispatch_complex128.csv`](results/rss_cpudispatch_complex128.csv);
relative to current final, the V3 deltas were `+44`, `+296`, `+464`, and
`+748 KiB` at batches 8, 32, 64, and 256. The V2 deltas were `+60`, `+1,364`,
`+192`, and `+584 KiB`. The dispatch V3 full correctness matrix has 2,034/2,034
successful rows, and both dispatch stages pass 172/172 FFT tests. A V3
dispatch build with `NPY_DISABLE_CPU_FEATURES=AVX2` correctly falls back to
the X86_V2 baseline; this differs from a baseline-only V3 build, which remains
unsafe to run on a CPU without its compiled V3 features.

Given the need for a small portable production diff and the negative dispatch
timing result, the prototype is retained as evidence only. The current final
implementation remains the shipped baseline build and does not rely on runtime
feature branches in every transform call.

## Experiment log

### Phase 0: clean starting point

The investigation branch was created from c222 while preserving the original
branch. The old parent and DUCC vendored commit were recorded before source
changes. Build settings, compiler commands, staging roots, and benchmark
environment are preserved in [`ENVIRONMENT.md`](ENVIRONMENT.md) and
[`COMMANDS.md`](COMMANDS.md).

### Phase 1: baseline matrix

`batch_compare.py` ran the complete 14-length × 8-batch × 2-operation ×
2-complex-dtype c2c grid: 448 cases per candidate. `api_matrix.py` ran 896
direct cases for all eight requested operation/dtype pairs and separate
secondary layout/staging/output/overlap grids. The authoritative parent/final
direct result is [`api_matrix_old_final_fct.csv`](results/api_matrix_old_final_fct.csv);
the c222/final result is [`api_matrix_fct.csv`](results/api_matrix_fct.csv).

### Phase 3: true 1-D c2c

The c2c-only true-1-D variant was compared with unchanged c222 and the old
backend around small, transition, and large sizes. It recovered the c2c 4096
cliff and improved single-transform aggregates. Applying the same generic
change to r2c/c2r was rejected because real paths have different DUCC
implementation branches and no consistent win.

### Phase 4: local c2c plans

The staged probe established one plan request per row with cache-off and a
22--28% warm improvement from one local plan/scratch allocation. The final
implementation extends that idea to all non-direct c2c fallbacks while
explicitly preserving overlap snapshot semantics. Direct contiguous batches
were measured separately and intentionally not routed through this path.

### Phase 5: contiguous-batch alternatives

The initial staged/local-plan variants were retained as controls, but were not
ordinary-direct row-loop tests. A fresh direct-only row-loop patch was built
from the current production commit and benchmarked across the requested c2c
grid. It helped some large complex128 forward cases, but regressed
complex64 inverse cases and remained behind the parent pocketfft path. Its
1,792-row raw result and focused RSS run are retained as
[`c2c_batch_compare_rowloop.csv`](results/c2c_batch_compare_rowloop.csv) and
[`rss_rowloop_complex128.csv`](results/rss_rowloop_complex128.csv). No stable
crossover was found without adding several dtype/operation conditions, so the
direct row loop was rejected.

### Phase 6/7: ISA and dispatch

Equivalent V2/V3 baseline-only builds were created for both parent and final.
DUCC vector widths, target compiler flags, symbol tables, and runtime feature
control behavior were inspected. V3 materially helps the final extension in a
baseline-only build. An actual isolated NumPy multi-target prototype was then
compiled and run; it passed link, fallback, symbol, correctness, and FFT-test
checks, but its repeated large-focus timing was neutral-to-negative versus
the shipped stage. No dispatch code was added to production.

### Phase 8: real operations

The real direct, staged, layout, output, negative-stride, and dtype-punning
overlap cases were included in `api_matrix.py` and `correctness_matrix.py`.
No real low-level plan adapter was retained because the existing high-level
path is already near c222 for the common real operations and packing complexity
would increase the maintenance surface.

### Phase 9: cache

Cache-on measurements were retained only as a diagnostic separation of plan
creation from execution. The no-op mutex configuration is unsafe, so the final
build remains cache-off. Internal DUCC worker threading was not enabled.

## Final benchmark tables

All ratios below are `previous / final`; higher is faster.

### Final versus unchanged c222

| population | cases | median | geometric mean |
|---|---:|---:|---:|
| direct API, all operations | 896 | 1.001 | 1.072 |
| direct API, single (`batch=1`) | 112 | 1.020 | 1.097 |
| direct API, batched | 784 | 1.001 | 1.068 |
| c2c focused grid | 448 | 1.036 | 1.143 |
| secondary layouts/staging/output/overlap | 680 | 1.025 | 1.108 |

For the secondary grid, the geometric means were 1.117 for layouts, 1.114
for staged padding/truncation, 1.075 for explicit outputs, and 1.088 for
overlap cases.

### Final versus parent pocketfft

| population | cases | median | geometric mean |
|---|---:|---:|---:|
| direct API, all operations | 896 | 0.911 | 0.890 |
| single (`batch=1`) | 112 | 0.993 | 1.035 |
| batched | 784 | 0.896 | 0.872 |
| large (`n >= 8192`) | 256 | 0.776 | 0.817 |
| `fft complex64` | 112 | 1.050 | 1.200 |
| `fft complex128` | 112 | 0.751 | 0.732 |
| `ifft complex64` | 112 | 0.648 | 0.771 |
| `ifft complex128` | 112 | 0.744 | 0.720 |
| `rfft float32` | 112 | 0.944 | 0.969 |
| `rfft float64` | 112 | 0.953 | 0.905 |
| `irfft complex64` | 112 | 1.007 | 0.999 |
| `irfft complex128` | 112 | 0.949 | 0.926 |

The table makes the remaining limitation explicit: DUCC does not match
pocketfft for all large complex128 batches on this host. The final wrapper
does, however, remove the known single c2c cliff and improves complex64 FFT
through typed loop selection. The focused and c222-relative tables are the
appropriate evidence for changes made in this task.

### Representative direct cases

| operation/dtype | `n` | batch | old ms | final ms | old/final |
|---|---:|---:|---:|---:|---:|
| fft complex64 | 4096 | 1 | 0.0347 | 0.0287 | 1.21 |
| ifft complex64 | 4096 | 1 | 0.0439 | 0.0308 | 1.42 |
| fft complex64 | 4096 | 8 | 0.1588 | 0.1499 | 1.06 |
| fft complex128 | 32768 | 4 | 0.6471 | 1.6767 | 0.39 |
| ifft complex128 | 32768 | 4 | 0.6459 | 1.6748 | 0.39 |

The last two rows are deliberately included: they show the remaining
algorithm-level parent regression rather than hiding it behind aggregate
figures. The c222-relative timing for these same large batches stays near the
unchanged generalized DUCC path.

### Correctness

[`correctness_old_final_fct.csv`](results/correctness_old_final_fct.csv)
contains 2,034 rows, all with status `ok`. Maximum observed errors by dtype
were:

| dtype | max absolute | max relative |
|---|---:|---:|
| complex64 | 2.61e-4 | 3.87e-7 |
| complex128 | 6.84e-13 | 8.32e-16 |
| float32 | 2.05e-4 | 2.60e-7 |
| float64 | 4.76e-13 | 8.41e-16 |
| longdouble | 2.31e-16 | 4.64e-19 |
| clongdouble | 3.14e-16 | 4.27e-19 |

These were all below the conservative, dtype-scaled comparison bounds. The
large float32 absolute values are expected roundoff growth at large lengths;
the relative errors remain well below the applied bound.

## Files and retained negative evidence

* [`batch_compare.py`](batch_compare.py) — complete c2c timing driver.
* [`api_matrix.py`](api_matrix.py) — direct and secondary public API timing
  driver.
* [`correctness_matrix.py`](correctness_matrix.py) — isolated old/final
  numerical comparison.
* [`ducc_policy_rss.py`](ducc_policy_rss.py) — fresh-process timing/RSS probe
  for the DUCC policy candidates.
* [`DUCC_PROPOSAL.md`](DUCC_PROPOSAL.md) — upstreamable automatic and explicit
  DUCC-side designs, decision, and validation gates.
* [`DIAGNOSTICS.md`](DIAGNOSTICS.md) — temporary DUCC path evidence; no
  diagnostic logging remains in production source.
* [`results/`](results/) — raw CSVs, including pre-final candidates, matched
  V2/V3 builds, DUCC policy comparisons/RSS, direct row-loop and CPU-dispatch
  prototypes, typed-factor comparisons, and correctness passes.
* [`experimental_patches/`](experimental_patches/) — the local DUCC patches;
  neither is vendored in the current NumPy commit. The direct row-loop patch
  is also retained as a rejected control.
* [`experimental_patches/README.md`](experimental_patches/README.md) — why
  rejected row-loop, cache, and dispatch alternatives were not included.

## Remaining limitations and follow-ups

1. Large complex128 batches remain slower than the parent pocketfft backend in
   this DUCC migration. The final wrapper does not worsen c222's generalized
   path and avoids a non-universal direct row-loop heuristic. A future DUCC
   algorithm or portable batch implementation should target this explicitly.
2. A baseline-only V3 build materially improves DUCC, but the isolated
   multi-target prototype did not reproduce that gain and adds build/link
   complexity. It is not part of the production diff.
3. A thread-safe global or thread-local plan cache is not justified by this
   task and would require separate fork, memory, and concurrency validation.
4. The DUCC-side batch-policy proposal needs upstream review and non-Ryzen
   validation before it can replace the current vendored DUCC source.

No other unresolved issue is being hidden as a benchmark exception. The
production diff contains only the selected one-dimensional paths and the
typed normalization factor; all diagnostics and experimental alternatives are
outside the production source.
