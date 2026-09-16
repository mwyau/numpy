# Experimental alternatives

These alternatives were built only in separate worktrees/stages. Their
production code is not part of the final source. Raw timings are retained in
`../results/`.

## True 1-D for every transform type — retained only for c2c

Hypothesis: presenting every one-row call as a true one-dimensional DUCC view
would avoid generalized-path overhead.

Result: c2c single transforms improved materially, including the `n=4096`
cliff. The same generic change for r2c/c2r sometimes regressed rfft and did
not reach the c2c simple implementation. Final code enables the one-
dimensional shape only for direct c2c calls.

## Direct row loop with a reusable c2c plan — rejected for ordinary batches

The first two local-plan variants were measured as staged/local-plan controls:

* `c2c_batch_compare.csv`: low-level row loop with vectorized execution
  enabled;
* `c2c_batch_compare_novec.csv`: the same design with low-level vectorization
  disabled.

They are not a clean ordinary-direct row-loop comparison: the modified
wrapper entered the local fallback before the normal direct path. The scalar
control nevertheless exposed reproducible 2.4--2.5x regressions for
`ifft(complex64, n=4097, batch=4..256)` and neither control recovered the old
pocketfft behavior broadly.

The clean ordinary-direct experiment is the separate
[`ducc-direct-rowloop.patch`](ducc-direct-rowloop.patch). It was applied to
the current production commit and only routes equal-length, no-stride,
contiguous batches through one local plan and a row loop. Its full raw result
is [`c2c_batch_compare_rowloop.csv`](../results/c2c_batch_compare_rowloop.csv).
The row loop improved some complex128 forward batches, but regressed
complex64 inverse batches and remained slower than pocketfft in the large
focus. The focused RSS result is
[`rss_rowloop_complex128.csv`](../results/rss_rowloop_complex128.csv). No
portable batch/length crossover was stable across dtype and operation, so the
final direct path leaves ordinary contiguous batches on DUCC's generalized
implementation.

The staged-only version is different: there is no competing generalized
batch decision after the direct path declines, and its one-plan benefit is
clear. That version is retained in the final source as the fallback path.

## Staged local plan and scratch reuse — retained

The local c2c plan candidate constructs one `pocketfft_c<T>` and one scratch
buffer per gufunc invocation and reuses them across non-overlapping rows. At
`n=4096`, its warm improvement over c222 was approximately 23% for batch 8
and 28% for batch 64. Overlapping views remain on the staged snapshot path.

## DUCC cache-on — rejected as unsafe

Removing `DUCC0_NO_FFT_CACHE` improved repeated same-shape calls, but the
tested build retained `DUCC0_NO_LOWLEVEL_THREADING`. DUCC's cache locks are
then no-ops while static LRU state is mutated. These results are diagnostic
only; no cache-on/no-low-level-threading code is shippable.

## DUCC blocking heuristic — unchanged

Diagnostics show the c2c `n=4096` regression while DUCC still has
`n_simul=2`/`n_bunch=2`; the `n_simul=1` transition occurs at `n=8192` in
the public double path. The hypothesized 4096/512-KiB causal threshold is
therefore disproved. No vendored DUCC heuristic or host-specific cache-size
constant was changed.

## DUCC-side contiguous-batch proposals — local only

The follow-up option-3 experiment is recorded in
[`DUCC_PROPOSAL.md`](../DUCC_PROPOSAL.md). It has two patch files:

* [`ducc-auto-heuristic.patch`](ducc-auto-heuristic.patch) adds an internal
  `batch_policy` abstraction and makes the automatic policy use
  `n_simul=n_bunch=vlen` only for contiguous c2c double batches when
  `vlen<=2`.
* [`ducc-explicit-batch-policy.patch`](ducc-explicit-batch-policy.patch) adds
  the same DUCC-side policy plus a temporary adapter call to the named
  `batch_policy::vectorize_contiguous` entry point for batched complex128
  c2c. It does not expose `n_simul` or any policy parameter through Python.

The automatic candidate improved the large V2 complex128 c2c focus by about
1.5x geometric mean while leaving V2/V3 complex64 near parity and leaving
real transforms on their existing paths. The explicit policy had no stable
complex64 or real regression after compile-time dtype scoping, but its V3
complex128 policy path added roughly 5.7--7.2 MiB peak RSS at `n=65536`.
Neither proposal was vendored into the production NumPy commit: the evidence
is from one Ryzen host and two x86 ISA builds, and the automatic default
change still needs DUCC-level cross-architecture/cache-size validation and
upstream review.

## NumPy V2/V3 dispatch — measured, not shipped

Fresh baseline-only builds show a broad V3 benefit for the corrected DUCC
extension: V2-to-V3 previous/candidate timing is 1.126x by geometric mean.
However, the current extension combines module initialization, gufunc
registration, exception handling, and DUCC templates in one translation
unit. A production `mod_features.multi_targets()` split needs target-specific
entry points and portability/linker validation. Standalone `nm`/`readelf`
inspection found local DUCC template symbols in the V2/V3 objects, but that
alone is not enough to approve a new multi-target topology. The final diff
therefore keeps the normal baseline build and records V3 as follow-up work.

An isolated multi-target prototype was also built in
`/home/albert/numpy-cpudispatch`, with the existing c2c entry points as the
only dispatched functions. It passed the V2/V3 build, runtime fallback, symbol,
correctness, and FFT-test checks, but the repeated large-focus raw result
[`c2c_batch_compare_cpudispatch_focus.csv`](../results/c2c_batch_compare_cpudispatch_focus.csv)
showed `current/dispatch_v3 = 0.954` median / `0.965` geometric mean, so the
prototype was slower than the shipped stage. Its full matrix and RSS probe are
[`c2c_batch_compare_cpudispatch.csv`](../results/c2c_batch_compare_cpudispatch.csv)
and [`rss_cpudispatch_complex128.csv`](../results/rss_cpudispatch_complex128.csv).
It remains evidence only and is not a production or DUCC-side patch.

No multi-axis FFT experiment is represented by these alternatives.
