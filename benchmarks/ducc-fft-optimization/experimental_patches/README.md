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

Two variants were measured across the c2c length/batch matrix:

* `c2c_batch_compare.csv`: low-level row loop with vectorized execution
  enabled;
* `c2c_batch_compare_novec.csv`: the same design with low-level vectorization
  disabled.

The vectorized row-loop candidate was approximately parity with unchanged
c222 overall but was slower in several batched clusters. The scalar variant
was often 5--20% faster than c222 for large complex128 batches, but had
reproducible 2.4--2.5x regressions for `ifft(complex64, n=4097,
batch=4..256)`. It also remained slower than the old pocketfft backend in
many large cases. No simple batch/length crossover was stable across dtype and
operation, so the final direct path leaves ordinary contiguous batches on
DUCC's generalized implementation.

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

No multi-axis FFT experiment is represented by these alternatives.
