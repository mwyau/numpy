# DUCC-side contiguous-batch proposal

Status: local experiment only. This document proposes a DUCC-side change for
the generalized contiguous-batch c2c regression; it is not part of the
shipped NumPy source.

## Decision

Keep the current NumPy production commit unchanged at
`5787f7e27f370a50278401807adaec56752608cc`. Do not vendor either experimental
patch into that commit and do not expose a raw `n_simul` control to Python.

The strongest upstream direction is the automatic DUCC heuristic in
[`experimental_patches/ducc-auto-heuristic.patch`](experimental_patches/ducc-auto-heuristic.patch):
for contiguous c2c double batches, use one SIMD-width group when the compiled
width is at most two lanes. Keep the explicit internal policy in
[`experimental_patches/ducc-explicit-batch-policy.patch`](experimental_patches/ducc-explicit-batch-policy.patch)
as the fallback design if DUCC maintainers do not want to change the default
heuristic.

The local evidence is promising but not sufficient to vendor. Both ISA builds
were run on one Ryzen host, so they do not establish behavior on non-x86
hardware or different cache geometries. The explicit policy also has a clear
V3 peak-RSS cost without a V3 speed benefit. The right handoff is therefore a
reviewable DUCC proposal, with the current NumPy implementation retained in
the production branch.

## Baseline and scope

* NumPy baseline: `5787f7e27f370a50278401807adaec56752608cc`, whose parent is
  c222 `c222665be45cbd361f2b0e96f6ff81f12b390ebd`.
* Vendored DUCC: 0.41.1 at
  `64f42ba531f609ba7029c82207a063b17f9d5275`, rooted from the vendoring
  script rather than a NumPy-side replacement.
* Experiment worktree: `/home/albert/numpy-force-nsimul`, detached at the
  NumPy baseline.
* Only the existing one-dimensional NumPy FFT wrapper was exercised. No
  `_ducc_nd_umath` source, native N-D dispatch, or N-D benchmark was changed.
* All candidates use `DUCC0_NO_FFT_CACHE` and
  `DUCC0_NO_LOWLEVEL_THREADING`; the wrapper passes `nthreads=1`.
* Workers run in fresh processes with `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` set to
  one.

The two candidate designs are deliberately separate from the current NumPy
implementation. The production commit still contains only the previously
selected one-dimensional shape, staged-plan reuse, overlap protection, and
typed normalization-factor changes.

## Design A: improve DUCC automatic selection

### Proposed change

The patch adds an internal `batch_policy` enum and threads it through DUCC's
existing c2c implementation. The default remains named `automatic`, but the
automatic branch adds this narrow rule after DUCC's existing working-set
calculation:

```text
if execution is c2c and scalar type is double and both views are no-stride:
    if SIMD width <= 2 lanes:
        n_simul = SIMD width
        n_bunch = SIMD width
```

The rule is expressed in terms of the compiled DUCC SIMD width, not a host
name, model string, or NumPy size threshold. `vlen=1` naturally leaves the
scalar path unchanged. The `ExecC2C` type guard keeps the automatic change out
of DUCC's separate real-transform execution paths.

### Measured result

Ratios below are `default / automatic`; values above one mean the automatic
candidate is faster. The focused matrix contains both fft and ifft, lengths
8,192, 16,384, 32,768, and 65,536, and batches 8, 32, 64, and 256. The
second run is retained as
[`c2c_auto_v2_v3.csv`](results/c2c_auto_v2_v3.csv).

| build / dtype | cases | median | geometric mean | minimum | maximum |
|---|---:|---:|---:|---:|---:|
| V2 / complex64 | 32 | 1.004 | 1.005 | 0.985 | 1.058 |
| V2 / complex128 | 32 | 1.668 | 1.515 | 1.035 | 1.832 |
| V3 / complex64 | 32 | 0.999 | 0.999 | 0.966 | 1.033 |
| V3 / complex128 | 32 | 0.999 | 1.006 | 0.975 | 1.067 |

This is the useful shape of the result: the narrow-width V2 complex128
cluster improves consistently, while V2/V3 complex64 remains at parity and
the wider V3 build does not pay the forced-vectorization cost.

The full public API control run is
[`api_auto_policy_v2_v3.csv`](results/api_auto_policy_v2_v3.csv). Because the
explicit policy is restricted to complex128 c2c in the adapter, comparing the
automatic and explicit stages is a same-build control for unaffected paths:

| same-build comparison (`automatic / explicit`) | cases | V2 median / gmean | V3 median / gmean |
|---|---:|---:|---:|
| c2c complex64 | 32 | 1.012 / 1.011 | 1.001 / 1.000 |
| real API (`rfft`/`irfft`) | 64 | 0.997 / 1.002 | 1.002 / 1.004 |
| c2c complex128 | 32 | 1.003 / 1.003 | 0.988 / 0.992 |

The small deviations in the unaffected rows are measurement noise; neither
the automatic rule nor the explicit policy changes the real-transform source
path. The V3 c2c complex128 row shows why the explicit policy should not be
forced unconditionally on a wider SIMD build.

## Design B: explicit internal DUCC policy

### Proposed change

The same DUCC header change exposes a C++-internal named entry point:

```text
c2c_with_batch_policy(..., batch_policy::vectorize_contiguous)
```

The temporary NumPy adapter calls it only when the ufunc loop is the double
loop and `n_outer > 1`; complex64 calls the existing `ducc0::c2c` entry point
at compile time. No Python argument, environment variable, or raw
`n_simul` value is added. The policy is a semantic batch choice owned by
DUCC, and the adapter remains limited to the existing one-dimensional gufunc.

### Full c2c matrix

[`c2c_batch_compare_ducc_policy.csv`](results/c2c_batch_compare_ducc_policy.csv)
contains old pocketfft, c222, current final NumPy, and the explicit DUCC
policy candidate over all 14 required lengths, all 8 required batches, both
c2c directions, and both complex dtypes: 1,792 successful rows. Ratios are
`previous / candidate`.

| comparison | cases | median | geometric mean | large focus: median / gmean |
|---|---:|---:|---:|---:|
| c222 / current final | 448 | 1.033 | 1.148 | 1.033 / 1.288 |
| current final / DUCC policy | 448 | 1.001 | 1.053 | 1.028 / 1.235 |
| c222 / DUCC policy | 448 | 1.056 | 1.209 | 1.640 / 1.591 |
| old pocketfft / DUCC policy | 448 | 0.807 | 0.883 | 0.786 / 0.901 |

The large focus is `n >= 8192` with batches 8, 32, 64, and 256. The policy
does not make DUCC match the former pocketfft algorithm for every large
complex128 batch; that remaining gap is separate from the NumPy wrapper
change. The policy does, however, improve the generalized DUCC path relative
to c222 while retaining current-final parity for complex64.

### Resource impact

The RSS probe uses one fresh process per candidate/case and measures Linux
`ru_maxrss`. The table gives `explicit - automatic` peak RSS in KiB for
`fft(complex128, n=65536)`:

| batch | 8 | 32 | 64 | 256 |
|---|---:|---:|---:|---:|
| V2 delta KiB | +92 | +64 | +300 | +332 |
| V3 delta KiB | +5,732 | +5,788 | +6,776 | +7,196 |

The corresponding complex64 deltas are within approximately +/-0.8 MiB in
both builds, with no systematic increase. Raw samples are in
[`rss_ducc_policy_complex128.csv`](results/rss_ducc_policy_complex128.csv) and
[`rss_ducc_policy_complex64.csv`](results/rss_ducc_policy_complex64.csv).
The V3 complex128 increase is consistent with forcing four simultaneous
transforms and is a reason to keep the explicit policy opt-in rather than
make it the default everywhere.

## Correctness and validation gates

The final scoped policy stage was checked against old pocketfft by
[`correctness_old_ducc_policy.csv`](results/correctness_old_ducc_policy.csv):
2,034/2,034 cases have `status=ok`, including all requested one-dimensional
lengths, representative batches, layouts, padding/truncation, norms, output
variants, overlap cases, and long-double variants. Maximum errors match the
already accepted final comparison: complex64 absolute/relative
`2.61e-4 / 3.87e-7`, complex128 `6.84e-13 / 8.32e-16`, float32
`2.05e-4 / 2.60e-7`, float64 `4.76e-13 / 8.41e-16`, longdouble
`2.31e-16 / 4.64e-19`, and clongdouble `3.14e-16 / 4.27e-19`.

The NumPy FFT test module passed `172/172` in both the V2 and V3 policy
stages. The automatic stages were built with the same source and cache-off
flags; its c2c numerical path is equivalent to the policy path on V2 and
remains the existing automatic path on V3.

The compile commands were inspected for both policy stages and contain:

```text
-DDUCC0_NO_FFT_CACHE
-DDUCC0_NO_LOWLEVEL_THREADING
```

All timing and correctness workers pass `nthreads=1` through the wrapper and
set external BLAS/OpenMP-style thread counts to one. Thus the reported result
does not rely on hidden DUCC worker parallelism or a plan cache. The exact
commands, stage paths, and raw artifact names are in
[`COMMANDS.md`](COMMANDS.md) and [`ENVIRONMENT.md`](ENVIRONMENT.md).

## Upstream recommendation and non-vendoring rationale

1. Send the automatic width-gated heuristic to DUCC review as the primary
   proposal, with a DUCC-level test matrix covering x86 narrow/wide SIMD and
   at least one non-x86 SIMD implementation. The test should verify c2c
   complex64, real transforms, no-stride and strided layouts, and peak
   scratch sizing.
2. Keep the named internal `batch_policy` hook as the fallback if maintainers
   prefer an explicit caller policy. The hook must remain internal and must
   not expose raw scheduling parameters to NumPy users.
3. Do not vendor either patch into NumPy now. The local automatic result
   passes the available V2/V3 and unaffected-dtype gates, but the evidence is
   one Ryzen host and cannot prove portability. The explicit path has no V3
   speed case and adds measurable V3 RSS. Upstream acceptance and broader
   DUCC validation are required before changing the vendored 0.41.1 source.

The current shipped NumPy changes therefore remain the production choice. The
DUCC-side solution is preserved as a local patch/proposal with raw results,
not silently folded into a NumPy-only heuristic.
