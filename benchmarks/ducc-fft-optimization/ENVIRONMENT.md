# Benchmark environment and build identity

Captured for the local c222-rooted investigation on 2026-09-15.

## Source identity

| item | value |
|---|---|
| required starting branch | `enh/duccfft` |
| required starting commit | `c222665be45cbd361f2b0e96f6ff81f12b390ebd` |
| starting subject | `ENH: migrate numpy.fft to duccfft` |
| parent/pocketfft comparison | `1819fd4beec3d15cc24cd6237b7d55e3a2debda4` |
| investigation branch | `investigation/ducc-fft-optimization` |
| vendored DUCC | 0.41.1, `64f42ba531f609ba7029c82207a063b17f9d5275` |
| DUCC source | `https://gitlab.mpcdf.mpg.de/mtr/ducc.git` |

The final source changes are limited to `numpy/fft/_ducc.py`,
`numpy/fft/_duccfft_umath.cpp`, and the benchmark/report directory. No
multi-axis FFT source or benchmark is part of this work.

## Host and toolchain

| item | value |
|---|---|
| host CPU | AMD Ryzen 9 5950X 16-Core Processor |
| logical CPUs | 32 |
| memory | 32 GiB |
| ISA observed | x86-64, AVX2, FMA |
| OS | Ubuntu 26.04.1 LTS |
| kernel | `7.0.14-15-pve` |
| Python | 3.14.7 |
| Python build compiler | Clang 22.1.3 |
| C/C++ compiler | GCC 15.2.0 |
| linker | GNU ld 2.46 |
| BLAS | OpenBLAS 0.3.32 |
| NumPy build version | `2.6.0.dev0+git20260915.5787f7e` |
| Meson | vendored 1.11.1 (`vendored-meson/meson/meson.py`) |
| Ninja | 1.13.2 |

Long-double representation on this host is 16-byte Intel extended precision;
`clongdouble` is 32 bytes.

## Build configurations

The default production-shaped build uses:

```text
buildtype = debugoptimized
prefix = /usr
cpu-baseline = min       # resolves to X86_V2 on this host
cpu-dispatch = max
DUCC0_NO_FFT_CACHE
DUCC0_NO_LOWLEVEL_THREADING
```

The FFT extension is a normal extension-module translation unit, so its DUCC
templates are compiled at the selected baseline rather than as a NumPy
multi-target source. Separate diagnostic builds used:

```text
X86_V2: -Dcpu-baseline=X86_V2 -Dcpu-dispatch=none
X86_V3: -Dcpu-baseline=X86_V3 -Dcpu-dispatch=none
```

The V2/V3 parent and final stages were built from fresh worktrees with the
same buildtype and compiler; only the source revision and baseline differed.

## Benchmark process policy

Every benchmark worker receives:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Workers run with `/home/albert/numpy/.venv/bin/python -S`, from `/tmp`, and
with exactly one staged site-packages path in `PYTHONPATH`. This avoids the
editable NumPy finder in the development virtual environment. One worker is
run per candidate so allocator state and plan/cache state do not cross
comparisons. Each case has an untimed warm-up, an adaptive inner loop, and
multiple timed samples. Median, best, and median absolute deviation are
recorded.

## Stage roots used by the raw results

| label | staged site-packages |
|---|---|
| `old` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-old/usr/local/lib/python3.14/site-packages` |
| `c222` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-ducc/usr/local/lib/python3.14/site-packages` |
| `final` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-staged/usr/lib/python3.14/site-packages` |
| `current_final` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-fct/usr/lib/python3.14/site-packages` |
| `old_v2` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-old-v2/usr/local/lib/python3.14/site-packages` |
| `final_v2` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-final-v2/usr/lib/python3.14/site-packages` |
| `old_v3` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-old-v3/usr/local/lib/python3.14/site-packages` |
| `final_v3` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-final-v3/usr/lib/python3.14/site-packages` |

The `current_final`, V2, and V3 final stages contain the typed-factor Python
wrapper. `final` is the retained staged-only pre-factor comparison; it is
kept for historical raw results and is not the current production candidate.

## DUCC-side experiment stages

The separate experiment worktree was `/home/albert/numpy-force-nsimul`,
detached at the preserved NumPy production commit `5787f7e27f370a50278401807adaec56752608cc`.
Its vendored DUCC source is the exact 0.41.1 tree at
`64f42ba531f609ba7029c82207a063b17f9d5275`, as recorded by
`tools/vendoring/vendor_duccfft.sh`.

| label | staged site-packages | build policy |
|---|---|---|
| `auto_v2` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v2/usr/local/lib/python3.14/site-packages` | automatic heuristic, `min` resolving to X86_V2 |
| `auto_v3` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v3/usr/local/lib/python3.14/site-packages` | automatic heuristic, X86_V3 baseline |
| `policy_v2` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api/usr/local/lib/python3.14/site-packages` | explicit internal policy, X86_V2-resolved build |
| `policy_v3` | `/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api-v3/usr/local/lib/python3.14/site-packages` | explicit internal policy, X86_V3 baseline |

The automatic experiment changes only the DUCC-side c2c heuristic for
contiguous no-stride double batches when the compiled SIMD width is at most
two lanes. The explicit experiment adds a named internal
`batch_policy::vectorize_contiguous` entry point and the temporary NumPy
adapter calls it only for batched complex128 c2c. Neither experiment changes
`_ducc_nd_umath`, native N-D dispatch, or any N-D benchmark. The experiment
builds retain `DUCC0_NO_FFT_CACHE` and `DUCC0_NO_LOWLEVEL_THREADING`.
