# Reproduction commands

Commands use the absolute paths and one-thread environment in
[`ENVIRONMENT.md`](ENVIRONMENT.md). They run from `/home/albert/numpy` unless
otherwise noted.

## Identity and default build

```bash
git show -s --format='%H%n%P%n%s' c222665be45cbd361f2b0e96f6ff81f12b390ebd
git show -s --format='%H%n%P%n%s' 1819fd4beec3d15cc24cd6237b7d55e3a2debda4
git submodule update --init --recursive
rg -n 'DUCC|0\.41\.1|64f42ba531f609ba7029c82207a063b17f9d5275' \
    tools/vendoring/vendor_duccfft.sh

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    setup --wipe build . -Dbuildtype=debugoptimized \
    -Dcpu-baseline=min -Dcpu-dispatch=max

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/ninja -C build -j16

/home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    install -C build --destdir /absolute/path/to/stage --no-rebuild
```

The FFT compile line was checked with:

```bash
sed -n '/_duccfft_umath.cpp.o:/,+4p' build/build.ninja
```

It contains `-DDUCC0_NO_FFT_CACHE` and
`-DDUCC0_NO_LOWLEVEL_THREADING`.

## Explicit ISA builds

```bash
/home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    setup --wipe build-v2 . -Dbuildtype=debugoptimized \
    -Dcpu-baseline=X86_V2 -Dcpu-dispatch=none
/home/albert/numpy/.venv/bin/ninja -C build-v2 -j16

/home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    setup --wipe build-v3 . -Dbuildtype=debugoptimized \
    -Dcpu-baseline=X86_V3 -Dcpu-dispatch=none
/home/albert/numpy/.venv/bin/ninja -C build-v3 -j16
```

## FFT tests and correctness

Use the installed test directory explicitly. This avoids the editable finder
in the development environment and keeps pytest's import path tied to the
staged candidate.

```bash
stage=/absolute/path/to/stage/usr/lib/python3.14/site-packages
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH="$stage:/home/albert/numpy/.venv/lib/python3.14/site-packages" \
    /home/albert/numpy/.venv/bin/python -S -m pytest "$stage/numpy/fft/tests" \
    -q --disable-warnings

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S \
    benchmarks/ducc-fft-optimization/correctness_matrix.py \
    --old-stage /home/albert/numpy-fft-ducc-benchmark-20260914/stage-old/usr/local/lib/python3.14/site-packages \
    --final-stage /home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-fct/usr/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/correctness_old_final_fct.csv
```

## Timing matrices

The c2c command uses `batch_compare.py` with `c222`, the staged-only
implementation, and the typed-factor implementation. The recorded final
typed-factor c2c run is retained as `results/c2c_batch_compare_fct.csv`:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S \
    benchmarks/ducc-fft-optimization/batch_compare.py \
    --candidate c222=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-ducc/usr/local/lib/python3.14/site-packages \
    --candidate final=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-staged/usr/lib/python3.14/site-packages \
    --candidate final_fct=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-fct/usr/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/c2c_batch_compare_repro.csv
```

The full API command uses `api_matrix.py` with `c222`, `final`, and
`final_fct` stages and writes `results/api_matrix_fct.csv`; the corresponding
`old`/`final_fct` comparison writes `results/api_matrix_old_final_fct.csv`.
Adding `--secondary-only` to the three-candidate command writes
`results/api_secondary_c222_final_fct.csv`. The V2/V3 commands use the same
driver with sizes `1024 2048 3072 3584 4095 4096 4097 4608 8192 16384 32768
65536`, batches `1 2 8 64 256`, and the corresponding `old_v2/final_v2` or
`old_v3/final_v3` stage paths.

### Direct ordinary-batch row-loop control

This control starts at the current production commit
`140acd6d898d681ae3e41150e9883bb22f3edfa2`, not at c222. It is a NumPy-side
experiment and does not modify vendored DUCC:

```bash
git worktree add --detach /home/albert/numpy-rowloop \
    140acd6d898d681ae3e41150e9883bb22f3edfa2
git -C /home/albert/numpy-rowloop submodule update --init --recursive
git -C /home/albert/numpy-rowloop apply \
    /home/albert/numpy/benchmarks/ducc-fft-optimization/experimental_patches/ducc-direct-rowloop.patch

cd /home/albert/numpy-rowloop
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    setup --wipe build-rowloop . -Dbuildtype=debugoptimized \
    -Dcpu-baseline=min -Dcpu-dispatch=max
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/ninja -C build-rowloop -j16
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    install -C build-rowloop --destdir \
    /home/albert/numpy-fft-ducc-benchmark-20260914/stage-rowloop --no-rebuild

cd /home/albert/numpy
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S \
    benchmarks/ducc-fft-optimization/batch_compare.py \
    --candidate old=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-old/usr/local/lib/python3.14/site-packages \
    --candidate c222=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-ducc/usr/local/lib/python3.14/site-packages \
    --candidate current_final=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-fct/usr/lib/python3.14/site-packages \
    --candidate rowloop=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-rowloop/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/c2c_batch_compare_rowloop.csv
```

## Static and script checks

```bash
nm -C --defined-only /absolute/path/to/_duccfft_umath*.so
readelf -Ws /absolute/path/to/_duccfft_umath*.so
objdump -d /absolute/path/to/_duccfft_umath*.so

env NPY_DISABLE_CPU_FEATURES=AVX2 PYTHONPATH=/absolute/v3/stage \
    /home/albert/numpy/.venv/bin/python -S -c \
    'import numpy as np; print(np.fft.fft(np.ones(16, dtype=np.complex64)).shape)'

env PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S -m py_compile \
    benchmarks/ducc-fft-optimization/*.py
```

The isolated CPU-dispatch prototype was built in
`/home/albert/numpy-cpudispatch` with the existing c2c wrapper as the only
dispatched source. Its V2 baseline and V2-plus-V3-dispatch builds used:

```bash
cd /home/albert/numpy-cpudispatch
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    setup --wipe build-cpudispatch-v2 . -Dbuildtype=debugoptimized \
    -Dcpu-baseline=X86_V2 -Dcpu-dispatch=none
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    compile -C build-cpudispatch-v2 -j16
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    install -C build-cpudispatch-v2 --destdir \
    /home/albert/numpy-fft-ducc-benchmark-20260914/stage-cpudispatch-v2 --no-rebuild

env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    setup --wipe build-cpudispatch-v3 . -Dbuildtype=debugoptimized \
    -Dcpu-baseline=X86_V2 -Dcpu-dispatch=X86_V3
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    compile -C build-cpudispatch-v3 -j16
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    install -C build-cpudispatch-v3 --destdir \
    /home/albert/numpy-fft-ducc-benchmark-20260914/stage-cpudispatch-v3 --no-rebuild
```

The full c2c comparison and repeat large-focus comparison were:

```bash
cd /home/albert/numpy
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S \
    benchmarks/ducc-fft-optimization/batch_compare.py \
    --candidate current_final=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-fct/usr/lib/python3.14/site-packages \
    --candidate cpudispatch_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-cpudispatch-v2/usr/local/lib/python3.14/site-packages \
    --candidate cpudispatch_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-cpudispatch-v3/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/c2c_batch_compare_cpudispatch.csv

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S \
    benchmarks/ducc-fft-optimization/batch_compare.py \
    --candidate current_final=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-fct/usr/lib/python3.14/site-packages \
    --candidate cpudispatch_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-cpudispatch-v2/usr/local/lib/python3.14/site-packages \
    --candidate cpudispatch_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-cpudispatch-v3/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/c2c_batch_compare_cpudispatch_focus.csv \
    --sizes 8192 16384 32768 65536 --batches 8 32 64 256

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S \
    benchmarks/ducc-fft-optimization/ducc_policy_rss.py \
    --candidate current_final=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-fct/usr/lib/python3.14/site-packages \
    --candidate cpudispatch_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-cpudispatch-v2/usr/local/lib/python3.14/site-packages \
    --candidate cpudispatch_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-cpudispatch-v3/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/rss_cpudispatch_complex128.csv \
    --operation fft --dtype complex128 --sizes 65536 --batches 8 32 64 256 \
    --samples 3 --repeats 2
```

## DUCC-side batch-policy experiment

The experiment was run in the separate worktree rooted at the preserved
production commit. The vendored DUCC identity was checked with:

```bash
cd /home/albert/numpy-force-nsimul
git show -s --format='%H%n%P%n%s' HEAD
rg -n '0\.41\.1|64f42ba531f609ba7029c82207a063b17f9d5275' \
    tools/vendoring/vendor_duccfft.sh
```

The automatic and explicit candidates used the same source tree and cache-off
single-threaded build policy. Build the V2-like automatic candidate and the
V3 automatic candidate with:

```bash
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    setup --reconfigure build-auto-v2 \
    -Dbuildtype=debugoptimized -Dcpu-baseline=min -Dcpu-dispatch=max
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/ninja -C build-auto-v2 -j16
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    install -C build-auto-v2 --destdir \
    /home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v2 --no-rebuild

env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    setup --reconfigure build-auto-v3 \
    -Dbuildtype=debugoptimized -Dcpu-baseline=X86_V3 -Dcpu-dispatch=none
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/ninja -C build-auto-v3 -j16
env PATH=/home/albert/numpy/.venv/bin:$PATH \
    /home/albert/numpy/.venv/bin/python vendored-meson/meson/meson.py \
    install -C build-auto-v3 --destdir \
    /home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v3 --no-rebuild
```

For the explicit policy stage, apply
`experimental_patches/ducc-explicit-batch-policy.patch` to the preserved
source, then repeat the same build/install commands with `build-force-api`
and `build-force-api-v3`, using stage destinations `stage-force-api` and
`stage-force-api-v3`. The patch adds no Python parameter and passes only the
named internal DUCC policy for batched complex128 c2c.

The required full c2c matrix and focused V2/V3 comparisons were generated by:

```bash
/home/albert/numpy/.venv/bin/python \
    benchmarks/ducc-fft-optimization/batch_compare.py \
    --candidate old=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-old/usr/local/lib/python3.14/site-packages \
    --candidate c222=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-ducc/usr/local/lib/python3.14/site-packages \
    --candidate current_final=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-fct/usr/lib/python3.14/site-packages \
    --candidate ducc_policy_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/c2c_batch_compare_ducc_policy.csv

/home/albert/numpy/.venv/bin/python \
    benchmarks/ducc-fft-optimization/batch_compare.py \
    --candidate auto_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v2/usr/local/lib/python3.14/site-packages \
    --candidate policy_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api/usr/local/lib/python3.14/site-packages \
    --candidate auto_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v3/usr/local/lib/python3.14/site-packages \
    --candidate policy_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api-v3/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/c2c_auto_policy_v2_v3.csv \
    --sizes 8192 16384 32768 65536 --batches 8 32 64 256
```

The public API and RSS probes were run with the same candidate paths:

```bash
/home/albert/numpy/.venv/bin/python \
    benchmarks/ducc-fft-optimization/api_matrix.py \
    --candidate auto_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v2/usr/local/lib/python3.14/site-packages \
    --candidate policy_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api/usr/local/lib/python3.14/site-packages \
    --candidate auto_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v3/usr/local/lib/python3.14/site-packages \
    --candidate policy_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api-v3/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/api_auto_policy_v2_v3.csv \
    --sizes 8192 16384 32768 65536 --batches 8 32 64 256

/home/albert/numpy/.venv/bin/python \
    benchmarks/ducc-fft-optimization/ducc_policy_rss.py \
    --candidate auto_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v2/usr/local/lib/python3.14/site-packages \
    --candidate policy_v2=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api/usr/local/lib/python3.14/site-packages \
    --candidate auto_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-auto-v3/usr/local/lib/python3.14/site-packages \
    --candidate policy_v3=/home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api-v3/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/rss_ducc_policy_complex128.csv \
    --operation fft --dtype complex128 --sizes 65536 \
    --batches 8 32 64 256 --samples 3 --repeats 2
```

The matching `complex64` RSS command differs only in
`--output ...complex64.csv --dtype complex64`. Correctness used:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S \
    benchmarks/ducc-fft-optimization/correctness_matrix.py \
    --old-stage /home/albert/numpy-fft-ducc-benchmark-20260914/stage-old/usr/local/lib/python3.14/site-packages \
    --final-stage /home/albert/numpy-fft-ducc-benchmark-20260914/stage-force-api/usr/local/lib/python3.14/site-packages \
    --output benchmarks/ducc-fft-optimization/results/correctness_old_ducc_policy.csv
```
