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

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=/absolute/path/to/stage/usr/lib/python3.14/site-packages:/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S -m pytest --pyargs numpy.fft.tests \
    -q --disable-warnings

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=/home/albert/numpy/.venv/lib/python3.14/site-packages \
    /home/albert/numpy/.venv/bin/python -S \
    benchmarks/ducc-fft-optimization/correctness_matrix.py \
    --old-stage /home/albert/numpy-fft-ducc-benchmark-20260914/stage-old/usr/local/lib/python3.14/site-packages \
    --final-stage /home/albert/numpy-fft-ducc-benchmark-20260914/stage-candidate-staged/usr/lib/python3.14/site-packages \
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
