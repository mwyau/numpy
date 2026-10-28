# Experimental native multi-axis DUCC FFT

Based on `enh/duccfft-multiarch` at `bde8f769`, without changing that branch.

## What is implemented

- A native `c2c_nd` C++ method calls `ducc0::c2c` once for all requested transform axes.
- Float32/float64 complex kernels use NumPy's V2/V3/V4 dispatch.
- `np.fft.fftn` and `np.fft.ifftn` use it for **aligned C-contiguous complex64/complex128 arrays**, at least two distinct axes, unchanged lengths, no `out=`.
- All other calls preserve the established sequential-gufunc implementation.
- `NPY_DUCCFFT_NATIVE_ND=0` switches to the original behavior before importing NumPy.

## Deliberate limitations

Not yet implemented: direct native `rfftn`/`irfftn`, native multidimensional `out=`, resizing/padding, noncontiguous arrays, repeated axes, long double. The public API signatures are unchanged.

This is an isolated prototype, **not upstream-ready** until successful builds, numerical parity tests, ABI checks, and performance measurements on multiple architectures.

## Comparison criteria

Benchmark the same `fftn`/`ifftn` shapes with `NPY_DUCCFFT_NATIVE_ND` enabled and disabled using the same build, CPU pinning, warmups, and repeated timings. Include transposed inputs (fallback), complex64/complex128, axes permutations, and shapes near L2/cache boundaries.

Keep DUCC worker threads off; leave cache and mutex settings unchanged in this branch, so timing differences isolate native-ND execution.
