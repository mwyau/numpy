# DUCC one-dimensional path diagnostics

These diagnostics came from a temporary cache-off build with
`DUCC0_FFT_DIAGNOSTICS` added to the DUCC headers. They were never used for
timing and no diagnostic code remains in the final source.

## Current wrapper, cache off

The public c2c calls below were passed to DUCC as a two-dimensional view with
axis 1. `simd` is DUCC's effective complex SIMD width, not the hardware float
lane count.

| call | path | ndim | axis | simd | n_outer | n_simul | n_bunch | wss(1) | wss(vlen) | temp bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c2c n=4096 | general | 2 | 1 | 2 | 1 | 2 | 2 | 197,120 | 328,192 | 132,096 |
| c2c n=4097 | general | 2 | 1 | 2 | 1 | 2 | 2 | 212,144 | 343,248 | 146,848 |
| c2c n=8192 | general | 2 | 1 | 2 | 1 | 1 | 1 | 393,728 | 655,872 | 131,584 |
| c2c complex64, 8x4096 | general | 2 | 1 | 2 | 8 | 2 | 2 | 197,120 | 328,192 | 264,192 |
| c2c complex128, 8x16384 | general | 2 | 1 | 2 | 8 | 1 | 1 | 805,376 | 1,329,664 | 281,088 |

Real transforms use separate paths:

| call | path | ndim | simd | plan buffer | temp bytes |
|---|---|---:|---:|---:|---:|
| rfft n=4096 | r2c | 2 | 2 | 4,096 | 65,792 |
| irfft complex64 n=4096 | c2r | 2 | 4 | 4,096 | 32,896 |
| irfft complex128 n=4096 | c2r | 2 | 2 | 4,096 | 65,792 |

## True 1-D c2c path

For a contiguous direct call with `n_outer == 1`, the experimental wrapper
reported:

```text
path=simple sizeof_T=16 sizeof_T0=8 simd=2 len=4096 ndim=1 axis=0
in_stride=1 out_stride=1 plan_buf=4128 n_outer=1 temp=0
```

The real path remains `r2c`/`c2r`; it does not use DUCC's c2c
`exec_simple` branch.

## SIMD and type resolution

The default FFT translation unit is compiled at the X86_V2 baseline. DUCC's
effective widths in that build are:

* directly typed float loop: width 4;
* directly typed double loop: width 2.

The public `np.fft.fft(complex64)` call at c222 passes integer `fct=1` and can
therefore resolve to the double loop (`sizeof_T=16`, `sizeof_T0=8`). A direct
float32 factor selects the float loop (`sizeof_T=8`, `sizeof_T0=4`, width 4).

At n=4096, the directly typed float path reports `wss(1)=98,560`,
`wss(vlen)=295,168`, `n_simul=4`, and `n_bunch=4`; at n=8192 it drops to
`n_simul=1`/`n_bunch=1`. The public double path already uses 2 at n=4096 and
drops to 1 at n=8192. There is no n=4096 vectorization drop.

## Plan construction and wrapper decisions

The diagnostic wrapper records that public unusual output/alignment/overlap
cases are often normalized by legacy ufunc machinery before reaching the
backend. Zero-padding does enter the staged wrapper. With cache-off, a batch
of eight padded c2c rows produces eight one-dimensional `simple` records and
therefore eight plan constructions in c222. The local plan candidate reduces
that to one per gufunc invocation.

The conservative range-overlap check is intentionally allowed to classify
strided views as overlapping. Those cases remain staged to preserve NumPy's
snapshot semantics.

## Cache safety

The cache-on diagnostic build removed only `DUCC0_NO_FFT_CACHE` and retained
`DUCC0_NO_LOWLEVEL_THREADING`. Under the latter define DUCC's `Mutex` and
`LockGuard` are no-ops, while the FFT cache mutates static LRU state. Cache-on
measurements are consequently not thread-safe production results and are not
used to justify the final build.

## Symbol inspection

The separate V2 and V3 final shared objects were inspected with `nm -C
--defined-only` and `readelf -Ws`. DUCC header instantiations observed in the
standalone objects are local text/data symbols (`t`/`b`); no global or weak
`ducc0::` definitions were found. This is useful evidence for standalone
target objects but is not a complete cross-compiler proof for a new NumPy
multi-target link topology, so runtime dispatch was not added.
