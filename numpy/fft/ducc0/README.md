The `ducc0` directory is populated by `tools/vendoring/vendor_duccfft.sh`.
The upstream project is [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc.git).
The script pins the `ducc0` branch snapshot at commit
`d8f5ffe71945aa5614cf61731dcb5dcc7789680f` and copies the explicit FFT source
manifest in that script. DUCC sources are dual-licensed; NumPy distributes
these copied sources under the BSD-3-Clause option.

The odd-sized DCT-IV helper is derived from FFTW3's `apply_re11()` and is
included under the 3-clause BSD permission recorded in `LICENSE.md`, with
Matteo Frigo and Steven G. Johnson attributed in the source and license file.

The vendoring script also applies one local change to `infra/threading.h`:
when `DUCC0_NO_LOWLEVEL_THREADING` and `DUCC0_FFT_CACHE_MUTEX` are defined,
DUCC's no-op lock types are replaced with `std::mutex` wrappers. NumPy disables
DUCC's worker pool but retains its FFT plan cache, so this mutex protects cache
access while FFT calls can run concurrently without the GIL. The cache mutex
macro is enabled by `numpy/fft/meson.build`; the worker pool remains disabled.

`infra/threading.cc` also contains an optional, disabled helper adapted from
TensorFlow. Its attribution and Apache-2.0 license are included in
`LICENSE.md`.
