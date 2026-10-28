The `ducc0` directory is populated by `tools/vendoring/vendor_duccfft.sh`.
The upstream project is [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc.git).
The script pins the `ducc0` branch snapshot at commit
`d8f5ffe71945aa5614cf61731dcb5dcc7789680f` and copies the explicit FFT source
manifest in that script. DUCC sources are dual-licensed; NumPy distributes
these copied sources under the BSD-3-Clause option.

The odd-sized DCT-IV helper is derived from FFTW3's `apply_re11()` and is
included under the 3-clause BSD permission recorded in `LICENSE.md`, with
Matteo Frigo and Steven G. Johnson attributed in the source and license file.

NumPy builds DUCC with `DUCC0_NO_LOWLEVEL_THREADING` and
`DUCC0_NO_FFT_CACHE`. No shared plan cache or custom mutex adaptation is
required. The vendored threading header is kept identical to upstream.

The script normalizes trailing blank whitespace in `infra/aligned_array.h` so
the vendored source diff passes `git diff --check`. The normalization is
applied in a staging directory before the checked-in source tree is replaced.
Run `tools/vendoring/vendor_duccfft.sh --check` to verify that the current
vendored tree matches the pinned DUCC source and these patches. For offline
verification, pass `--source-dir PATH` for a local checkout at the pinned
commit. Update mode requires that local source tree to be clean.

`infra/threading.cc` also contains an optional, disabled helper adapted from
TensorFlow. Its attribution and Apache-2.0 license are included in
`LICENSE.md`.
