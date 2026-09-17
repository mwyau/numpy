The `ducc0` directory is populated by `tools/vendoring/vendor_duccfft.sh`.
The upstream project is [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc.git).
The script pins DUCC 0.41.1 at commit
`64f42ba531f609ba7029c82207a063b17f9d5275` and copies the explicit FFT source
manifest in that script.  DUCC sources are dual-licensed; NumPy distributes
these copied sources under the BSD-3-Clause option.
