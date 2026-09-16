"""Benchmarks for the public one-dimensional FFT API."""

import numpy as np

from .common import Benchmark


class FFT(Benchmark):
    params = [
        ["complex64", "complex128"],
        [1024, 4096, 4097, 65536],
        [1, 8],
    ]
    param_names = ["dtype", "n", "batch"]

    def setup(self, dtype, n, batch):
        shape = (n,) if batch == 1 else (batch, n)
        rng = np.random.default_rng(20260915 + n + batch)
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)
        self.complex_input = (real + 1j * imag).astype(dtype)
        real_dtype = "float32" if dtype == "complex64" else "float64"
        self.real_input = real.astype(real_dtype)
        self.real_output = np.fft.rfft(self.real_input, axis=-1)

    def time_fft(self, dtype, n, batch):
        np.fft.fft(self.complex_input, axis=-1)

    def time_ifft(self, dtype, n, batch):
        np.fft.ifft(self.complex_input, axis=-1)

    def time_rfft(self, dtype, n, batch):
        np.fft.rfft(self.real_input, axis=-1)

    def time_irfft(self, dtype, n, batch):
        np.fft.irfft(self.real_output, n=n, axis=-1)
