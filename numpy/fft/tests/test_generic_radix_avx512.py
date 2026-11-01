"""Independent numerical checks for experimental DUCC generic-radix FFTs.

This file is identical in the V2/A/B/C/D/E source branches. The reference is
a direct DFT at selected frequency bins, not NumPy's FFT implementation.
"""
import numpy as np
import pytest


ODD_PRIMES = (
    23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109,
)
LENGTHS = sorted(set(
    ODD_PRIMES
    + tuple(p * k for p in (23, 31, 89, 109) for k in (2, 3, 4, 7, 23))
    + (1023, 2047, 4095, 1024, 2048, 4096)
))


def _direct_bins(x, indices):
    n = x.shape[-1]
    locations = np.arange(n, dtype=np.float64)
    phase = -2j * np.pi * np.outer(indices, locations / n)
    return x.astype(np.complex128) @ np.exp(phase).T


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("batch", (1, 7, 16))
def test_generic_radix_direct_dft_and_roundtrip(dtype, n, batch):
    rng = np.random.default_rng(200_000 + n * 31 + batch)
    x = (rng.standard_normal((batch, n))
         + 1j * rng.standard_normal((batch, n))).astype(dtype)
    got = np.fft.fft(x, axis=-1)
    bins = sorted(set((0, 1, 2, n // 4, n // 2, n - 1)))
    ref = _direct_bins(x, bins)
    rel = np.linalg.norm(got[:, bins] - ref) / np.linalg.norm(ref)
    assert rel < (8e-6 if dtype == np.complex64 else 4e-11), (
        dtype, n, batch, rel
    )
    back = np.fft.ifft(got, axis=-1)
    rel_roundtrip = np.linalg.norm(back - x) / np.linalg.norm(x)
    assert rel_roundtrip < (4e-6 if dtype == np.complex64 else 4e-12), (
        dtype, n, batch, rel_roundtrip
    )


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("n", (23, 89, 667, 1023, 2047, 4095))
@pytest.mark.parametrize("batch", (1, 3, 8, 16))
def test_generic_radix_strided(dtype, n, batch):
    rng = np.random.default_rng(500_000 + n * 13 + batch)
    raw = np.empty((batch, n, 2), dtype=dtype)
    raw[:, :, 0] = (
        rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))
    )
    x = raw[:, :, 0]
    assert x.strides[-1] != x.itemsize
    result = np.fft.fft(x, axis=-1)
    bins = sorted(set((0, 1, n // 3, n // 2, n - 1)))
    ref = _direct_bins(x, bins)
    rel = np.linalg.norm(result[:, bins] - ref) / np.linalg.norm(ref)
    assert rel < (8e-6 if dtype == np.complex64 else 4e-11), (
        dtype, n, batch, rel
    )


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("radix", (2, 3, 4, 5, 7, 8, 11))
def test_dedicated_butterflies_not_regressed(dtype, radix):
    n = radix * 23
    rng = np.random.default_rng(n)
    x = (rng.standard_normal((7, n))
         + 1j * rng.standard_normal((7, n))).astype(dtype)
    bins = (0, 1, n // 3, n - 1)
    ref = _direct_bins(x, bins)
    got = np.fft.fft(x, axis=-1)[:, bins]
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < (8e-6 if dtype == np.complex64 else 4e-11)
