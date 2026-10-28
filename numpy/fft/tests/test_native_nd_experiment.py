"""Experimental DUCC native multi-axis C2C parity and fallback checks."""

import numpy as np
import pytest

from numpy.fft import _ducc as backend
from numpy.fft import _duccfft_umath


def _sequential(a, axes, inverse, norm):
    func = np.fft.ifft if inverse else np.fft.fft
    value = a
    for axis in reversed(axes):
        value = func(value, axis=axis, norm=norm)
    return value


@pytest.mark.parametrize("shape,axes", [
    ((4, 8), (0, 1)),
    ((7, 9), (1, 0)),
    ((3, 8, 17), (0, 2)),
    ((4, 7, 8), (2, 0, 1)),
])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_native_c2c_nd_matches_sequential(shape, axes, dtype, inverse, norm):
    rng = np.random.default_rng(2026)
    a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    selected = np.fft.ifftn if inverse else np.fft.fftn
    fn = np.fft.ifft if inverse else np.fft.fft
    got = selected(a, axes=axes, norm=norm)
    expected = _sequential(a, axes, inverse, norm)
    atol = 2e-5 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(got, expected, rtol=atol, atol=atol)
    assert got.dtype == dtype
    # Verify that private native entry point exists and is capable of
    # processing the eligible input, rather than only testing fallback.
    native = backend._try_native_c2c_nd(
        a, [a.shape[axis] for axis in axes], axes, fn, norm, None
    )
    assert native is not None
    np.testing.assert_allclose(native, expected, rtol=atol, atol=atol)


@pytest.mark.parametrize("shape", [(4, 7), (2, 5, 8)])
def test_noncontiguous_and_repeated_axis_fallback(shape):
    rng = np.random.default_rng(90)
    a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    view = a[..., ::-1]
    axes = (0, view.ndim - 1)
    assert backend._try_native_c2c_nd(
        view, [view.shape[ax] for ax in axes], axes, np.fft.fft,
        None, None
    ) is None
    assert backend._try_native_c2c_nd(
        a, [a.shape[0], a.shape[0]], (0, 0), np.fft.fft,
        None, None
    ) is None
    np.testing.assert_allclose(
        np.fft.fftn(view, axes=axes), _sequential(view, axes, False, None),
        rtol=1e-12, atol=1e-12
    )


def test_resizing_and_out_fallback():
    a = np.arange(48, dtype=np.complex128).reshape(6, 8)
    assert backend._try_native_c2c_nd(a, (4, 8), (0, 1), np.fft.fft,
                                       None, None) is None
    result = np.empty_like(a)
    assert backend._try_native_c2c_nd(a, (6, 8), (0, 1), np.fft.fft,
                                       None, result) is None
    np.testing.assert_allclose(
        np.fft.fftn(a, s=(4, 8)), _sequential(a[:4], (0, 1), False, None),
        atol=1e-12
    )
    assert np.fft.fftn(a, out=result) is result


def test_native_extension_rejects_invalid_arrays():
    a = np.ones((4, 8), np.complex128)
    out = np.empty_like(a)
    with pytest.raises(ValueError, match="invalid or repeated"):
        _duccfft_umath.c2c_nd(a, out, (1, 1), True, 1.0)
    with pytest.raises(ValueError, match="matching complex"):
        _duccfft_umath.c2c_nd(a.real, out, (0, 1), True, 1.0)
