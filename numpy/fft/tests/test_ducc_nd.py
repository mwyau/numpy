"""Tests for NumPy's guarded native DUCC N-dimensional FFT path."""

import pytest

import numpy as np
from numpy.fft import _ducc_nd, _ducc_nd_umath as ducc_nd
from numpy.testing import assert_allclose


def _complex_input(shape, dtype=np.complex128):
    size = np.prod(shape)
    real = np.arange(size, dtype=np.float64).reshape(shape)
    imag = np.arange(size, dtype=np.float64)[::-1].reshape(shape)
    return (real + 1j * imag).astype(dtype)


def _real_input(shape, dtype=np.float64):
    return np.arange(np.prod(shape), dtype=np.float64).reshape(shape).astype(dtype)


def _sequential_c2c(a, axes, forward=True, norm="backward"):
    function = np.fft.fft if forward else np.fft.ifft
    for axis in reversed(axes):
        a = function(a, axis=axis, norm=norm)
    return a


def _sequential_r2c(a, axes, norm="backward"):
    a = np.fft.rfft(a, axis=axes[-1], norm=norm)
    for axis in reversed(axes[:-1]):
        a = np.fft.fft(a, axis=axis, norm=norm)
    return a


def _sequential_c2r(a, axes, shape, norm="backward", s=None):
    if s is None:
        s = tuple(shape[axis] for axis in axes)
    for axis, size in zip(axes[:-1], s[:-1]):
        a = np.fft.ifft(a, n=size, axis=axis, norm=norm)
    return np.fft.irfft(a, n=s[-1], axis=axes[-1], norm=norm)


def _tolerance(dtype, size):
    dtype = np.dtype(dtype)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    eps = np.finfo(real_dtype).eps
    size = max(int(size), 2)
    # Match the established FFT-test tolerance, with the same modest extra
    # factor used for DUCC's different multidimensional traversal order and a
    # sqrt(size) scale for accumulated absolute roundoff.
    return 8.0 * 8.0 * np.sqrt(size * np.log2(size)) * eps


def _byte_swapped(array):
    return array.astype(array.dtype.newbyteorder("S"), copy=True)


def _record_native_calls(monkeypatch, name):
    calls = []
    function = getattr(_ducc_nd.ducc_nd, name)

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr(_ducc_nd.ducc_nd, name, wrapped)
    return calls


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.clongdouble])
@pytest.mark.parametrize("layout", ["C", "F", "transpose", "negative"])
def test_native_c2c_layouts(dtype, layout):
    shape = (3, 4, 5)
    base = _complex_input(shape, dtype)
    if layout == "C":
        a = base
        out = np.empty(shape, dtype=dtype)
    elif layout == "F":
        a = np.asfortranarray(base)
        out = np.empty(shape, dtype=dtype, order="F")
    elif layout == "transpose":
        a = base.transpose(2, 1, 0)
        out = np.empty((3, 4, 5), dtype=dtype).transpose(2, 1, 0)
    else:
        a = base[::-1, :, ::-1]
        out = np.empty(shape, dtype=dtype)[:, :, ::-1]

    axes = (0, 2)
    assert ducc_nd.c2c(a, out, axes, 0, True)
    expected = _sequential_c2c(a, axes)
    tol = _tolerance(dtype, np.prod(shape))
    assert_allclose(out, expected, rtol=tol, atol=tol)


def test_native_c2c_in_place():
    a = _complex_input((4, 5, 6))
    expected = _sequential_c2c(a, (0, 2))
    assert ducc_nd.c2c(a, a, (0, 2), 0, True)
    assert_allclose(a, expected)


@pytest.mark.parametrize("operation", ["c2c", "r2c", "c2r"])
def test_native_rejects_non_native_byte_order(operation):
    shape = (3, 4, 6)
    axes = (0, 2)
    real = _real_input(shape)
    if operation == "c2c":
        input_array = _byte_swapped(_complex_input(shape))
        output = np.empty(shape, dtype=np.complex128)
        assert not ducc_nd.c2c(input_array, output, axes, 0, True)
    elif operation == "r2c":
        input_array = _byte_swapped(real)
        output_shape = list(shape)
        output_shape[axes[-1]] = shape[axes[-1]] // 2 + 1
        output = np.empty(output_shape, dtype=np.complex128)
        assert not ducc_nd.r2c(input_array, output, axes, 0)
    else:
        input_array = _byte_swapped(_sequential_r2c(real, axes))
        output = np.empty(shape, dtype=np.float64)
        assert not ducc_nd.c2r(input_array, output, axes, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize("axes", [(0, 2), (2, 0), (1, 2, 0)])
def test_native_real_transforms(dtype, axes):
    shape = (3, 4, 6)
    a = _real_input(shape, dtype)
    output_shape = list(shape)
    output_shape[axes[-1]] = shape[axes[-1]] // 2 + 1
    output = np.empty(output_shape, dtype=np.result_type(dtype, 1j), order="F")

    assert ducc_nd.r2c(a, output, axes, 0)
    expected = _sequential_r2c(a, axes)
    tol = _tolerance(dtype, np.prod(shape))
    assert_allclose(output, expected, rtol=tol, atol=tol)

    recovered = np.empty(shape, dtype=dtype)
    assert ducc_nd.c2r(output, recovered, axes, 2)
    expected = _sequential_c2r(output, axes, shape)
    assert_allclose(recovered, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("axes", [(0, 1), (2, 0), (1, 2, 0)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_public_complex_nd_matches_sequential(axes, norm):
    a = _complex_input((4, 5, 6))
    tol = _tolerance(a.dtype, a.size)
    expected = _sequential_c2c(a, axes, norm=norm)
    result = np.fft.fftn(a, axes=axes, norm=norm)
    assert_allclose(result, expected, rtol=tol, atol=tol)

    expected = _sequential_c2c(a, axes, forward=False, norm=norm)
    result = np.fft.ifftn(a, axes=axes, norm=norm)
    assert_allclose(result, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("axes", [(0, 1), (2, 0), (1, 2, 0)])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_public_real_nd_matches_sequential(axes, norm):
    shape = (4, 5, 6)
    a = _real_input(shape)
    tol = _tolerance(a.dtype, a.size)
    transform_shape = tuple(shape[axis] for axis in axes)
    expected = _sequential_r2c(a, axes, norm)
    result = np.fft.rfftn(a, axes=axes, norm=norm)
    assert_allclose(result, expected, rtol=tol, atol=tol)

    expected = _sequential_c2r(result, axes, shape, norm)
    recovered = np.fft.irfftn(result, s=transform_shape, axes=axes, norm=norm)
    assert_allclose(recovered, expected, rtol=tol, atol=tol)


def test_public_byte_swapped_inputs_fall_back(monkeypatch):
    shape = (3, 4, 6)
    axes = (0, 2)
    real = _byte_swapped(_real_input(shape))
    complex_input = _byte_swapped(_complex_input(shape))
    real_spectrum = _byte_swapped(_sequential_r2c(_real_input(shape), axes))

    def fail(*args):
        raise AssertionError("byte-swapped input used native DUCC")

    monkeypatch.setattr(_ducc_nd.ducc_nd, "c2c", fail)
    monkeypatch.setattr(_ducc_nd.ducc_nd, "r2c", fail)
    monkeypatch.setattr(_ducc_nd.ducc_nd, "c2r", fail)

    expected_r2c = _sequential_r2c(real, axes)
    result_r2c = np.fft.rfftn(real, axes=axes)
    assert result_r2c.dtype == expected_r2c.dtype
    assert result_r2c.dtype.isnative == expected_r2c.dtype.isnative
    assert_allclose(result_r2c, expected_r2c)

    expected_c2c = _sequential_c2c(complex_input, axes)
    result_c2c = np.fft.fftn(complex_input, axes=axes)
    assert result_c2c.dtype == expected_c2c.dtype
    assert result_c2c.dtype.isnative == expected_c2c.dtype.isnative
    assert_allclose(result_c2c, expected_c2c)

    transform_shape = tuple(shape[axis] for axis in axes)
    expected_c2r = _sequential_c2r(real_spectrum, axes, shape)
    result_c2r = np.fft.irfftn(real_spectrum, s=transform_shape, axes=axes)
    assert result_c2r.dtype == expected_c2r.dtype
    assert result_c2r.dtype.isnative == expected_c2r.dtype.isnative
    assert_allclose(result_c2r, expected_c2r)


@pytest.mark.parametrize("operation", ["c2c", "r2c", "c2r"])
def test_public_eligible_calls_use_native(monkeypatch, operation):
    calls = _record_native_calls(monkeypatch, operation)
    shape = (3, 4, 6)
    axes = (0, 2)
    real = _real_input(shape)
    if operation == "c2c":
        result = np.fft.fftn(_complex_input(shape), axes=axes)
    elif operation == "r2c":
        result = np.fft.rfftn(real, axes=axes)
    else:
        spectrum = _sequential_r2c(real, axes)
        result = np.fft.irfftn(
            spectrum,
            s=tuple(shape[axis] for axis in axes),
            axes=axes,
        )
    assert len(calls) == 1
    assert result.size


@pytest.mark.parametrize(
    "operation, layout", [("c2c", "transpose"), ("r2c", "F"), ("c2r", "negative")]
)
def test_public_strided_layouts_use_native(monkeypatch, operation, layout):
    calls = _record_native_calls(monkeypatch, operation)
    shape = (3, 4, 6)
    axes = (0, 2)
    real = _real_input(shape)
    if operation == "c2c":
        array = _complex_input(shape).transpose(2, 1, 0)
        expected = _sequential_c2c(array, axes)
        result = np.fft.fftn(array, axes=axes)
    elif operation == "r2c":
        array = np.asfortranarray(real)
        expected = _sequential_r2c(array, axes)
        result = np.fft.rfftn(array, axes=axes)
    else:
        array = _sequential_r2c(real, axes)[::-1, :, ::-1]
        expected = _sequential_c2r(array, axes, shape)
        result = np.fft.irfftn(
            array,
            s=tuple(shape[axis] for axis in axes),
            axes=axes,
        )
    assert len(calls) == 1
    tol = _tolerance(real.dtype, real.size)
    assert_allclose(result, expected, rtol=tol, atol=tol)


def test_subclass_output_uses_established_fallback(monkeypatch):
    class RecordingArray(np.ndarray):
        calls = 0

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            type(self).calls += 1
            inputs = tuple(
                value.view(np.ndarray) if isinstance(value, type(self)) else value
                for value in inputs
            )
            if kwargs.get("out") is not None:
                kwargs["out"] = tuple(
                    value.view(np.ndarray) if isinstance(value, type(self)) else value
                    for value in kwargs["out"]
                )
            return getattr(ufunc, method)(*inputs, **kwargs)

    def fail(*args):
        raise AssertionError("ndarray subclass output used native DUCC")

    monkeypatch.setattr(_ducc_nd.ducc_nd, "c2c", fail)
    array = _complex_input((3, 4, 6))
    expected = _sequential_c2c(array, (0, 2))
    output = np.empty_like(array).view(RecordingArray)
    result = np.fft.fftn(array, axes=(0, 2), out=output)

    assert RecordingArray.calls
    assert_allclose(np.asarray(output), expected)
    assert_allclose(np.asarray(result), expected)


def test_public_nd_out_and_layouts():
    a = _complex_input((4, 5, 6))
    expected = np.fft.fftn(a, axes=(0, 2))
    out = np.empty(expected.shape, dtype=expected.dtype, order="F")
    result = np.fft.fftn(a, axes=(0, 2), out=out)
    assert result is out
    assert_allclose(result, expected)

    real = _real_input((4, 5, 6))
    expected = np.fft.rfftn(real, axes=(2, 0))
    out = np.empty(expected.shape, dtype=expected.dtype)[:, ::-1, :]
    result = np.fft.rfftn(real, axes=(2, 0), out=out)
    assert result is out
    assert_allclose(result, expected)

    transform_shape = tuple(real.shape[axis] for axis in (2, 0))
    expected = np.fft.irfftn(expected, s=transform_shape, axes=(2, 0))
    out = np.empty(expected.shape, dtype=expected.dtype, order="F")
    result = np.fft.irfftn(
        np.fft.rfftn(real, axes=(2, 0)),
        s=transform_shape,
        axes=(2, 0),
        out=out,
    )
    assert result is out
    assert_allclose(result, expected)


def test_s_minus_one_and_aliases():
    a = _complex_input((4, 5, 6))
    expected = _sequential_c2c(a, (0, 2))
    result = np.fft.fftn(a, s=(-1, -1), axes=(0, 2))
    assert_allclose(result, expected)

    aliased = a.copy()
    result = np.fft.fftn(aliased, axes=(0, 2), out=aliased)
    assert result is aliased
    assert_allclose(result, expected)

    aliased = a.copy().transpose(2, 1, 0)
    expected = np.fft.fftn(aliased.copy(), axes=(0, 2))
    result = np.fft.fftn(aliased, axes=(0, 2), out=aliased)
    assert result is aliased
    assert_allclose(result, expected)


def test_fallback_for_repeated_axes(monkeypatch):
    def fail(*args):
        raise AssertionError("native DUCC path was not supposed to run")

    monkeypatch.setattr(_ducc_nd.ducc_nd, "c2c", fail)
    a = _complex_input((3, 4, 5))
    expected = _sequential_c2c(a, (0, 0, 2))
    result = np.fft.fftn(a, axes=(0, 0, 2))
    assert_allclose(result, expected)


@pytest.mark.parametrize("operation", ["r2c", "c2r"])
def test_fallback_for_repeated_real_axes(monkeypatch, operation):
    def fail(*args):
        raise AssertionError("repeated axes used native DUCC")

    monkeypatch.setattr(_ducc_nd.ducc_nd, operation, fail)
    shape = (3, 4, 6)
    axes = (0, 0, 2)
    real = _real_input(shape)
    if operation == "r2c":
        expected = _sequential_r2c(real, axes)
        result = np.fft.rfftn(real, axes=axes)
    else:
        spectrum = _sequential_r2c(real, axes)
        expected = _sequential_c2r(spectrum, axes, shape, s=shape)
        result = np.fft.irfftn(spectrum, s=shape, axes=axes)
    assert_allclose(result, expected)


@pytest.mark.parametrize("operation", ["c2c", "r2c"])
def test_fallback_for_padding_and_unaligned_input(monkeypatch, operation):
    def fail(*args):
        raise AssertionError("native DUCC path was not supposed to run")

    monkeypatch.setattr(_ducc_nd.ducc_nd, operation, fail)
    if operation == "c2c":
        a = _complex_input((3, 4))
        expected = _sequential_c2c(np.fft.fft(a, n=5, axis=1), (0,))
        result = np.fft.fftn(a, s=(3, 5), axes=(0, 1))
        assert_allclose(result, expected)

        storage = bytearray(a.nbytes + 1)
        unaligned = np.ndarray(a.shape, dtype=a.dtype, buffer=storage, offset=1)
        unaligned[...] = a
        expected = _sequential_c2c(unaligned, (0, 1))
        result = np.fft.fftn(unaligned)
    else:
        a = _real_input((3, 4), np.float32)
        storage = bytearray(a.nbytes + 1)
        unaligned = np.ndarray(a.shape, dtype=a.dtype, buffer=storage, offset=1)
        unaligned[...] = a
        expected = _sequential_r2c(unaligned, (0, 1))
        result = np.fft.rfftn(unaligned, axes=(0, 1))
    assert_allclose(result, expected)

    output_storage = bytearray(1 + expected.nbytes)
    unaligned_out = np.ndarray(
        expected.shape,
        dtype=expected.dtype,
        buffer=output_storage,
        offset=1,
    )
    if operation == "c2c":
        result = np.fft.fftn(a, axes=(0, 1), out=unaligned_out)
    else:
        result = np.fft.rfftn(a, axes=(0, 1), out=unaligned_out)
    assert_allclose(result, expected)


def test_dtype_promotion_stays_on_fallback(monkeypatch):
    def fail(*args):
        raise AssertionError("promoted dtype used native DUCC path")

    monkeypatch.setattr(_ducc_nd.ducc_nd, "c2c", fail)
    monkeypatch.setattr(_ducc_nd.ducc_nd, "r2c", fail)
    for dtype in [np.float16, np.int16, np.bool_]:
        a = np.arange(12).reshape(3, 4).astype(dtype)
        expected_dtype = np.result_type(dtype, 1j)
        result = np.fft.fftn(a)
        assert result.dtype == expected_dtype
        result = np.fft.rfftn(a)
        assert result.dtype == expected_dtype


def test_empty_axes_fall_back():
    a = np.empty((0, 4, 5), dtype=np.float64)
    result = np.fft.rfftn(a, axes=(1, 2))
    assert result.shape == (0, 4, 3)
