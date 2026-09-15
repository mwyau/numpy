"""Eligibility and dispatch helpers for the private DUCC N-D extension."""

from numbers import Integral

from numpy._core import asarray, empty_like, result_type
from numpy._core.multiarray import may_share_memory, ndarray
from numpy.lib.array_utils import normalize_axis_index

from . import _ducc_nd_umath as ducc_nd


def _native_axes(a, s, axes, check_sizes=True):
    """Return normalized, unique axes when native N-D is applicable."""
    # A one-axis call is already handled by the gufunc backend.  Empty arrays
    # also stay on the established path because DUCC requires positive extents.
    if a.size == 0 or len(axes) < 2:
        return None
    normalized = tuple(normalize_axis_index(axis, a.ndim) for axis in axes)
    if len(set(normalized)) != len(normalized):
        return None
    if any(not isinstance(size, Integral) or size < 1 for size in s):
        return None
    if check_sizes and any(size != a.shape[axis] for size, axis in zip(s, normalized)):
        return None
    return normalized


def _native_array(a, dtype):
    """Return a DUCC-compatible working array without hiding unsafe strides."""
    if dtype.kind not in "fc" or dtype.itemsize < 4 or not dtype.isnative:
        return None
    a = asarray(a, dtype=dtype)
    if not a.flags.aligned:
        return None
    if any(stride % dtype.itemsize for stride in a.strides):
        return None
    return a


def _native_nonoverlapping(a):
    """Conservatively reject output views whose elements can overlap."""
    if a.size <= 1:
        return True
    dimensions = sorted(
        (abs(stride), size) for size, stride in zip(a.shape, a.strides) if size > 1
    )
    span = a.itemsize
    for stride, size in dimensions:
        if stride < span:
            return False
        span += (size - 1) * stride
    return True


def _native_output_valid(out, shape, dtype):
    return (
        type(out) is ndarray
        and out.shape == shape
        and out.dtype == dtype
        and out.flags.writeable
        and out.flags.aligned
        and all(stride % dtype.itemsize == 0 for stride in out.strides)
        and _native_nonoverlapping(out)
    )


def _native_output(a, out, shape, dtype):
    """Return a safe direct output, or ``None`` for the fallback path."""
    if out is None:
        return empty_like(a, shape=shape, dtype=dtype)
    if not _native_output_valid(out, shape, dtype) or may_share_memory(a, out):
        return None
    return out


def _native_alias_output(a, out, shape, dtype):
    """Return a temporary output and validated aliased destination."""
    if out is None or not _native_output_valid(out, shape, dtype):
        return None, None
    if not may_share_memory(a, out):
        return None, None
    return empty_like(a, shape=shape, dtype=dtype), out


def _native_norm(norm, forward):
    """Map NumPy's normalization names to DUCC's N-D convention."""
    if norm is None or norm == "backward":
        return 0 if forward else 2
    if norm == "ortho":
        return 1
    if norm == "forward":
        return 2 if forward else 0
    raise ValueError(
        f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".'
    )


def _native_fftnd(a, s, axes, norm, out, forward):
    axes = _native_axes(a, s, axes)
    if axes is None:
        return None

    # Preserve the public promotion rules.  Float32/64/longdouble inputs are
    # safe to convert to their corresponding complex DUCC type; float16,
    # integers, booleans, and other dtypes use the existing gufunc path.
    if (
        not a.dtype.isnative
        or a.dtype.kind not in "fc"
        or (a.dtype.kind == "f" and a.dtype.itemsize < 4)
    ):
        return None
    dtype = result_type(a.dtype, 1j)
    native_a = _native_array(a, dtype)
    if native_a is None:
        return None

    native_out, alias_out = _native_alias_output(a, out, a.shape, dtype)
    if alias_out is None:
        native_out = _native_output(a, out, a.shape, dtype)
        if native_out is None:
            return None
    if not ducc_nd.c2c(
        native_a, native_out, axes, _native_norm(norm, forward), forward
    ):
        return None
    if alias_out is not None:
        alias_out[...] = native_out
        return alias_out
    return native_out


def _native_rfftn(a, s, axes, norm, out):
    if not a.dtype.isnative or a.dtype.kind != "f" or a.dtype.itemsize < 4:
        return None
    axes = _native_axes(a, s, axes)
    if axes is None:
        return None
    native_a = _native_array(a, a.dtype)
    if native_a is None:
        return None

    shape = list(a.shape)
    shape[axes[-1]] = s[-1] // 2 + 1
    shape = tuple(shape)
    dtype = result_type(a.dtype, 1j)
    native_out, alias_out = _native_alias_output(a, out, shape, dtype)
    if alias_out is None:
        native_out = _native_output(a, out, shape, dtype)
        if native_out is None:
            return None
    if not ducc_nd.r2c(native_a, native_out, axes, _native_norm(norm, True)):
        return None
    if alias_out is not None:
        alias_out[...] = native_out
        return alias_out
    return native_out


def _native_irfftn(a, s, axes, norm, out):
    if not a.dtype.isnative or a.dtype.kind != "c" or a.dtype.itemsize < 8:
        return None
    axes = _native_axes(a, s, axes, check_sizes=False)
    if axes is None:
        return None
    if any(size != a.shape[axis] for size, axis in zip(s[:-1], axes[:-1])):
        return None
    if s[-1] // 2 + 1 != a.shape[axes[-1]]:
        return None

    native_a = _native_array(a, a.dtype)
    if native_a is None:
        return None
    shape = list(a.shape)
    shape[axes[-1]] = s[-1]
    shape = tuple(shape)
    real_dtype = a.real.dtype
    native_out, alias_out = _native_alias_output(a, out, shape, real_dtype)
    if alias_out is None:
        native_out = _native_output(a, out, shape, real_dtype)
        if native_out is None:
            return None
    if not ducc_nd.c2r(native_a, native_out, axes, _native_norm(norm, False)):
        return None
    if alias_out is not None:
        alias_out[...] = native_out
        return alias_out
    return native_out
