/*
 * Native multidimensional FFT entry points backed by DUCC.
 *
 * The public dispatch and compatibility fallback remain in _ducc.py.  This
 * module only supplies guarded native N-D primitives.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "numpy/arrayobject.h"

#include "ducc0/fft/fftnd_impl.h"

using ducc_shape_t = ducc0::fmav_info::shape_t;
using ducc_stride_t = ducc0::fmav_info::stride_t;

namespace {

static PyObject *
bool_result(bool value)
{
    return PyBool_FromLong(value ? 1 : 0);
}

/* Return false without leaving an exception for an unsupported native view. */
static int
parse_axes(PyObject *axes_obj, int ndim, ducc_shape_t& axes)
{
    PyObject *seq = PySequence_Fast(axes_obj, "axes must be a sequence");
    if (seq == nullptr) {
        return -1;
    }

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > ndim) {
        Py_DECREF(seq);
        PyErr_Clear();
        return 0;
    }

    std::vector<bool> seen(static_cast<size_t>(ndim), false);
    axes.clear();
    axes.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        Py_ssize_t axis = PyLong_AsSsize_t(item);
        if (axis == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            Py_DECREF(seq);
            return 0;
        }
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim || seen[static_cast<size_t>(axis)]) {
            Py_DECREF(seq);
            PyErr_Clear();
            return 0;
        }
        seen[static_cast<size_t>(axis)] = true;
        axes.push_back(static_cast<size_t>(axis));
    }

    Py_DECREF(seq);
    return 1;
}

static bool
absolute_stride(ptrdiff_t stride, size_t& result)
{
    if (stride < 0) {
        result = static_cast<size_t>(-(stride + 1)) + 1;
    }
    else {
        result = static_cast<size_t>(stride);
    }
    return true;
}

template <typename T>
static bool
view_info(PyArrayObject *array, ducc_shape_t& shape, ducc_stride_t& strides)
{
    if (!PyArray_ISALIGNED(array) || PyArray_DATA(array) == nullptr ||
            reinterpret_cast<std::uintptr_t>(PyArray_DATA(array)) % alignof(T) != 0) {
        return false;
    }

    int ndim = PyArray_NDIM(array);
    shape.resize(static_cast<size_t>(ndim));
    strides.resize(static_cast<size_t>(ndim));
    for (int i = 0; i < ndim; ++i) {
        npy_intp dimension = PyArray_DIM(array, i);
        if (dimension <= 0 ||
                static_cast<std::uintmax_t>(dimension) >
                    std::numeric_limits<size_t>::max()) {
            return false;
        }
        shape[static_cast<size_t>(i)] = static_cast<size_t>(dimension);

        npy_intp byte_stride = PyArray_STRIDE(array, i);
        npy_intp itemsize = static_cast<npy_intp>(sizeof(T));
        if (byte_stride % itemsize != 0) {
            return false;
        }
        npy_intp element_stride = byte_stride / itemsize;
        if (element_stride > std::numeric_limits<ptrdiff_t>::max() ||
                element_stride < std::numeric_limits<ptrdiff_t>::min()) {
            return false;
        }
        strides[static_cast<size_t>(i)] = static_cast<ptrdiff_t>(element_stride);
    }
    return true;
}

static bool
array_range(PyArrayObject *array, size_t itemsize,
            std::uintptr_t& begin, std::uintptr_t& end)
{
    const std::uintptr_t max_uint = std::numeric_limits<std::uintptr_t>::max();
    std::uintptr_t low = reinterpret_cast<std::uintptr_t>(PyArray_DATA(array));
    std::uintptr_t high = low;
    int ndim = PyArray_NDIM(array);
    for (int i = 0; i < ndim; ++i) {
        npy_intp dimension = PyArray_DIM(array, i);
        if (dimension <= 1 || PyArray_STRIDE(array, i) == 0) {
            continue;
        }
        ptrdiff_t stride = static_cast<ptrdiff_t>(PyArray_STRIDE(array, i));
        size_t magnitude;
        absolute_stride(stride, magnitude);
        size_t extent = static_cast<size_t>(dimension - 1);
        if (magnitude != 0 && extent > max_uint / magnitude) {
            return false;
        }
        std::uintptr_t offset = extent * magnitude;
        if (stride < 0) {
            if (low < offset) {
                return false;
            }
            low -= offset;
        }
        else {
            if (high > max_uint - offset) {
                return false;
            }
            high += offset;
        }
    }
    if (high > max_uint - itemsize) {
        return false;
    }
    begin = low;
    end = high + itemsize;
    return true;
}

static bool
arrays_overlap(PyArrayObject *first, size_t first_itemsize,
               PyArrayObject *second, size_t second_itemsize)
{
    std::uintptr_t first_begin, first_end, second_begin, second_end;
    if (!array_range(first, first_itemsize, first_begin, first_end) ||
            !array_range(second, second_itemsize, second_begin, second_end)) {
        return true;
    }
    return first_begin < second_end && second_begin < first_end;
}

static bool
has_internal_overlap(const ducc_shape_t& shape, const ducc_stride_t& strides,
                     size_t itemsize)
{
    std::vector<std::pair<size_t, size_t>> dimensions;
    dimensions.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 1) {
            continue;
        }
        size_t stride;
        absolute_stride(strides[i], stride);
        if (stride > std::numeric_limits<size_t>::max() / itemsize) {
            return true;
        }
        dimensions.emplace_back(stride * itemsize, shape[i]);
    }
    std::sort(dimensions.begin(), dimensions.end());

    size_t span = itemsize;
    for (const auto& dimension : dimensions) {
        if (dimension.first < span) {
            return true;
        }
        size_t extent = dimension.second - 1;
        if (extent != 0 &&
                dimension.first >
                    (std::numeric_limits<size_t>::max() - span) / extent) {
            return true;
        }
        span += extent * dimension.first;
    }
    return false;
}

static bool
same_layout(PyArrayObject *first, PyArrayObject *second)
{
    if (PyArray_DATA(first) != PyArray_DATA(second) ||
            PyArray_NDIM(first) != PyArray_NDIM(second)) {
        return false;
    }
    for (int i = 0; i < PyArray_NDIM(first); ++i) {
        if (PyArray_DIM(first, i) != PyArray_DIM(second, i) ||
                PyArray_STRIDE(first, i) != PyArray_STRIDE(second, i)) {
            return false;
        }
    }
    return true;
}

template <typename Tin, typename Tout>
static bool
prepare_views(PyArrayObject *input, PyArrayObject *output, bool allow_inplace,
              ducc_shape_t& input_shape, ducc_stride_t& input_strides,
              ducc_shape_t& output_shape, ducc_stride_t& output_strides)
{
    if (!PyArray_ISNOTSWAPPED(input) || !PyArray_ISNOTSWAPPED(output) ||
            !PyArray_ISWRITEABLE(output) ||
            PyArray_NDIM(input) != PyArray_NDIM(output) ||
            !view_info<Tin>(input, input_shape, input_strides) ||
            !view_info<Tout>(output, output_shape, output_strides) ||
            has_internal_overlap(output_shape, output_strides, sizeof(Tout))) {
        return false;
    }

    bool exact_inplace = allow_inplace && same_layout(input, output);
    if (arrays_overlap(input, sizeof(Tin), output, sizeof(Tout)) &&
            !exact_inplace) {
        return false;
    }
    return true;
}

static bool
same_shape(const ducc_shape_t& first, const ducc_shape_t& second)
{
    return first == second;
}

static bool
normalization_factor(int inorm, const ducc_shape_t& shape,
                     const ducc_shape_t& axes, long double& factor)
{
    if (inorm == 0) {
        factor = 1;
        return true;
    }
    if (inorm != 1 && inorm != 2) {
        return false;
    }

    size_t length = 1;
    for (size_t axis : axes) {
        if (axis >= shape.size() || shape[axis] == 0 ||
                length > std::numeric_limits<size_t>::max() / shape[axis]) {
            return false;
        }
        length *= shape[axis];
    }
    long double n = static_cast<long double>(length);
    factor = inorm == 1 ? 1 / std::sqrt(n) : 1 / n;
    return true;
}

template <typename Func>
static PyObject *
run_transform(Func&& transform)
{
    try {
        std::exception_ptr error;
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS;
        try {
            transform();
        }
        catch (...) {
            error = std::current_exception();
        }
        NPY_END_THREADS;
        if (error) {
            std::rethrow_exception(error);
        }
        return bool_result(true);
    }
    catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    }
    catch (const std::invalid_argument&) {
        // A DUCC precondition was not representable by this view; let Python
        // run the established sequential implementation instead.
        return bool_result(false);
    }
    catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

template <typename Func>
static PyObject *
call_safely(Func&& func)
{
    try {
        return func();
    }
    catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    }
    catch (const std::invalid_argument&) {
        return bool_result(false);
    }
    catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

template <typename T>
static PyObject *
c2c_impl(PyArrayObject *input, PyArrayObject *output,
         const ducc_shape_t& axes, int inorm, bool forward)
{
    using complex_t = std::complex<T>;
    ducc_shape_t input_shape, output_shape;
    ducc_stride_t input_strides, output_strides;
    if (!prepare_views<complex_t, complex_t>(
                input, output, true, input_shape, input_strides,
                output_shape, output_strides) ||
            !same_shape(input_shape, output_shape)) {
        return bool_result(false);
    }

    long double factor;
    if (!normalization_factor(inorm, input_shape, axes, factor)) {
        return bool_result(false);
    }
    auto input_view = ducc0::cfmav<complex_t>(
        reinterpret_cast<const complex_t *>(PyArray_DATA(input)),
        input_shape, input_strides);
    auto output_view = ducc0::vfmav<complex_t>(
        reinterpret_cast<complex_t *>(PyArray_DATA(output)),
        output_shape, output_strides);
    T fct = static_cast<T>(factor);
    return run_transform([&]() {
        ducc0::c2c(input_view, output_view, axes, forward, fct, 1);
    });
}

template <typename T>
static PyObject *
r2c_impl(PyArrayObject *input, PyArrayObject *output,
         const ducc_shape_t& axes, int inorm)
{
    using complex_t = std::complex<T>;
    ducc_shape_t input_shape, output_shape;
    ducc_stride_t input_strides, output_strides;
    if (!prepare_views<T, complex_t>(
                input, output, false, input_shape, input_strides,
                output_shape, output_strides)) {
        return bool_result(false);
    }
    if (axes.empty() || axes.back() >= input_shape.size()) {
        return bool_result(false);
    }
    for (size_t i = 0; i < input_shape.size(); ++i) {
        size_t expected = input_shape[i];
        if (i == axes.back()) {
            expected = expected / 2 + 1;
        }
        if (output_shape[i] != expected) {
            return bool_result(false);
        }
    }

    long double factor;
    if (!normalization_factor(inorm, input_shape, axes, factor)) {
        return bool_result(false);
    }
    auto input_view = ducc0::cfmav<T>(
        reinterpret_cast<const T *>(PyArray_DATA(input)),
        input_shape, input_strides);
    auto output_view = ducc0::vfmav<complex_t>(
        reinterpret_cast<complex_t *>(PyArray_DATA(output)),
        output_shape, output_strides);
    T fct = static_cast<T>(factor);
    return run_transform([&]() {
        ducc0::r2c(input_view, output_view, axes, true, fct, 1);
    });
}

template <typename T>
static PyObject *
c2r_impl(PyArrayObject *input, PyArrayObject *output,
         const ducc_shape_t& axes, int inorm)
{
    using complex_t = std::complex<T>;
    ducc_shape_t input_shape, output_shape;
    ducc_stride_t input_strides, output_strides;
    if (!prepare_views<complex_t, T>(
                input, output, false, input_shape, input_strides,
                output_shape, output_strides)) {
        return bool_result(false);
    }
    if (axes.empty() || axes.back() >= output_shape.size()) {
        return bool_result(false);
    }
    for (size_t i = 0; i < output_shape.size(); ++i) {
        size_t expected = output_shape[i];
        if (i == axes.back()) {
            expected = expected / 2 + 1;
        }
        if (input_shape[i] != expected) {
            return bool_result(false);
        }
    }

    long double factor;
    if (!normalization_factor(inorm, output_shape, axes, factor)) {
        return bool_result(false);
    }
    auto input_view = ducc0::cfmav<complex_t>(
        reinterpret_cast<const complex_t *>(PyArray_DATA(input)),
        input_shape, input_strides);
    auto output_view = ducc0::vfmav<T>(
        reinterpret_cast<T *>(PyArray_DATA(output)),
        output_shape, output_strides);
    T fct = static_cast<T>(factor);
    return run_transform([&]() {
        ducc0::c2r(input_view, output_view, axes, false, fct, 1);
    });
}

static PyObject *
ducc_c2c(PyObject *, PyObject *args)
{
    PyObject *input_obj, *output_obj, *axes_obj;
    int inorm, forward;
    if (!PyArg_ParseTuple(args, "OOOip", &input_obj, &output_obj, &axes_obj,
                          &inorm, &forward)) {
        return nullptr;
    }
    return call_safely([&]() -> PyObject * {
        if (!PyArray_Check(input_obj) || !PyArray_Check(output_obj)) {
            return bool_result(false);
        }
        auto *input = reinterpret_cast<PyArrayObject *>(input_obj);
        auto *output = reinterpret_cast<PyArrayObject *>(output_obj);
        if (inorm < 0 || inorm > 2 || PyArray_TYPE(input) != PyArray_TYPE(output)) {
            return bool_result(false);
        }

        ducc_shape_t axes;
        int parsed = parse_axes(axes_obj, PyArray_NDIM(input), axes);
        if (parsed < 0) {
            return nullptr;
        }
        if (parsed == 0) {
            return bool_result(false);
        }
        switch (PyArray_TYPE(input)) {
            case NPY_CFLOAT:
                return c2c_impl<npy_float>(input, output, axes, inorm,
                                           forward != 0);
            case NPY_CDOUBLE:
                return c2c_impl<npy_double>(input, output, axes, inorm,
                                            forward != 0);
            case NPY_CLONGDOUBLE:
                return c2c_impl<npy_longdouble>(input, output, axes, inorm,
                                                forward != 0);
            default:
                return bool_result(false);
        }
    });
}

static PyObject *
ducc_r2c(PyObject *, PyObject *args)
{
    PyObject *input_obj, *output_obj, *axes_obj;
    int inorm;
    if (!PyArg_ParseTuple(args, "OOOi", &input_obj, &output_obj, &axes_obj,
                          &inorm)) {
        return nullptr;
    }
    return call_safely([&]() -> PyObject * {
        if (!PyArray_Check(input_obj) || !PyArray_Check(output_obj)) {
            return bool_result(false);
        }
        auto *input = reinterpret_cast<PyArrayObject *>(input_obj);
        auto *output = reinterpret_cast<PyArrayObject *>(output_obj);
        if (inorm < 0 || inorm > 2) {
            return bool_result(false);
        }

        ducc_shape_t axes;
        int parsed = parse_axes(axes_obj, PyArray_NDIM(input), axes);
        if (parsed < 0) {
            return nullptr;
        }
        if (parsed == 0) {
            return bool_result(false);
        }
        if (PyArray_TYPE(input) == NPY_FLOAT &&
                PyArray_TYPE(output) == NPY_CFLOAT) {
            return r2c_impl<npy_float>(input, output, axes, inorm);
        }
        if (PyArray_TYPE(input) == NPY_DOUBLE &&
                PyArray_TYPE(output) == NPY_CDOUBLE) {
            return r2c_impl<npy_double>(input, output, axes, inorm);
        }
        if (PyArray_TYPE(input) == NPY_LONGDOUBLE &&
                PyArray_TYPE(output) == NPY_CLONGDOUBLE) {
            return r2c_impl<npy_longdouble>(input, output, axes, inorm);
        }
        return bool_result(false);
    });
}

static PyObject *
ducc_c2r(PyObject *, PyObject *args)
{
    PyObject *input_obj, *output_obj, *axes_obj;
    int inorm;
    if (!PyArg_ParseTuple(args, "OOOi", &input_obj, &output_obj, &axes_obj,
                          &inorm)) {
        return nullptr;
    }
    return call_safely([&]() -> PyObject * {
        if (!PyArray_Check(input_obj) || !PyArray_Check(output_obj)) {
            return bool_result(false);
        }
        auto *input = reinterpret_cast<PyArrayObject *>(input_obj);
        auto *output = reinterpret_cast<PyArrayObject *>(output_obj);
        if (inorm < 0 || inorm > 2) {
            return bool_result(false);
        }

        ducc_shape_t axes;
        int parsed = parse_axes(axes_obj, PyArray_NDIM(output), axes);
        if (parsed < 0) {
            return nullptr;
        }
        if (parsed == 0) {
            return bool_result(false);
        }
        if (PyArray_TYPE(input) == NPY_CFLOAT &&
                PyArray_TYPE(output) == NPY_FLOAT) {
            return c2r_impl<npy_float>(input, output, axes, inorm);
        }
        if (PyArray_TYPE(input) == NPY_CDOUBLE &&
                PyArray_TYPE(output) == NPY_DOUBLE) {
            return c2r_impl<npy_double>(input, output, axes, inorm);
        }
        if (PyArray_TYPE(input) == NPY_CLONGDOUBLE &&
                PyArray_TYPE(output) == NPY_LONGDOUBLE) {
            return c2r_impl<npy_longdouble>(input, output, axes, inorm);
        }
        return bool_result(false);
    });
}

static PyMethodDef module_methods[] = {
    {"c2c", ducc_c2c, METH_VARARGS, nullptr},
    {"r2c", ducc_r2c, METH_VARARGS, nullptr},
    {"c2r", ducc_c2r, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

static int
_ducc_nd_umath_exec(PyObject *)
{
    return PyArray_ImportNumPyAPI() < 0 ? -1 : 0;
}

static struct PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void *)_ducc_nd_umath_exec},
#if PY_VERSION_HEX >= 0x030c00f0
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0 && (!defined(Py_LIMITED_API) || Py_LIMITED_API >= 0x030d0000)
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, nullptr}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_ducc_nd_umath",
    nullptr,
    0,
    module_methods,
    module_slots,
};

}  // namespace

PyMODINIT_FUNC
PyInit__ducc_nd_umath(void)
{
    return PyModuleDef_Init(&moduledef);
}
