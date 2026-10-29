/*
 * Python gufunc module for the DUCC-backed one-dimensional FFT loops.
 *
 * The computational implementation is in _duccfft_impl.h.  Float and double
 * entry points are supplied by _duccfft.dispatch.cpp and selected once here
 * by NumPy's CPU dispatcher.  This translation unit is compiled exactly once
 * so module initialization and gufunc registration are never duplicated.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <new>
#include <vector>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "npy_config.h"
#include "npy_cpu_dispatch.h"

#include "_duccfft.dispatch.h"
#include "_duccfft_impl.h"

using duccfft_loop = PyUFuncGenericFunction;

#define DUCCFFT_DISPATCH_SIGNATURE \
    (char **, npy_intp const *, npy_intp const *, void *)

NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_c2c_double, DUCCFFT_DISPATCH_SIGNATURE)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_c2c_float, DUCCFFT_DISPATCH_SIGNATURE)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_rfft_even_double, DUCCFFT_DISPATCH_SIGNATURE)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_rfft_even_float, DUCCFFT_DISPATCH_SIGNATURE)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_rfft_odd_double, DUCCFFT_DISPATCH_SIGNATURE)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_rfft_odd_float, DUCCFFT_DISPATCH_SIGNATURE)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_irfft_double, DUCCFFT_DISPATCH_SIGNATURE)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_irfft_float, DUCCFFT_DISPATCH_SIGNATURE)


/* Native ND C2C is dispatched through the same CPU-target machinery. */
#define DUCCFFT_ND_SIGNATURE \
    (const void *, void *, const npy_intp *, const npy_intp *, \
     const npy_intp *, const npy_intp *, npy_intp, npy_intp, bool, double)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_native_nd_double, DUCCFFT_ND_SIGNATURE)
NPY_CPU_DISPATCH_DECLARE(
    NPY_NO_EXPORT void duccfft_native_nd_float, DUCCFFT_ND_SIGNATURE)
#undef DUCCFFT_ND_SIGNATURE

static void
wrap_cpp_ufunc(duccfft_loop implementation, char **args,
               npy_intp const *dimensions, npy_intp const *steps,
               void *func)
{
    NPY_ALLOW_C_API_DEF
    try {
        implementation(args, dimensions, steps, func);
    }
    catch (const std::bad_alloc&) {
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
    catch (const std::exception& e) {
        NPY_ALLOW_C_API;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        NPY_DISABLE_C_API;
    }
}

template <duccfft_loop implementation>
static void
wrap_fixed_cpp_ufunc(char **args, npy_intp const *dimensions,
                     npy_intp const *steps, void *func)
{
    wrap_cpp_ufunc(implementation, args, dimensions, steps, func);
}

static duccfft_loop duccfft_c2c_double_ptr = nullptr;
static duccfft_loop duccfft_c2c_float_ptr = nullptr;
static duccfft_loop duccfft_rfft_even_double_ptr = nullptr;
static duccfft_loop duccfft_rfft_even_float_ptr = nullptr;
static duccfft_loop duccfft_rfft_odd_double_ptr = nullptr;
static duccfft_loop duccfft_rfft_odd_float_ptr = nullptr;
static duccfft_loop duccfft_irfft_double_ptr = nullptr;
static duccfft_loop duccfft_irfft_float_ptr = nullptr;

#define DUCCFFT_DEFINE_WRAPPER(NAME)                                      \
    static void NAME##_wrapper(char **args,                              \
                               npy_intp const *dimensions,               \
                               npy_intp const *steps, void *func)         \
    {                                                                     \
        wrap_cpp_ufunc(NAME##_ptr, args, dimensions, steps, func);       \
    }

DUCCFFT_DEFINE_WRAPPER(duccfft_c2c_double)
DUCCFFT_DEFINE_WRAPPER(duccfft_c2c_float)
DUCCFFT_DEFINE_WRAPPER(duccfft_rfft_even_double)
DUCCFFT_DEFINE_WRAPPER(duccfft_rfft_even_float)
DUCCFFT_DEFINE_WRAPPER(duccfft_rfft_odd_double)
DUCCFFT_DEFINE_WRAPPER(duccfft_rfft_odd_float)
DUCCFFT_DEFINE_WRAPPER(duccfft_irfft_double)
DUCCFFT_DEFINE_WRAPPER(duccfft_irfft_float)

#undef DUCCFFT_DEFINE_WRAPPER
#undef DUCCFFT_DISPATCH_SIGNATURE

static bool ducc_forward = true;
static bool ducc_backward = false;

static PyUFuncGenericFunction fft_functions[] = {
    duccfft_c2c_double_wrapper,
    duccfft_c2c_float_wrapper,
    wrap_fixed_cpp_ufunc<ducc0::duccfft_impl::fft_loop<npy_longdouble>>,
};
static const char fft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};
static void *const fft_data[] = {
    static_cast<void *>(&ducc_forward),
    static_cast<void *>(&ducc_forward),
    static_cast<void *>(&ducc_forward)
};
static void *const ifft_data[] = {
    static_cast<void *>(&ducc_backward),
    static_cast<void *>(&ducc_backward),
    static_cast<void *>(&ducc_backward)
};

static PyUFuncGenericFunction rfft_n_even_functions[] = {
    duccfft_rfft_even_double_wrapper,
    duccfft_rfft_even_float_wrapper,
    wrap_fixed_cpp_ufunc<ducc0::duccfft_impl::rfft_n_even_loop<npy_longdouble>>
};
static PyUFuncGenericFunction rfft_n_odd_functions[] = {
    duccfft_rfft_odd_double_wrapper,
    duccfft_rfft_odd_float_wrapper,
    wrap_fixed_cpp_ufunc<ducc0::duccfft_impl::rfft_n_odd_loop<npy_longdouble>>
};
static const char rfft_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_FLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};

static PyUFuncGenericFunction irfft_functions[] = {
    duccfft_irfft_double_wrapper,
    duccfft_irfft_float_wrapper,
    wrap_fixed_cpp_ufunc<ducc0::duccfft_impl::irfft_loop<npy_longdouble>>
};
static const char irfft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE
};

static int
add_gufuncs(PyObject *dictionary)
{
    PyObject *f;

    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, fft_data, fft_types, 3, 2, 1, PyUFunc_None,
        "fft", "complex forward FFT\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "fft", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, ifft_data, fft_types, 3, 2, 1, PyUFunc_None,
        "ifft", "complex backward FFT\n", 0, "(m),()->(n)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "ifft", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_even_functions, NULL, rfft_types, 3, 2, 1, PyUFunc_None,
        "rfft_n_even", "real forward FFT for even n\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "rfft_n_even", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_odd_functions, NULL, rfft_types, 3, 2, 1, PyUFunc_None,
        "rfft_n_odd", "real forward FFT for odd n\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "rfft_n_odd", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        irfft_functions, NULL, irfft_types, 3, 2, 1, PyUFunc_None,
        "irfft", "real backward FFT\n", 0, "(m),()->(n)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "irfft", f);
    Py_DECREF(f);
    return 0;
}


using duccfft_nd_loop = void (*)(
    const void *, void *, const npy_intp *, const npy_intp *,
    const npy_intp *, const npy_intp *, npy_intp, npy_intp, bool, double);
static duccfft_nd_loop duccfft_native_nd_double_ptr = nullptr;
static duccfft_nd_loop duccfft_native_nd_float_ptr = nullptr;

static PyObject *
duccfft_c2c_nd(PyObject * /*module*/, PyObject *args)
{
    PyObject *input_obj, *output_obj, *axes_obj;
    int forward;
    double factor;
    if (!PyArg_ParseTuple(args, "OOOpd:c2c_nd", &input_obj, &output_obj,
                          &axes_obj, &forward, &factor)) {
        return nullptr;
    }
    if (!PyArray_Check(input_obj) || !PyArray_Check(output_obj) ||
        !PyTuple_Check(axes_obj)) {
        PyErr_SetString(PyExc_TypeError, "expected input/output ndarrays and axes tuple");
        return nullptr;
    }
    auto *in = reinterpret_cast<PyArrayObject *>(input_obj);
    auto *out = reinterpret_cast<PyArrayObject *>(output_obj);
    const int ndim = PyArray_NDIM(in);
    const int typenum = PyArray_TYPE(in);
    const npy_intp naxes = PyTuple_GET_SIZE(axes_obj);
    if (ndim < 2 || naxes < 2 || naxes > ndim ||
        PyArray_NDIM(out) != ndim || typenum != PyArray_TYPE(out) ||
        (typenum != NPY_CFLOAT && typenum != NPY_CDOUBLE) ||
        !PyArray_IS_C_CONTIGUOUS(in) || !PyArray_IS_C_CONTIGUOUS(out) ||
        !PyArray_ISALIGNED(in) || !PyArray_ISALIGNED(out) ||
        !PyArray_ISWRITEABLE(out)) {
        PyErr_SetString(PyExc_ValueError,
                        "native ND needs aligned contiguous matching complex arrays");
        return nullptr;
    }
    if (!std::isfinite(factor)) {
        PyErr_SetString(PyExc_ValueError, "nonfinite FFT scale");
        return nullptr;
    }
    for (int j = 0; j < ndim; ++j) {
        if (PyArray_DIMS(in)[j] != PyArray_DIMS(out)[j]) {
            PyErr_SetString(PyExc_ValueError, "native ND dimensions differ");
            return nullptr;
        }
    }
    std::vector<npy_intp> axes(static_cast<size_t>(naxes));
    for (npy_intp j = 0; j < naxes; ++j) {
        npy_intp axis = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(axes_obj, j));
        if (axis == -1 && PyErr_Occurred()) {
            return nullptr;
        }
        if (axis < 0 || axis >= ndim ||
            std::find(axes.begin(), axes.begin() + j, axis) != axes.begin() + j) {
            PyErr_SetString(PyExc_ValueError, "invalid or repeated native ND axis");
            return nullptr;
        }
        axes[static_cast<size_t>(j)] = axis;
    }
    auto transform = typenum == NPY_CFLOAT
        ? duccfft_native_nd_float_ptr : duccfft_native_nd_double_ptr;
    if (!transform) {
        PyErr_SetString(PyExc_RuntimeError, "native FFT CPU dispatch not initialized");
        return nullptr;
    }
    std::exception_ptr error;
    PyThreadState *thread_state = PyEval_SaveThread();
    try {
        transform(PyArray_DATA(in), PyArray_DATA(out), PyArray_DIMS(in),
                  PyArray_STRIDES(in), PyArray_STRIDES(out), axes.data(),
                  ndim, naxes, forward != 0, factor);
    }
    catch (...) {
        error = std::current_exception();
    }
    PyEval_RestoreThread(thread_state);
    if (error) {
        try {
            std::rethrow_exception(error);
        }
        catch (const std::bad_alloc &) {
            return PyErr_NoMemory();
        }
        catch (const std::exception &exc) {
            PyErr_SetString(PyExc_RuntimeError, exc.what());
            return nullptr;
        }
    }
    Py_RETURN_NONE;
}

static PyMethodDef duccfft_private_methods[] = {
    {"c2c_nd", duccfft_c2c_nd, METH_VARARGS,
     "Private single DUCC C2C transform across several axes."},
    {nullptr, nullptr, 0, nullptr}
};

static int module_loaded = 0;

static int
_duccfft_umath_exec(PyObject *m)
{
    if (module_loaded) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot load module more than once per process");
        return -1;
    }
    module_loaded = 1;

    if (PyArray_ImportNumPyAPI() < 0) {
        return -1;
    }
    if (PyUFunc_ImportUFuncAPI() < 0) {
        return -1;
    }
    if (npy_cpu_init() < 0) {
        return -1;
    }

    // Keep a private record of the target selected by this dispatcher.  The
    // FFT tests use this to verify runtime dispatch rather than only checking
    // that a transform succeeds.
    const char *dispatch_info[] = NPY_CPU_DISPATCH_INFO();
    if (PyModule_AddStringConstant(
                m, "__duccfft_cpu_dispatch_target__", dispatch_info[0]) < 0 ||
        PyModule_AddStringConstant(
                m, "__duccfft_cpu_dispatch_targets__", dispatch_info[1]) < 0) {
        return -1;
    }

    duccfft_c2c_double_ptr = NPY_CPU_DISPATCH_CALL(duccfft_c2c_double);
    duccfft_c2c_float_ptr = NPY_CPU_DISPATCH_CALL(duccfft_c2c_float);
    duccfft_rfft_even_double_ptr =
        NPY_CPU_DISPATCH_CALL(duccfft_rfft_even_double);
    duccfft_rfft_even_float_ptr =
        NPY_CPU_DISPATCH_CALL(duccfft_rfft_even_float);
    duccfft_rfft_odd_double_ptr =
        NPY_CPU_DISPATCH_CALL(duccfft_rfft_odd_double);
    duccfft_rfft_odd_float_ptr =
        NPY_CPU_DISPATCH_CALL(duccfft_rfft_odd_float);
    duccfft_irfft_double_ptr = NPY_CPU_DISPATCH_CALL(duccfft_irfft_double);
    duccfft_irfft_float_ptr = NPY_CPU_DISPATCH_CALL(duccfft_irfft_float);
    duccfft_native_nd_double_ptr = NPY_CPU_DISPATCH_CALL(duccfft_native_nd_double);
    duccfft_native_nd_float_ptr = NPY_CPU_DISPATCH_CALL(duccfft_native_nd_float);

    if (add_gufuncs(PyModule_GetDict(m)) < 0) {
        return -1;
    }
    return 0;
}

static struct PyModuleDef_Slot _duccfft_umath_slots[] = {
    {Py_mod_exec, (void*)_duccfft_umath_exec},
#if PY_VERSION_HEX >= 0x030c00f0
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0 && (!defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030d0000)
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_duccfft_umath",
    NULL,
    0,
    duccfft_private_methods,
    _duccfft_umath_slots,
};

PyMODINIT_FUNC
PyInit__duccfft_umath(void)
{
    return PyModuleDef_Init(&moduledef);
}
