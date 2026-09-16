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

#include <exception>
#include <new>

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
    wrap_fixed_cpp_ufunc<duccfft_impl::fft_loop<npy_longdouble>>,
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
    wrap_fixed_cpp_ufunc<duccfft_impl::rfft_n_even_loop<npy_longdouble>>
};
static PyUFuncGenericFunction rfft_n_odd_functions[] = {
    duccfft_rfft_odd_double_wrapper,
    duccfft_rfft_odd_float_wrapper,
    wrap_fixed_cpp_ufunc<duccfft_impl::rfft_n_odd_loop<npy_longdouble>>
};
static const char rfft_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_FLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};

static PyUFuncGenericFunction irfft_functions[] = {
    duccfft_irfft_double_wrapper,
    duccfft_irfft_float_wrapper,
    wrap_fixed_cpp_ufunc<duccfft_impl::irfft_loop<npy_longdouble>>
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
    NULL,
    _duccfft_umath_slots,
};

PyMODINIT_FUNC
PyInit__duccfft_umath(void)
{
    return PyModuleDef_Init(&moduledef);
}
