/* Target-specific DUCC FFT loop entry points for NumPy's CPU dispatcher. */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "npy_cpu_dispatch.h"
#include "_duccfft.dispatch.h"
#include "_duccfft_target.h"

#define DUCCFFT_DEFINE_ENTRY(NAME, IMPLEMENTATION)                         \
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(NAME)(                       \
            char **args, npy_intp const *dimensions,                     \
            npy_intp const *steps, void *func)                            \
    {                                                                      \
        DUCC0_NAMESPACE::duccfft_impl::IMPLEMENTATION(                    \
                args, dimensions, steps, func);                            \
    }

DUCCFFT_DEFINE_ENTRY(duccfft_c2c_double, fft_loop<npy_double>)
DUCCFFT_DEFINE_ENTRY(duccfft_c2c_float, fft_loop<npy_float>)
DUCCFFT_DEFINE_ENTRY(duccfft_rfft_even_double,
                     rfft_n_even_loop<npy_double>)
DUCCFFT_DEFINE_ENTRY(duccfft_rfft_even_float,
                     rfft_n_even_loop<npy_float>)
DUCCFFT_DEFINE_ENTRY(duccfft_rfft_odd_double,
                     rfft_n_odd_loop<npy_double>)
DUCCFFT_DEFINE_ENTRY(duccfft_rfft_odd_float,
                     rfft_n_odd_loop<npy_float>)
DUCCFFT_DEFINE_ENTRY(duccfft_irfft_double, irfft_loop<npy_double>)
DUCCFFT_DEFINE_ENTRY(duccfft_irfft_float, irfft_loop<npy_float>)

#undef DUCCFFT_DEFINE_ENTRY

/* Native C2C across multiple FFT axes; input/output are NumPy arrays checked
 * for matching, aligned, contiguous complex storage by the Python wrapper. */
#define DUCCFFT_DEFINE_NATIVE_ND(NAME, TYPE)                               \
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(NAME)(                        \
            const void *input, void *output, const npy_intp *shape,         \
            const npy_intp *istrides, const npy_intp *ostrides,             \
            const npy_intp *axes, npy_intp ndim, npy_intp naxes,           \
            bool forward, double factor)                                   \
    {                                                                      \
        DUCC0_NAMESPACE::duccfft_impl::native_c2c_nd<TYPE>(                \
            input, output, shape, istrides, ostrides, axes, ndim, naxes,    \
            forward, factor);                                              \
    }

DUCCFFT_DEFINE_NATIVE_ND(duccfft_native_nd_double, npy_double)
DUCCFFT_DEFINE_NATIVE_ND(duccfft_native_nd_float, npy_float)
#undef DUCCFFT_DEFINE_NATIVE_ND

