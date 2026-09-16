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
        duccfft_impl::IMPLEMENTATION(args, dimensions, steps, func);       \
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
