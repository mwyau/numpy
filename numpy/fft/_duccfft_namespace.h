/* Map each NumPy multi-target translation unit to DUCC's native namespace. */
#ifndef NUMPY_FFT_DUCCFFT_NAMESPACE_H_
#define NUMPY_FFT_DUCCFFT_NAMESPACE_H_

#define DUCCFFT_NAMESPACE_JOIN_IMPL(a, b) a##b
#define DUCCFFT_NAMESPACE_JOIN(a, b) DUCCFFT_NAMESPACE_JOIN_IMPL(a, b)

#ifndef DUCC0_NAMESPACE
#if defined(NPY_MTARGETS_CURRENT)
#define DUCC0_NAMESPACE DUCCFFT_NAMESPACE_JOIN(ducc0_, NPY_MTARGETS_CURRENT)
#elif defined(NPY_MTARGETS_BASELINE)
#define DUCC0_NAMESPACE ducc0_baseline
#else
#define DUCC0_NAMESPACE ducc0
#endif
#endif

#endif  // NUMPY_FFT_DUCCFFT_NAMESPACE_H_
