/* Use DUCC's namespace macro to isolate the target-sensitive FFT headers. */
#ifndef NUMPY_FFT_DUCCFFT_TARGET_H_
#define NUMPY_FFT_DUCCFFT_TARGET_H_

/* These support interfaces have one out-of-line implementation in DUCC's
 * default namespace.  Import them into each target namespace before compiling
 * the target-specific FFT headers. */
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/threading.h"

#undef DUCC0_NAMESPACE
#include "_duccfft_namespace.h"

#if defined(NPY_MTARGETS_CURRENT) || defined(NPY_MTARGETS_BASELINE)
namespace DUCC0_NAMESPACE {
using namespace ::ducc0;
}
#endif

#include "_duccfft_impl.h"

#endif  // NUMPY_FFT_DUCCFFT_TARGET_H_
