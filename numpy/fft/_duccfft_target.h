/*
 * Include the DUCC FFT headers in a target-specific namespace.
 *
 * DUCC's FFT implementation is header-only, but its support sources are
 * compiled once in the extension.  Include the target-sensitive header-only
 * DUCC dependency graph in a namespace named for the NumPy CPU target.  The
 * isolation is intentionally broader than SIMD alone: target compile flags
 * also affect inline/template support code, which must not be emitted as
 * ordinary weak/COMDAT symbols shared by different targets.
 *
 * Maintenance warning: when updating vendored DUCC, rerun the multi-target
 * symbol/link audit.  This header intentionally recreates the target-sensitive
 * header dependency graph in CPU-target-specific namespaces; new headers or
 * out-of-line dependencies can invalidate these isolation assumptions.
 */
#ifndef NUMPY_FFT_DUCCFFT_TARGET_H_
#define NUMPY_FFT_DUCCFFT_TARGET_H_

#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/threading.h"

#define DUCCFFT_TARGET_JOIN_IMPL(a, b) a##b
#define DUCCFFT_TARGET_JOIN(a, b) DUCCFFT_TARGET_JOIN_IMPL(a, b)

#if defined(NPY_MTARGETS_CURRENT)
#define DUCCFFT_TARGET_ID NPY_MTARGETS_CURRENT
#else
#define DUCCFFT_TARGET_ID baseline
#endif
#define DUCCFFT_TARGET_NAMESPACE \
    DUCCFFT_TARGET_JOIN(ducc0_, DUCCFFT_TARGET_ID)

namespace DUCCFFT_TARGET_NAMESPACE {
using namespace ::ducc0;
}

/*
 * DUCC's include guards are namespace-blind.  Reset the header-only support
 * guards so each target gets its own definitions.  The support .cc files are
 * deliberately not included here; Meson links their ordinary definitions
 * exactly once into the extension.  In particular, the target audit checks
 * that this broader isolation does not leave target-local out-of-line support
 * declarations unresolved.
 */
#undef DUCC0_SIMD_H
#undef DUCC0_MAV_H
#undef DUCC0_ALIGNED_ARRAY_H
#undef DUCC0_CMPLX_H
#undef DUCC0_UNITY_ROOTS_H
#undef DUCC0_ERROR_HANDLING_H
#undef DUCC0_MISC_UTILS_H

#define ducc0 DUCCFFT_TARGET_NAMESPACE
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/mav.h"
#include "ducc0/math/cmplx.h"
#include "ducc0/math/unity_roots.h"
#include "_duccfft_impl.h"
#undef ducc0

#undef DUCCFFT_TARGET_NAMESPACE
#undef DUCCFFT_TARGET_ID
#undef DUCCFFT_TARGET_JOIN
#undef DUCCFFT_TARGET_JOIN_IMPL

#endif  // NUMPY_FFT_DUCCFFT_TARGET_H_
