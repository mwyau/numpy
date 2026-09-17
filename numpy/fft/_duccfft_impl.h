/*
 * Internal one-dimensional FFT implementation shared by the Python module
 * and the NumPy CPU-dispatch targets.
 *
 * This file deliberately contains no module initialization or gufunc
 * registration.  The dispatch translation unit includes it once for each
 * generated target, while the module translation unit uses it only for the
 * baseline long-double loops.
 */
#ifndef NUMPY_FFT_DUCCFFT_IMPL_H_
#define NUMPY_FFT_DUCCFFT_IMPL_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <assert.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#include "numpy/ndarraytypes.h"

#include "ducc0/fft/fftnd_impl.h"

namespace duccfft_impl {

using ducc_shape_t = ducc0::fmav_info::shape_t;
using ducc_stride_t = ducc0::fmav_info::stride_t;

/* DUCC reinterprets NumPy's std::complex storage as Cmplx<T>. */
template <typename T>
static inline void
check_complex_representation()
{
    static_assert(
        sizeof(ducc0::Cmplx<T>) == sizeof(std::complex<T>),
        "DUCC and NumPy complex values must have the same size");
    static_assert(
        alignof(ducc0::Cmplx<T>) == alignof(std::complex<T>),
        "DUCC and NumPy complex values must have the same alignment");
    static_assert(
        std::is_trivially_copyable_v<ducc0::Cmplx<T>>,
        "DUCC complex values must be trivially copyable");
    static_assert(
        std::is_trivially_copyable_v<std::complex<T>>,
        "NumPy complex values must be trivially copyable");
}

template <typename T>
static inline T
load_unaligned(const char *ptr)
{
    T value;
    std::memcpy(&value, ptr, sizeof(T));
    return value;
}

template <typename T>
static inline void
copy_input(char *in, npy_intp step, size_t nin, T *buffer, size_t n)
{
    size_t ncopy = nin < n ? nin : n;
    for (size_t i = 0; i < ncopy; ++i, in += step) {
        buffer[i] = load_unaligned<T>(in);
    }
    for (size_t i = ncopy; i < n; ++i) {
        buffer[i] = T(0);
    }
}

template <typename T>
static inline void
copy_output(const T *buffer, char *out, npy_intp step, size_t n)
{
    for (size_t i = 0; i < n; ++i, out += step) {
        std::memcpy(out, &buffer[i], sizeof(T));
    }
}

template <typename T>
static inline bool
can_use_direct_view(const void *ptr, npy_intp outer_stride,
                    npy_intp inner_stride)
{
    return reinterpret_cast<std::uintptr_t>(ptr) % alignof(T) == 0 &&
           outer_stride % static_cast<npy_intp>(sizeof(T)) == 0 &&
           inner_stride % static_cast<npy_intp>(sizeof(T)) == 0;
}

static bool
memory_range(const char *ptr, size_t n_outer, size_t n_inner,
             npy_intp outer_stride, npy_intp inner_stride, size_t itemsize,
             std::uintptr_t *begin, std::uintptr_t *end)
{
    std::uintptr_t low = reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t high = low;
    const auto max_uint = std::numeric_limits<std::uintptr_t>::max();

    auto add_extent = [&](npy_intp stride, size_t size) {
        if (size <= 1 || stride == 0) {
            return true;
        }
        bool negative = stride < 0;
        std::uintptr_t magnitude = negative
            ? std::uintptr_t(-(stride + 1)) + 1
            : std::uintptr_t(stride);
        size_t extent = size - 1;
        if (magnitude != 0 && extent > max_uint / magnitude) {
            return false;
        }
        std::uintptr_t offset = extent * magnitude;
        if (negative) {
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
        return true;
    };

    if (!add_extent(outer_stride, n_outer) ||
        !add_extent(inner_stride, n_inner) ||
        high > max_uint - itemsize) {
        return false;
    }
    *begin = low;
    *end = high + itemsize;
    return true;
}

/* DUCC only permits separate input and output views when they do not overlap.
 * The bounds check is deliberately conservative: strided views with holes may
 * be classified as overlapping and use the staged path instead.
 */
template <typename Tin, typename Tout>
static bool
may_overlap(const char *input, const char *output, size_t n_outer,
            size_t n_input, size_t n_output, npy_intp input_outer_stride,
            npy_intp input_inner_stride, npy_intp output_outer_stride,
            npy_intp output_inner_stride)
{
    if (n_outer == 0 || n_input == 0 || n_output == 0) {
        return false;
    }
    std::uintptr_t input_begin, input_end, output_begin, output_end;
    if (!memory_range(input, n_outer, n_input, input_outer_stride,
                      input_inner_stride, sizeof(Tin), &input_begin,
                      &input_end) ||
        !memory_range(output, n_outer, n_output, output_outer_stride,
                      output_inner_stride, sizeof(Tout), &output_begin,
                      &output_end)) {
        return true;
    }
    return input_begin < output_end && output_begin < input_end;
}

template <typename T>
static inline ptrdiff_t
element_stride(npy_intp byte_stride)
{
    return static_cast<ptrdiff_t>(
        byte_stride / static_cast<npy_intp>(sizeof(T)));
}

/* Execute independent c2c rows with one local DUCC plan.  This path is for
 * staged layouts that cannot use DUCC's generalized direct view.  Keeping the
 * plan local preserves the cache-off and single-threaded policy while avoiding
 * one plan construction per row.
 */
template <typename T>
static bool
run_local_c2c(char *ip, char *fp, char *op, size_t n_outer,
              size_t nin_available, size_t nin, size_t nout,
              npy_intp si, npy_intp sf, npy_intp so,
              npy_intp step_in, npy_intp step_out, bool forward)
{
    using complex_t = ducc0::Cmplx<T>;
    check_complex_representation<T>();

    // A row loop cannot preserve NumPy's snapshot semantics for overlapping
    // views (including zero-stride rows); leave those cases on run_staged().
    if (may_overlap<complex_t, complex_t>(
            ip, op, n_outer, nin_available < nin ? nin_available : nin,
            nout, si, step_in, so, step_out)) {
        return false;
    }

    size_t n_input = nin_available < nin ? nin_available : nin;
    ducc0::pocketfft_c<T> plan(nout, false);
    ducc0::aligned_array<complex_t> scratch(plan.bufsize());

    const bool output_is_contiguous =
        step_out == static_cast<npy_intp>(sizeof(complex_t)) &&
        can_use_direct_view<complex_t>(op, so, step_out);
    ducc0::aligned_array<complex_t> buffer(output_is_contiguous ? 0 : nout);

    for (size_t i = 0; i < n_outer; ++i,
            ip += si, fp += sf, op += so) {
        if (output_is_contiguous) {
            auto *work = reinterpret_cast<complex_t *>(op);
            copy_input(ip, step_in, n_input, work, nout);
            plan.exec_copyback(work, scratch.data(),
                               load_unaligned<T>(fp), forward, 1);
        }
        else {
            copy_input(ip, step_in, n_input, buffer.data(), nout);
            auto *result = plan.exec(buffer.data(), scratch.data(),
                                     load_unaligned<T>(fp), forward, 1);
            copy_output(result, op, step_out, nout);
        }
    }
    return true;
}

template <typename Tin, typename Tout, typename Scale, typename Transform>
static bool
run_direct(char *ip, char *fp, char *op, size_t n_outer,
           size_t nin, size_t nout, npy_intp si, npy_intp so,
           npy_intp step_in, npy_intp step_out, bool allow_inplace,
           bool use_one_dimensional,
           Transform&& transform)
{
    check_complex_representation<Scale>();
    // DUCC can consume NumPy's strided storage directly when the byte strides
    // are element strides, the data is aligned, and the views are compatible.
    if (!can_use_direct_view<Tin>(ip, si, step_in) ||
        !can_use_direct_view<Tout>(op, so, step_out)) {
        return false;
    }

    bool overlap = may_overlap<Tin, Tout>(
        ip, op, n_outer, nin, nout, si, step_in, so, step_out);
    // DUCC permits c2c in-place operation only for the same view. Real/complex
    // transforms always take the staged path when the views overlap.
    bool exact_inplace = allow_inplace &&
        ip == op && si == so && step_in == step_out;
    if (overlap && !exact_inplace) {
        return false;
    }

    bool one_dimensional = use_one_dimensional && n_outer == 1;
    ducc_shape_t shape_in = one_dimensional
        ? ducc_shape_t{nin} : ducc_shape_t{n_outer, nin};
    ducc_shape_t shape_out = one_dimensional
        ? ducc_shape_t{nout} : ducc_shape_t{n_outer, nout};
    ducc_stride_t strides_in{
        element_stride<Tin>(si), element_stride<Tin>(step_in)};
    ducc_stride_t strides_out{
        element_stride<Tout>(so), element_stride<Tout>(step_out)};
    if (one_dimensional) {
        strides_in = ducc_stride_t{element_stride<Tin>(step_in)};
        strides_out = ducc_stride_t{element_stride<Tout>(step_out)};
    }
    ducc_shape_t axes = one_dimensional ? ducc_shape_t{0} : ducc_shape_t{1};

    auto in = ducc0::cfmav<Tin>(
        reinterpret_cast<const Tin *>(ip), shape_in, strides_in);
    auto out = ducc0::vfmav<Tout>(
        reinterpret_cast<Tout *>(op), shape_out, strides_out);
    transform(in, out, axes, load_unaligned<Scale>(fp));
    return true;
}

template <typename Tin, typename Tout, typename Scale, typename Transform>
static void
run_staged(char *ip, char *fp, char *op, size_t n_outer,
           size_t nin_available, size_t nin, size_t nout,
           npy_intp si, npy_intp sf, npy_intp so,
           npy_intp step_in, npy_intp step_out, Transform&& transform)
{
    check_complex_representation<Scale>();
    size_t n_input = nin_available < nin ? nin_available : nin;
    // Preserve all input transforms before writing an overlapping output.
    bool overlap = may_overlap<Tin, Tout>(
        ip, op, n_outer, n_input, nout, si, step_in, so, step_out);
    size_t input_size = nin;
    if (overlap) {
        if (nin != 0 && n_outer > std::numeric_limits<size_t>::max() / nin) {
            throw std::bad_alloc();
        }
        input_size = n_outer * nin;
    }
    std::vector<Tin> input(input_size);
    std::vector<Tout> output(nout);
    ducc_shape_t shape_in{nin};
    ducc_shape_t shape_out{nout};
    ducc_stride_t stride{1};
    ducc_shape_t axes{0};
    auto in = ducc0::cfmav<Tin>(input.data(), shape_in, stride);
    auto out = ducc0::vfmav<Tout>(output.data(), shape_out, stride);

    if (overlap) {
        char *input_ptr = ip;
        for (size_t i = 0; i < n_outer; ++i) {
            copy_input(input_ptr, step_in, n_input,
                       input.data() + i * nin, nin);
            input_ptr += si;
        }
    }

    for (size_t i = 0; i < n_outer; ++i, ip += si, fp += sf, op += so) {
        if (!overlap) {
            copy_input(ip, step_in, n_input, input.data(), nin);
        }
        auto input_view = overlap
            ? ducc0::cfmav<Tin>(input.data() + i * nin, shape_in, stride)
            : in;
        transform(input_view, out, axes, load_unaligned<Scale>(fp));
        copy_output(output.data(), op, step_out, nout);
    }
}

template <typename T>
static void
fft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *func)
{
    using complex_t = std::complex<T>;

    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = static_cast<size_t>(dimensions[0]);
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = static_cast<size_t>(dimensions[1]);
    size_t nout = static_cast<size_t>(dimensions[2]);
    npy_intp step_in = steps[3], step_out = steps[4];
    bool forward = *static_cast<bool *>(func);

    assert(nout > 0);
    if (n_outer == 0) {
        return;
    }

    auto transform = [forward](const auto& in, auto& out,
                               const ducc_shape_t& axes, T fct) {
        ducc0::c2c(in, out, axes, forward, fct, 1);
    };

    if (sf == 0 && nin >= nout &&
        run_direct<complex_t, complex_t, T>(
            ip, fp, op, n_outer, nout, nout,
            si, so, step_in, step_out, true, true, transform)) {
        return;
    }

    if (run_local_c2c<T>(ip, fp, op, n_outer, nin, nout, nout,
                         si, sf, so, step_in, step_out, forward)) {
        return;
    }

    run_staged<complex_t, complex_t, T>(
        ip, fp, op, n_outer, nin, nout, nout,
        si, sf, so, step_in, step_out, transform);
}

template <typename T>
static void
rfft_impl(char **args, npy_intp const *dimensions, npy_intp const *steps,
          size_t npts)
{
    using complex_t = std::complex<T>;

    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = static_cast<size_t>(dimensions[0]);
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = static_cast<size_t>(dimensions[1]);
    size_t nout = static_cast<size_t>(dimensions[2]);
    npy_intp step_in = steps[3], step_out = steps[4];

    assert(nout > 0 && nout == npts / 2 + 1);
    if (n_outer == 0) {
        return;
    }

    auto transform = [](const auto& in, auto& out,
                        const ducc_shape_t& axes, T fct) {
        ducc0::r2c(in, out, axes, true, fct, 1);
    };

    if (sf == 0 && nin >= npts &&
        run_direct<T, complex_t, T>(
            ip, fp, op, n_outer, npts, nout,
            si, so, step_in, step_out, false, false, transform)) {
        return;
    }

    run_staged<T, complex_t, T>(
        ip, fp, op, n_outer, nin, npts, nout,
        si, sf, so, step_in, step_out, transform);
}

template <typename T>
static void
rfft_n_even_loop(char **args, npy_intp const *dimensions,
                 npy_intp const *steps, void * /*func*/)
{
    size_t nout = static_cast<size_t>(dimensions[2]);
    assert(nout > 0);
    rfft_impl<T>(args, dimensions, steps, 2 * nout - 2);
}

template <typename T>
static void
rfft_n_odd_loop(char **args, npy_intp const *dimensions,
                npy_intp const *steps, void * /*func*/)
{
    size_t nout = static_cast<size_t>(dimensions[2]);
    assert(nout > 0);
    rfft_impl<T>(args, dimensions, steps, 2 * nout - 1);
}

template <typename T>
static void
irfft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void * /*func*/)
{
    using complex_t = std::complex<T>;

    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = static_cast<size_t>(dimensions[0]);
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = static_cast<size_t>(dimensions[1]);
    size_t nout = static_cast<size_t>(dimensions[2]);
    npy_intp step_in = steps[3], step_out = steps[4];
    size_t npts_in = nout / 2 + 1;

    assert(nout > 0);
    if (n_outer == 0) {
        return;
    }

    auto transform = [](const auto& in, auto& out,
                        const ducc_shape_t& axes, T fct) {
        ducc0::c2r(in, out, axes, false, fct, 1);
    };

    if (sf == 0 && nin >= npts_in &&
        run_direct<complex_t, T, T>(
            ip, fp, op, n_outer, npts_in, nout,
            si, so, step_in, step_out, false, false, transform)) {
        return;
    }

    run_staged<complex_t, T, T>(
        ip, fp, op, n_outer, nin, npts_in, nout,
        si, sf, so, step_in, step_out, transform);
}

}  // namespace duccfft_impl

#endif  // NUMPY_FFT_DUCCFFT_IMPL_H_
