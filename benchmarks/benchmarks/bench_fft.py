import os

import numpy as np


def _check_cpu_target():
    """Assert the selected FFT kernel inside ASV's timed worker process."""
    expected = os.environ.get("DUCCFFT_ASV_EXPECTED_TARGET")
    if expected is None:
        return
    from numpy.fft import _duccfft_umath
    selected = _duccfft_umath.__duccfft_cpu_dispatch_target__
    if selected != expected:
        raise RuntimeError(
            f"ASV worker DUCC target mismatch: {selected!r} != {expected!r}"
        )


from .common import Benchmark


def _random_real(shape, dtype):
    rng = np.random.default_rng(1234)
    return rng.standard_normal(shape).astype(dtype)


def _random_complex(shape, dtype):
    rng = np.random.default_rng(1234)
    values = np.empty(shape, dtype=dtype)
    values.real = rng.standard_normal(shape)
    values.imag = rng.standard_normal(shape)
    return values


class FFT1D(Benchmark):
    params = [
        [32, 100, 257, 313, 1000, 1024, 4093, 4094, 4095, 4096, 4097,
         6144, 65536],
        ['complex64', 'complex128'],
        ['fft', 'ifft'],
    ]
    param_names = ['size', 'dtype', 'operation']

    def setup(self, size, dtype, operation):
        _check_cpu_target()
        self.x = _random_complex((size,), dtype)
        self.func = getattr(np.fft, operation)

    def time_transform(self, size, dtype, operation):
        self.func(self.x)


class RealFFT1D(Benchmark):
    params = [
        [7, 32, 100, 313, 1000, 1023, 1024, 4095, 4096, 4097],
        ['float32', 'float64'],
        ['rfft', 'irfft'],
    ]
    param_names = ['size', 'dtype', 'operation']

    def setup(self, size, dtype, operation):
        _check_cpu_target()
        self.func = getattr(np.fft, operation)
        self.kwargs = {'n': size} if operation == 'irfft' else {}
        self.x = _random_real((size,), dtype)
        if operation == 'irfft':
            self.x = np.fft.rfft(self.x)

    def time_transform(self, size, dtype, operation):
        self.func(self.x, **self.kwargs)


class FFTBatched(Benchmark):
    params = [
        [1, 2, 4, 8, 16, 64],
        [4095, 4096],
        ['complex64', 'complex128'],
    ]
    param_names = ['batch', 'size', 'dtype']

    def setup(self, batch, size, dtype):
        _check_cpu_target()
        self.x = _random_complex((batch, size), dtype)

    def time_fft(self, batch, size, dtype):
        np.fft.fft(self.x, axis=-1)


class RealFFTBatched(Benchmark):
    params = [
        [1, 4, 16, 64],
        [4095, 4096],
        ['rfft', 'irfft'],
    ]
    param_names = ['batch', 'size', 'operation']

    def setup(self, batch, size, operation):
        _check_cpu_target()
        self.x = _random_real((batch, size), 'float64')
        self.func = getattr(np.fft, operation)
        self.kwargs = {'axis': -1}
        if operation == 'irfft':
            self.x = np.fft.rfft(self.x, axis=-1)
            self.kwargs['n'] = size

    def time_transform(self, batch, size, operation):
        self.func(self.x, **self.kwargs)


class FFTND(Benchmark):
    params = [
        [(64, 64), (313, 100), (256, 256), (16, 17, 31)],
        ['complex64', 'complex128'],
        ['fftn', 'ifftn'],
    ]
    param_names = ['shape', 'dtype', 'operation']

    def setup(self, shape, dtype, operation):
        _check_cpu_target()
        self.x = _random_complex(shape, dtype)
        self.func = getattr(np.fft, operation)

    def time_transform(self, shape, dtype, operation):
        self.func(self.x)


class RealFFTND(Benchmark):
    params = [
        [(64, 64), (16, 17, 31)],
        ['float32', 'float64'],
        ['rfftn', 'irfftn'],
    ]
    param_names = ['shape', 'dtype', 'operation']

    def setup(self, shape, dtype, operation):
        _check_cpu_target()
        self.func = getattr(np.fft, operation)
        self.kwargs = ({'s': shape, 'axes': tuple(range(len(shape)))}
                       if operation == 'irfftn' else {})
        self.x = _random_real(shape, dtype)
        if operation == 'irfftn':
            self.x = np.fft.rfftn(self.x)

    def time_transform(self, shape, dtype, operation):
        self.func(self.x, **self.kwargs)


class FFTStrided(Benchmark):
    params = [
        ['contiguous', 'fortran', 'positive_stride', 'negative_stride'],
        ['complex128_fftn', 'float64_rfftn'],
    ]
    param_names = ['layout', 'transform']

    def setup(self, layout, transform):
        _check_cpu_target()
        if transform == 'complex128_fftn':
            dtype = 'complex128'
            self.func = np.fft.fftn
            make_input = _random_complex
        else:
            dtype = 'float64'
            self.func = np.fft.rfftn
            make_input = _random_real

        shape = (32, 64)
        base = make_input(shape, dtype)
        if layout == 'contiguous':
            self.x = base
        elif layout == 'fortran':
            self.x = np.asfortranarray(base)
        elif layout == 'positive_stride':
            storage = np.empty((shape[0], 2 * shape[1]), dtype=dtype)
            self.x = storage[:, ::2]
            self.x[...] = base
        else:
            storage = np.empty(shape, dtype=dtype)
            self.x = storage[:, ::-1]
            self.x[...] = base
        self.kwargs = {'axes': (0, 1)}

    def time_transform(self, layout, transform):
        self.func(self.x, **self.kwargs)


class FFTOut(Benchmark):
    params = [
        [
            'fft_c128_64x4095',
            'fft_c64_64x4096',
            'rfft_f64_64x4095',
            'irfft_c128_64x4096',
            'fftn_c128_128x256',
            'rfftn_f64_64x128',
        ],
        ['default', 'out'],
    ]
    param_names = ['workload', 'mode']

    workloads = {
        'fft_c128_64x4095': (
            np.fft.fft, (64, 4095), 'complex128', (64, 4095), 'complex128',
            {'axis': -1}, 'complex',
        ),
        'fft_c64_64x4096': (
            np.fft.fft, (64, 4096), 'complex64', (64, 4096), 'complex64',
            {'axis': -1}, 'complex',
        ),
        'rfft_f64_64x4095': (
            np.fft.rfft, (64, 4095), 'float64', (64, 2048), 'complex128',
            {'axis': -1}, 'real',
        ),
        'irfft_c128_64x4096': (
            np.fft.irfft, (64, 2049), 'complex128', (64, 4096), 'float64',
            {'axis': -1, 'n': 4096}, 'spectrum',
        ),
        'fftn_c128_128x256': (
            np.fft.fftn, (128, 256), 'complex128', (128, 256), 'complex128',
            {}, 'complex',
        ),
        'rfftn_f64_64x128': (
            np.fft.rfftn, (64, 128), 'float64', (64, 65), 'complex128',
            {}, 'real',
        ),
    }

    def setup(self, workload, mode):
        _check_cpu_target()
        (self.func, input_shape, input_dtype, output_shape, output_dtype,
         self.kwargs, input_kind) = self.workloads[workload]
        if input_kind == 'complex':
            self.x = _random_complex(input_shape, input_dtype)
        elif input_kind == 'real':
            self.x = _random_real(input_shape, input_dtype)
        else:
            real_shape = input_shape[:-1] + (self.kwargs['n'],)
            real_input = _random_real(real_shape, 'float64')
            self.x = np.fft.rfft(real_input, axis=-1)

        self.out = np.empty(output_shape, dtype=output_dtype)
        if mode == 'out':
            try:
                self.func(self.x, out=self.out, **self.kwargs)
            except TypeError as exc:
                raise NotImplementedError(
                    'This NumPy version does not support FFT out=.'
                ) from exc
        else:
            # Equalize initial plan-cache and allocation warmup for both modes.
            self.func(self.x, **self.kwargs)

    def time_transform(self, workload, mode):
        if mode == 'out':
            self.func(self.x, out=self.out, **self.kwargs)
        else:
            self.func(self.x, **self.kwargs)


class FFTResize(Benchmark):
    params = [
        [1000, 1024, 1031],
        ['complex128_fft', 'float64_rfft'],
    ]
    param_names = ['size', 'transform']

    def setup(self, size, transform):
        _check_cpu_target()
        if transform == 'complex128_fft':
            self.x = _random_complex((1024,), 'complex128')
            self.func = np.fft.fft
        else:
            self.x = _random_real((1024,), 'float64')
            self.func = np.fft.rfft
        self.kwargs = {'n': size}

    def time_transform(self, size, transform):
        self.func(self.x, **self.kwargs)


class FFTSmallOverhead(Benchmark):
    params = [
        [1, 2, 8, 16],
        ['complex64', 'complex128'],
    ]
    param_names = ['size', 'dtype']

    def setup(self, size, dtype):
        _check_cpu_target()
        self.x = _random_complex((size,), dtype)

    def time_fft(self, size, dtype):
        np.fft.fft(self.x)
