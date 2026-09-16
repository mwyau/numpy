largefile_define = []
if host_machine.system() == 'aix' or host_machine.system() == 'AIX'
  largefile_define += '-D_LARGE_FILES'
endif

# Keep DUCC single-threaded and disable its plan cache to preserve
# numpy.fft's current execution and resource policies.  numpy.fft does not
# expose a worker-count API and NumPy deliberately does not cache FFT plans.
duccfft_args = [
  '-DDUCC0_NO_FFT_CACHE',
  '-DDUCC0_NO_LOWLEVEL_THREADING',
]

duccfft_sources = [
  'ducc0/infra/mav.cc',
  'ducc0/infra/string_utils.cc',
  'ducc0/infra/threading.cc',
]

duccfft_dispatch_cpp_args = largefile_define + duccfft_args + max_opt
duccfft_dispatch_cpp_args += cpp.get_supported_arguments('-fvisibility=hidden')

# Compile the FFT implementation once for each NumPy CPU target.  The
# generated dispatcher header is private to this static library; the Python
# module below only contains stable wrappers and selects these entry points at
# module initialization.
mod_features = import('features')
duccfft_dispatch = mod_features.multi_targets(
  '_duccfft.dispatch.h',
  '_duccfft.dispatch.cpp',
  dispatch: [X86_V4, X86_V3, X86_V2],
  baseline: CPU_BASELINE,
  prefix: 'NPY_',
  dependencies: [py_dep, np_core_dep],
  cpp_args: duccfft_dispatch_cpp_args,
  include_directories: include_directories('ducc0'),
)

py.extension_module('_duccfft_umath',
  # CPU feature helpers are hidden per extension; keep a private copy here,
  # as the _simd extension does, rather than depending on _multiarray_umath.
  ['_duccfft_umath.cpp', '../_core/src/common/npy_cpu_features.c']
    + duccfft_sources,
  c_args: c_args_common + largefile_define,
  cpp_args: largefile_define + duccfft_args,
  include_directories: include_directories('ducc0'),
  dependencies: np_core_dep,
  link_with: duccfft_dispatch.static_lib('_duccfft_dispatch'),
  install: true,
  subdir: 'numpy/fft',
  limited_api: py_limited_api,
)

py.install_sources(
  [
    '__init__.py',
    '__init__.pyi',
    '_ducc.py',
    '_ducc.pyi',
    '_helper.py',
    '_helper.pyi',
  ],
  subdir: 'numpy/fft'
)

py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_helper.py',
    'tests/test_duccfft.py',
  ],
  subdir: 'numpy/fft/tests',
  install_tag: 'tests'
)
