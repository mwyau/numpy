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

py.extension_module('_duccfft_umath',
  ['_duccfft_umath.cpp'] + duccfft_sources,
  cpp_args: largefile_define + duccfft_args,
  include_directories: include_directories('ducc0'),
  dependencies: np_core_dep,
  install: true,
  subdir: 'numpy/fft',
  limited_api: py_limited_api,
)

py.extension_module('_ducc_nd_umath',
  ['_ducc_nd_umath.cpp'] + duccfft_sources,
  cpp_args: largefile_define + duccfft_args,
  include_directories: include_directories('ducc0'),
  dependencies: np_core_dep,
  install: true,
  subdir: 'numpy/fft',
  limited_api: py_limited_api,
)

py.install_sources(
  [
    '__init__.py',
    '__init__.pyi',
    '_ducc.py',
    '_ducc_nd.py',
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
    'tests/test_ducc_nd.py',
  ],
  subdir: 'numpy/fft/tests',
  install_tag: 'tests'
)
