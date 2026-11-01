largefile_define = []
if host_machine.system() == 'aix' or host_machine.system() == 'AIX'
  largefile_define += '-D_LARGE_FILES'
endif

# Disable DUCC's worker pool and FFT plan cache.
duccfft_args = [
  '-DDUCC0_NO_LOWLEVEL_THREADING',
  '-DDUCC0_NO_FFT_CACHE',
]

# Reset the baseline ISA before adding per-target flags.
# -msse4.2 alone does not disable compiler-default AVX2.
duccfft_baseline_c_args = []
duccfft_baseline_cpp_args = []
if host_machine.cpu_family() == 'x86_64' and cpp.get_id() in ['gcc', 'clang']
  duccfft_baseline_arch = 'x86-64'
  if 'X86_V4' in CPU_BASELINE_NAMES
    duccfft_baseline_arch = 'x86-64-v4'
  elif 'X86_V3' in CPU_BASELINE_NAMES
    duccfft_baseline_arch = 'x86-64-v3'
  elif 'X86_V2' in CPU_BASELINE_NAMES
    duccfft_baseline_arch = 'x86-64-v2'
  endif
  duccfft_baseline_arg = '-march=' + duccfft_baseline_arch
  duccfft_baseline_c_args = cc.get_supported_arguments(duccfft_baseline_arg)
  duccfft_baseline_cpp_args = cpp.get_supported_arguments(duccfft_baseline_arg)
endif

duccfft_dispatch_cpp_args = largefile_define + duccfft_args + max_opt + duccfft_baseline_cpp_args
duccfft_dispatch_cpp_args += cpp.get_supported_arguments('-fvisibility=hidden')

# Build FFT kernels for each CPU target.
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
  # Keep CPU feature helpers local, as in _simd.
  [
    '_duccfft_umath.cpp',
    '../_core/src/common/npy_cpu_features.c',
    'ducc0/infra/mav.cc',
    'ducc0/infra/string_utils.cc',
    'ducc0/infra/threading.cc',
  ],
  c_args: c_args_common + largefile_define + duccfft_baseline_c_args,
  cpp_args: largefile_define + duccfft_args + duccfft_baseline_cpp_args,
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
    'tests/test_native_nd_experiment.py',
    'tests/test_generic_radix_avx512.py',
  ],
  subdir: 'numpy/fft/tests',
  install_tag: 'tests'
)
