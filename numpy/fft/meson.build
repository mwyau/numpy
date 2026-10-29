largefile_define = []
if host_machine.system() == 'aix' or host_machine.system() == 'AIX'
  largefile_define += '-D_LARGE_FILES'
endif

# Match NumPy's no-plan-cache policy.  No worker pool, shared cache, or
# cache mutex is needed; separate Python threads can still invoke FFTs.
duccfft_args = [
  '-DDUCC0_NO_LOWLEVEL_THREADING',
  '-DDUCC0_NO_FFT_CACHE',
]

duccfft_sources = [
  'ducc0/infra/mav.cc',
  'ducc0/infra/string_utils.cc',
  'ducc0/infra/threading.cc',
]

# NumPy's per-target x86 -m flags *enable* features, but do not disable
# features that the compiler enables by default.  For example, GCC 15 on
# Ubuntu 26.04 defaults to x86-64-v3; adding -msse4.2 still emits AVX2 code
# for an X86_V2 baseline.  Explicitly establish the requested baseline ISA
# before multi_targets() adds the higher-target feature flags.  Apply the
# same baseline to the non-dispatched extension and shared DUCC support code.
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
  ],
  subdir: 'numpy/fft/tests',
  install_tag: 'tests'
)
