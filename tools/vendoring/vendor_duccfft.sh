#!/usr/bin/env bash

# Vendor the DUCC sources used by NumPy's FFT backend.
#
# The manifest below is the source of truth for the vendored implementation
# closure.  SPDX validation is kept separate so that a DUCC update cannot
# silently change the license of a required file.

set -o errexit
set -o nounset
set -o pipefail

readonly REPO_URL="https://gitlab.mpcdf.mpg.de/mtr/ducc.git"
readonly COMMIT_HASH="d8f5ffe71945aa5614cf61731dcb5dcc7789680f"
readonly SPDX_LICENSE="BSD-3-Clause OR GPL-2.0-or-later"

# Transitive DUCC FFT implementation closure used by numpy/fft.  The .cc
# files are compiled by numpy/fft/meson.build; the headers are reached from
# fftnd_impl.h and the support sources.
readonly DUCC_SOURCE_FILES=(
    "fft/fft.h"
    "fft/fft1d_impl.h"
    "fft/fftnd_impl.h"
    "infra/aligned_array.h"
    "infra/error_handling.h"
    "infra/mav.cc"
    "infra/mav.h"
    "infra/misc_utils.h"
    "infra/simd.h"
    "infra/string_utils.cc"
    "infra/string_utils.h"
    "infra/threading.cc"
    "infra/threading.h"
    "infra/useful_macros.h"
    "math/cmplx.h"
    "math/unity_roots.h"
)

readonly ROOT_DIR="$(git rev-parse --show-toplevel)/numpy/fft/ducc0"
readonly TEMP_DIR="$(mktemp -d)"
readonly SOURCE_DIR="${TEMP_DIR}/ducc"
readonly SELECTED_DIR="${TEMP_DIR}/selected"

cleanup() {
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

# Keep the checked-in license and reproduction note while refreshing the
# source files generated below.
readonly METADATA_DIR="${TEMP_DIR}/metadata"
mkdir -p "${METADATA_DIR}"
cp "${ROOT_DIR}/LICENSE.md" "${METADATA_DIR}/LICENSE.md"
cp "${ROOT_DIR}/README.md" "${METADATA_DIR}/README.md"

git clone "${REPO_URL}" "${SOURCE_DIR}"
git -C "${SOURCE_DIR}" checkout --quiet "${COMMIT_HASH}"

actual_commit="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT_HASH}" ]]; then
    echo "DUCC checkout is ${actual_commit}, expected ${COMMIT_HASH}" >&2
    exit 1
fi

mkdir -p "${SELECTED_DIR}/src/ducc0"
for relative_path in "${DUCC_SOURCE_FILES[@]}"; do
    source_file="${SOURCE_DIR}/src/ducc0/${relative_path}"
    if [[ ! -f "${source_file}" ]]; then
        echo "DUCC manifest entry does not exist: src/ducc0/${relative_path}" >&2
        exit 1
    fi
    if ! grep -Fqx \
            "/* SPDX-License-Identifier: ${SPDX_LICENSE} */" "${source_file}"; then
        echo "DUCC source has an unexpected license: ${relative_path}" >&2
        exit 1
    fi
    destination="${SELECTED_DIR}/src/ducc0/${relative_path}"
    mkdir -p "$(dirname "${destination}")"
    cp "${source_file}" "${destination}"
done

# Keep the embedded FFTW-derived DCT-IV code's copyright and permission
# auditable when refreshing the vendored source snapshot.
python3 - "${SELECTED_DIR}/src/ducc0/fft/fft.h" \
    "${SELECTED_DIR}/src/ducc0/fft/fft1d_impl.h" \
    "${SELECTED_DIR}/src/ducc0/fft/fftnd_impl.h" <<'PY'
from pathlib import Path
import sys

fft_h, fft1d_h, fftnd_h = (Path(name).read_text() for name in sys.argv[1:])
notice = """        // The following code is derived from the FFTW3 function apply_re11()
        // and is released under the 3-clause BSD license with friendly
        // permission of Matteo Frigo and Steven G. Johnson."""
if fft_h.count(notice) != 1:
    raise SystemExit("expected one FFTW-derived BSD permission notice in fft.h")
for name, text in (("fft1d_impl.h", fft1d_h), ("fftnd_impl.h", fftnd_h)):
    for attribution in (
        "Copyright (C) 2003, 2007-14 Matteo Frigo",
        "Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology",
    ):
        if text.count(attribution) != 1:
            raise SystemExit(f"missing or duplicate FFTW attribution in {name}")
PY

rm -rf "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}"
for relative_path in "${DUCC_SOURCE_FILES[@]}"; do
    destination="${ROOT_DIR}/${relative_path}"
    mkdir -p "$(dirname "${destination}")"
    cp "${SELECTED_DIR}/src/ducc0/${relative_path}" "${destination}"
done

# Avoid a whitespace-only extra line at EOF in the upstream aligned-array
# header so the vendored diff passes git diff --check.
python3 - "${ROOT_DIR}/infra/aligned_array.h" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.write_text(path.read_text().rstrip() + "\n")
PY

# NumPy retains DUCC's plan cache but disables its worker pool.  The upstream
# no-worker configuration provides no-op locks, so give only the cache-enabled
# no-worker build real mutex types.  Fail closed if an upstream refresh moves
# either of the expected sections.
python3 - "${ROOT_DIR}/infra/threading.h" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
header_before = """#ifdef DUCC0_STDCXX_LOWLEVEL_THREADING
#include <mutex>
#include <condition_variable>
#endif

#ifdef DUCC0_NO_LOWLEVEL_THREADING
// no headers needed
#endif
"""
header_after = """#if defined(DUCC0_STDCXX_LOWLEVEL_THREADING) || defined(DUCC0_FFT_CACHE_MUTEX)
#include <mutex>
#endif

#ifdef DUCC0_STDCXX_LOWLEVEL_THREADING
#include <condition_variable>
#endif
"""
locks_before = """#ifdef DUCC0_NO_LOWLEVEL_THREADING
struct Mutex
  {
  void lock(){}
  void unlock(){}
  };
struct LockGuard
  {
  LockGuard(const Mutex &){}
  };
struct UniqueLock
  {
  UniqueLock(const Mutex &){}
  void lock() {}
  void unlock() {}
  };
struct CondVar
"""
locks_after = """#ifdef DUCC0_NO_LOWLEVEL_THREADING
#ifdef DUCC0_FFT_CACHE_MUTEX
// NumPy keeps DUCC's plan cache while disabling its worker pool.
using Mutex = std::mutex;
using LockGuard = std::lock_guard<Mutex>;
using UniqueLock = std::unique_lock<Mutex>;
#else
struct Mutex
  {
  void lock(){}
  void unlock(){}
  };
struct LockGuard
  {
  LockGuard(const Mutex &){}
  };
struct UniqueLock
  {
  UniqueLock(const Mutex &){}
  void lock() {}
  void unlock() {}
  };
#endif
struct CondVar
"""

for before, after in ((header_before, header_after), (locks_before, locks_after)):
    if text.count(before) != 1:
        raise SystemExit(f"expected one DUCC threading section in {path}")
    text = text.replace(before, after, 1)

path.write_text(text)
PY

cp "${METADATA_DIR}/LICENSE.md" "${ROOT_DIR}/LICENSE.md"
cp "${METADATA_DIR}/README.md" "${ROOT_DIR}/README.md"
