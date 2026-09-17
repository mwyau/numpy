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
readonly COMMIT_HASH="64f42ba531f609ba7029c82207a063b17f9d5275"
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

rm -rf "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}"
for relative_path in "${DUCC_SOURCE_FILES[@]}"; do
    destination="${ROOT_DIR}/${relative_path}"
    mkdir -p "$(dirname "${destination}")"
    cp "${SELECTED_DIR}/src/ducc0/${relative_path}" "${destination}"
done
cp "${METADATA_DIR}/LICENSE.md" "${ROOT_DIR}/LICENSE.md"
cp "${METADATA_DIR}/README.md" "${ROOT_DIR}/README.md"
