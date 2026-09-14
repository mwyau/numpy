#!/usr/bin/env bash

# Vendor the FFT portion of DUCC into NumPy.
#
# The vendored files are selected by their SPDX expression rather than by a
# submodule so that a source checkout is self-contained and reproducible.

set -o errexit
set -o nounset
set -o pipefail

readonly REPO_URL="https://gitlab.mpcdf.mpg.de/mtr/ducc.git"
readonly COMMIT_HASH="64f42ba531f609ba7029c82207a063b17f9d5275"

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

# Only the files explicitly released under the compatible dual license are
# copied.  The tar archive preserves the source tree below src/ducc0.
mkdir -p "${SELECTED_DIR}"
(
    cd "${SOURCE_DIR}"
    git grep -l "SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-or-later" \
        -- src/ducc0 | tar -cf "${TEMP_DIR}/ducc_bsd.tar" -T -
)
tar -xf "${TEMP_DIR}/ducc_bsd.tar" -C "${SELECTED_DIR}"

rm -rf "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}"
mv "${SELECTED_DIR}/src/ducc0"/* "${ROOT_DIR}/"
cp "${METADATA_DIR}/LICENSE.md" "${ROOT_DIR}/LICENSE.md"
cp "${METADATA_DIR}/README.md" "${ROOT_DIR}/README.md"
