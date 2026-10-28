#!/usr/bin/env bash

# Vendor the pinned DUCC sources used by NumPy's FFT backend.
#
# Run with --check to verify the checked-in snapshot without changing it. A
# local source checkout can be supplied for offline checks, but it must be at
# the pinned commit. Updates from that checkout also require a clean DUCC tree.

set -o errexit
set -o nounset
set -o pipefail

readonly REPO_URL="https://gitlab.mpcdf.mpg.de/mtr/ducc.git"
readonly COMMIT_HASH="d8f5ffe71945aa5614cf61731dcb5dcc7789680f"
readonly SPDX_LICENSE="BSD-3-Clause OR GPL-2.0-or-later"

# Transitive DUCC FFT implementation closure used by numpy/fft. The .cc
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

usage() {
    cat <<'EOF'
Usage: vendor_duccfft.sh [--check] [--source-dir PATH]

  --check          Compare the generated files with the checked-in snapshot.
  --source-dir     Use a local DUCC checkout already at the pinned commit.
  -h, --help       Show this help.
EOF
}

CHECK_ONLY=0
SOURCE_DIR_OVERRIDE=""
while (($#)); do
    case "$1" in
        --check)
            CHECK_ONLY=1
            ;;
        --source-dir)
            if (($# < 2)); then
                echo "--source-dir requires a path" >&2
                exit 2
            fi
            SOURCE_DIR_OVERRIDE="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

readonly REPO_ROOT="$(git rev-parse --show-toplevel)"
readonly ROOT_DIR="${REPO_ROOT}/numpy/fft/ducc0"
readonly ROOT_PARENT="$(dirname "${ROOT_DIR}")"
readonly TEMP_DIR="$(mktemp -d)"
readonly STAGE_PARENT="$(mktemp -d "${ROOT_PARENT}/.ducc0.stage.XXXXXX")"
readonly STAGED_DIR="${STAGE_PARENT}/ducc0"
readonly SOURCE_DIR="${TEMP_DIR}/ducc"
readonly SELECTED_DIR="${TEMP_DIR}/selected"
readonly BACKUP_DIR="${ROOT_DIR}.backup.$$"

BACKUP_MOVED=0

cleanup() {
    local status=$?
    if ((BACKUP_MOVED)) && [[ ! -e "${ROOT_DIR}" ]]; then
        if ! mv "${BACKUP_DIR}" "${ROOT_DIR}"; then
            echo "failed to restore ${ROOT_DIR}; recovery copy remains at ${BACKUP_DIR}" >&2
            status=1
        fi
    fi
    rm -rf "${TEMP_DIR}" "${STAGE_PARENT}"
    exit "${status}"
}
trap cleanup EXIT

if [[ ! -f "${ROOT_DIR}/LICENSE.md" || ! -f "${ROOT_DIR}/README.md" ]]; then
    echo "expected vendored LICENSE.md and README.md in ${ROOT_DIR}" >&2
    exit 1
fi

if [[ -n "${SOURCE_DIR_OVERRIDE}" ]]; then
    if [[ ! -d "${SOURCE_DIR_OVERRIDE}" ]]; then
        echo "DUCC source directory does not exist: ${SOURCE_DIR_OVERRIDE}" >&2
        exit 1
    fi
    source_dir="$(cd "${SOURCE_DIR_OVERRIDE}" && pwd)"
else
    git clone --quiet "${REPO_URL}" "${SOURCE_DIR}"
    git -C "${SOURCE_DIR}" checkout --quiet --detach "${COMMIT_HASH}"
    source_dir="${SOURCE_DIR}"
fi

actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT_HASH}" ]]; then
    echo "DUCC checkout is ${actual_commit}, expected ${COMMIT_HASH}" >&2
    exit 1
fi
if [[ -n "${SOURCE_DIR_OVERRIDE}" ]] && ((!CHECK_ONLY)); then
    if ! git -C "${source_dir}" diff --quiet "${COMMIT_HASH}" -- src/ducc0; then
        echo "DUCC source tree has local changes: ${source_dir}" >&2
        exit 1
    fi
    untracked_files="$(git -C "${source_dir}" ls-files --others --exclude-standard -- src/ducc0)"
    if [[ -n "${untracked_files}" ]]; then
        echo "DUCC source tree has untracked files: ${source_dir}" >&2
        exit 1
    fi
fi

mkdir -p "${SELECTED_DIR}/src/ducc0" "${STAGED_DIR}"
for relative_path in "${DUCC_SOURCE_FILES[@]}"; do
    source_file="${source_dir}/src/ducc0/${relative_path}"
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

for relative_path in "${DUCC_SOURCE_FILES[@]}"; do
    destination="${STAGED_DIR}/${relative_path}"
    mkdir -p "$(dirname "${destination}")"
    cp "${SELECTED_DIR}/src/ducc0/${relative_path}" "${destination}"
done

# Preserve the checked-in license selection and reproduction note.
cp "${ROOT_DIR}/LICENSE.md" "${STAGED_DIR}/LICENSE.md"
cp "${ROOT_DIR}/README.md" "${STAGED_DIR}/README.md"

# Local patch 1: normalize DUCC's trailing blank whitespace in this header so
# the vendored diff passes git diff --check.
python3 - "${STAGED_DIR}/infra/aligned_array.h" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.write_text(path.read_text().rstrip() + "\n")
PY

if ((CHECK_ONLY)); then
    if diff -ruN "${ROOT_DIR}" "${STAGED_DIR}"; then
        echo "vendored DUCC snapshot matches ${COMMIT_HASH}"
    else
        echo "vendored DUCC snapshot differs from ${COMMIT_HASH}" >&2
        exit 1
    fi
    exit 0
fi

# Install only after fetching, manifest/SPDX/license checks, and the local
# whitespace normalization have succeeded. The sibling staging directory keeps rename on the
# same filesystem. If installation fails before the new tree is in place, the
# EXIT trap restores the old tree from its backup.
if [[ -e "${BACKUP_DIR}" ]]; then
    echo "refusing to overwrite existing recovery path: ${BACKUP_DIR}" >&2
    exit 1
fi
mv "${ROOT_DIR}" "${BACKUP_DIR}"
BACKUP_MOVED=1
if ! mv "${STAGED_DIR}" "${ROOT_DIR}"; then
    echo "failed to install staged DUCC tree; restoring previous tree" >&2
    mv "${BACKUP_DIR}" "${ROOT_DIR}"
    BACKUP_MOVED=0
    exit 1
fi
rm -rf "${BACKUP_DIR}"
BACKUP_MOVED=0
echo "updated vendored DUCC sources at ${ROOT_DIR} from ${COMMIT_HASH}"
