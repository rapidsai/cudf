#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

# Function to display script usage
function display_usage {
    echo "Usage: $0 [--no-cudf]"
}

# Default value for the --no-cudf option
no_cudf=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cudf)
            no_cudf=true
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            display_usage
            exit 1
            ;;
    esac
done

if [ "$no_cudf" = true ]; then
    echo "Skipping cudf install"
else
    RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

    # Download the cudf, libcudf, and pylibcudf built in the previous step
    RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
    RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
    RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

    # echo to expand wildcard before adding `[extra]` requires for pip
    python -m pip install \
        "$(echo ./dist/cudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test,cudf-pandas-tests]" \
        "$(echo ./dist/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
        "$(echo ./dist/pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)"
fi

python -m pytest -p cudf.pandas \
    --cov-config=./python/cudf/.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
    --cov-report=term \
    ./python/cudf/cudf_pandas_tests/
