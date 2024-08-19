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
    echo "Usage: $0 [--no-cudf] [pandas-version]"
}

# Default value for the --no-cudf option
no_cudf=false
PANDAS_VERSION=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cudf)
            no_cudf=true
            shift
            ;;
        *)
            if [[ -z "$PANDAS_VERSION" ]]; then
                PANDAS_VERSION=$1
                shift
            else
                echo "Error: Unknown option $1"
                display_usage
                exit 1
            fi
            ;;
    esac
done

if [ "$no_cudf" = true ]; then
    echo "Skipping cudf install"
else
    RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
    RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-pylibcudf-dep
    RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
    python -m pip install $(ls ./local-pylibcudf-dep/pylibcudf*.whl)
    python -m pip install $(ls ./local-cudf-dep/cudf*.whl)[test,cudf-pandas-tests]
fi

# Conditionally install the specified version of pandas
if [ -n "$PANDAS_VERSION" ]; then
    echo "Installing pandas version: $PANDAS_VERSION"
    python -m pip install pandas==$PANDAS_VERSION
else
    echo "No pandas version specified, using existing pandas installation"
fi

python -m pytest -p cudf.pandas \
    --cov-config=./python/cudf/.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
    --cov-report=term \
    ./python/cudf/cudf_pandas_tests/
