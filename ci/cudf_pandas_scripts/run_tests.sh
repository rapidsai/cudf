#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

DEPENDENCIES_PATH="dependencies.yaml"
package_name="pandas"

# Use grep to find the line containing the package name and version constraint
pandas_version_constraint=$(grep -oP "pandas>=\d+\.\d+,\<\d+\.\d+\.\d+dev\d+" $DEPENDENCIES_PATH)

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

    # Download the cudf, libcudf, and pylibcudf built in the previous step
    RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
    RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
    RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

    # generate constraints (possibly pinning to oldest support versions of dependencies)
    rapids-generate-pip-constraints test_python_cudf_pandas ./constraints.txt

    python -m pip install \
        -v \
        --constraint ./constraints.txt \
        "$(echo ./dist/cudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test,cudf-pandas-tests]" \
        "$(echo ./dist/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
        "$(echo ./dist/pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)"
fi

python -m pip install ipykernel
python -m ipykernel install --user --name python3

# The third-party integration tests are ignored because they are run nightly in seperate CI job
python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    --numprocesses=8 \
    --dist=worksteal \
    --cov-config=./python/cudf/.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
    --cov-report=term \
    ./python/cudf/cudf_pandas_tests/

output=$(python ci/cudf_pandas_scripts/fetch_pandas_versions.py $pandas_version_constraint)

# Convert the comma-separated list into an array
IFS=',' read -r -a versions <<< "$output"

for version in "${versions[@]}"; do
    echo "Installing pandas version: ${version}"
    python -m pip install "numpy>=1.23,<2.0a0" "pandas==${version}.*"
    python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    --numprocesses=8 \
    --dist=worksteal \
    --cov-config=./python/cudf/.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
    --cov-report=term \
    ./python/cudf/cudf_pandas_tests/
done
