#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

DEPENDENCIES_PATH="dependencies.yaml"
# https://github.com/jupyter/jupyter_core/pull/292
export JUPYTER_PLATFORM_DIRS=1

# Use grep to find the line containing the package name and version constraint
pandas_version_constraint=$(grep -oP "pandas>=\d+\.\d+(\.\d+)?,<\d+\.\d+(\.\d+)?" $DEPENDENCIES_PATH)

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
    RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

    # Download the cudf, libcudf, and pylibcudf built in the previous step
    CUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
    LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
    PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

    # generate constraints (possibly pinning to oldest support versions of dependencies)
    rapids-generate-pip-constraints test_python_cudf_pandas ./constraints.txt

    # notes:
    #
    #   * echo to expand wildcard before adding `[test,cudf-pandas-tests]` requires for pip
    #   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
    #     ignored if any other --constraint are passed via the CLI
    #
    python -m pip install \
        -v \
        --constraint ./constraints.txt \
        --constraint "${PIP_CONSTRAINT}" \
        "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,cudf-pandas-tests]" \
        "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
        "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"
fi

python -m pip install certifi ipykernel
python -m ipykernel install --user --name python3

rapids-logger "pytest cudf.pandas parallel"
# The third-party integration tests are ignored because they are run in a separate nightly CI job
# TODO: Root-cause why we cannot run the tests in profile.py in parallel and reconsider adding
# them back. Tracking https://github.com/rapidsai/cudf/issues/18261
python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    --numprocesses=8 \
    --dist=worksteal \
    -k "not profiler" \
    -m "not serial" \
    --config-file=./python/cudf/pyproject.toml \
    --cov-config=./python/cudf/.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
    --cov-report=term \
    ./python/cudf/cudf_pandas_tests/


rapids-logger "pytest cudf.pandas serial"

python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    -k "not profiler" \
    -m "serial" \
    --config-file=./python/cudf/pyproject.toml \
    --cov-config=./python/cudf/.coveragerc \
    --cov=cudf \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage-serial.xml" \
    --cov-report=term \
    ./python/cudf/cudf_pandas_tests/

# pytest-xdist and pytest-cov prevent our profiler's trace function from running,
# even with pytest.mark.no_cover. This likely stems from specialized logic in
# coveragepy and pytest-cov for distributed testing (pytest-dev/pytest-cov#246).
# As a workaround, we run profiler tests separately without parallelism or `--cov`.
# More details: https://github.com/rapidsai/cudf/pull/16930#issuecomment-2707873968
python -m pytest -p cudf.pandas \
    --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
    --numprocesses=0 \
    -k "profiler" \
    ./python/cudf/cudf_pandas_tests/

output=$(python ci/cudf_pandas_scripts/fetch_pandas_versions.py "$pandas_version_constraint")

# Convert the comma-separated list into an array
IFS=',' read -r -a versions <<< "$output"

for version in "${versions[@]}"; do
    echo "Installing pandas version: ${version}"
    python -m pip install "numpy>=1.23,<2.0a0" "pandas==${version}.*"
    python -m pytest -p cudf.pandas \
        --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
        --numprocesses=8 \
        --dist=worksteal \
        -k "not profiler" \
        -m "not serial" \
        --cov-config=./python/cudf/.coveragerc \
        --cov=cudf \
        --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
        --cov-report=term \
        ./python/cudf/cudf_pandas_tests/

    # NOTE: We don't currently run serial tests (only 1 as of 2025-07-25)
    # with multiple versions of pandas.

    python -m pytest -p cudf.pandas \
        --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
        --numprocesses=0 \
        -k "profiler" \
        ./python/cudf/cudf_pandas_tests/
done
