#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

# https://github.com/jupyter/jupyter_core/pull/292
export JUPYTER_PLATFORM_DIRS=1

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
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,cudf-pandas-tests]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

python -m ipykernel install --user --name python3

rapids-logger "pytest cudf.pandas parallel"
# The third-party integration tests are ignored because they are run in a separate nightly CI job
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
    --config-file=./python/cudf/pyproject.toml \
    --numprocesses=0 \
    -k "profiler" \
    ./python/cudf/cudf_pandas_tests/

available_pandas_versions=$(python -m pip index versions pandas --json | jq '.versions')
output=$(python ci/utils/filter_package_versions.py dependencies.yaml run_common pandas "$available_pandas_versions")

# Convert the space-separated list into an array
read -r -a versions <<< "${output}"

version_lte() {
  [ "$1" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

if version_lte "${RAPIDS_PY_VERSION}" "3.13"; then
    for version in "${versions[@]}"; do
        echo "Installing pandas version: ${version}"
        # This loop tests cudf.pandas compatibility with older pandas-numpy versions,
        # requiring numpy<2. cupy>=14 dropped support for numpy<2, so we explicitly
        # downgrade cupy here to avoid an import failure when cupy tries
        # to load against the older numpy.
        rapids-pip-retry install "numpy>=1.26,<2.0a0" "pandas==${version}" "cupy-cuda${RAPIDS_CUDA_VERSION%%.*}x<14"
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

        # NOTE: We don't currently run serial tests (only 1 as of 2025-07-25)
        # with multiple versions of pandas.

        python -m pytest -p cudf.pandas \
            --ignore=./python/cudf/cudf_pandas_tests/third_party_integration_tests/ \
            --numprocesses=0 \
            -k "profiler" \
            ./python/cudf/cudf_pandas_tests/
    done
else
    rapids-logger "Python ${RAPIDS_PY_VERSION} detected (>= 3.13). Skipping cudf.pandas compatibility tests with numpy<2"
fi
