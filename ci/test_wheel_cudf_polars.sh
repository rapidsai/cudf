#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# Download libcudf and pylibcudf built in the previous step
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Installing cudf_polars and its dependencies"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf_polars "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,experimental]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

rapids-logger "Run cudf_polars tests"

read -r -a VERSIONS <<< "$(python ci/utils/get_matrix_values.py dependencies.yaml test_cudf_polars_compat polars_compat_version)"
LATEST_VERSION="${VERSIONS[-1]}"

# shellcheck disable=SC2317
function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

PASSED=()
FAILED=()

for version in "${VERSIONS[@]}"; do
    rapids-logger "Testing cudf_polars with polars ${version}.*"

    # Generate requirements for this polars compat version.
    rapids-dependency-file-generator \
        --config dependencies.yaml \
        --file-key test_cudf_polars_compat \
        --output requirements \
        --matrix "polars_compat_version=${version}" \
        > "polars-compat-${version}-requirements.txt"

    # Create an isolated virtual environment inheriting the already-installed cudf_polars
    # wheels so we only need to override the polars version.
    python -m venv --system-site-packages "venv_polars_${version}"
    # shellcheck disable=SC1090
    source "venv_polars_${version}/bin/activate"

    rapids-pip-retry install -r "polars-compat-${version}-requirements.txt"

    rapids-logger "Running tests for polars ${version}.*"

    if [ "${version}" == "${LATEST_VERSION}" ]; then
        COVERAGE_ARGS=(
            --cov=cudf_polars
            --cov-fail-under=100
            --cov-report=term-missing:skip-covered
            --cov-config=./pyproject.toml
        )
    else
        COVERAGE_ARGS=(--no-cov)
    fi

    timeout 1h ./ci/run_cudf_polars_pytests.sh \
        "${COVERAGE_ARGS[@]}" \
        --numprocesses=8 \
        --dist=worksteal \
        --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-${version}.xml"

    test_exitcode=$?
    deactivate
    rm -rf "venv_polars_${version}" "polars-compat-${version}-requirements.txt"

    if [ ${test_exitcode} -ne 0 ]; then
        EXITCODE=1
        FAILED+=("${version}")
        rapids-logger "Tests failed for polars ${version}.*"
    else
        PASSED+=("${version}")
        rapids-logger "Tests passed for polars ${version}.*"
    fi
done

trap ERR
set -e

rapids-logger "Polars test summary:"
rapids-logger "PASSED: ${PASSED[*]:-none}"
rapids-logger "FAILED: ${FAILED[*]:-none}"

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "Testing FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "Testing PASSED"
fi
exit ${EXITCODE}
