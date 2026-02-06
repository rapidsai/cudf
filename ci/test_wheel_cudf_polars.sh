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
rapids-generate-pip-constraints py_test_cudf_polars ./constraints.txt

# notes:
#
#   * echo to expand wildcard before adding `[test,experimental]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,experimental]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

rapids-logger "Run cudf_polars tests"

POLARS_VERSIONS=$(python ci/utils/fetch_polars_versions.py --latest-patch-only dependencies.yaml)

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

read -r -a VERSIONS <<< "${POLARS_VERSIONS}"
LATEST_VERSION="${VERSIONS[-1]}"

for version in "${VERSIONS[@]}"; do
    rapids-logger "Installing polars==${version}"
    pip install -U "polars==${version}"

    rapids-logger "Running tests for polars==${version}"

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

    if [ $? -ne 0 ]; then
        EXITCODE=1
        FAILED+=("${version}")
        rapids-logger "Tests failed for polars==${version}"
    else
        PASSED+=("${version}")
        rapids-logger "Tests passed for polars==${version}"
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
