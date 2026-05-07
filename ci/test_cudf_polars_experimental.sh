#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Installing cudf_polars and its dependencies (including rapidsmpf)"

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

rapids-logger "Run cudf_polars tests with rapidsmpf"

# Get the latest polars version for testing
available_polars_versions=$(python -m pip index versions polars --json | jq '.versions')
POLARS_VERSION=$(python ci/utils/filter_package_versions.py dependencies.yaml run_cudf_polars polars "$available_polars_versions" | awk '{print $NF}')

rapids-logger "Installing polars==${POLARS_VERSION}"
rapids-pip-retry install -U "polars==${POLARS_VERSION}"

# shellcheck disable=SC2317
function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

rapids-logger "Running cudf_polars experimental tests (non-ci-blocking)"
timeout 30m ./ci/run_cudf_polars_experimental_pytests.sh \
    --no-cov \
    --numprocesses=8 \
    --dist=worksteal \
    -v \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-rapidsmpf.xml"

trap - ERR
set -e

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "cudf_polars + rapidsmpf tests FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "cudf_polars + rapidsmpf tests PASSED"
fi
exit ${EXITCODE}
