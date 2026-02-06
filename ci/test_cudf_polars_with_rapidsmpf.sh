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

rapids-pip-retry install \
    -v \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,experimental,rapidsmpf]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

rapids-logger "Run cudf_polars tests with rapidsmpf"

# Get the latest polars version for testing
POLARS_VERSION=$(python ci/utils/fetch_polars_versions.py --latest-patch-only dependencies.yaml | awk '{print $NF}')

rapids-logger "Installing polars==${POLARS_VERSION}"
pip install -U "polars==${POLARS_VERSION}"

# shellcheck disable=SC2317
function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

rapids-logger "Running cudf_polars tests with rapidsmpf"

# Run cudf_polars tests with rapidsmpf using dedicated test runner
timeout 15m ./ci/run_cudf_polars_with_rapidsmpf_pytests.sh \
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
