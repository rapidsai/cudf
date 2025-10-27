#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels from build job"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download cudf-polars wheel (built in the combined build job)
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# Download libcudf and pylibcudf from earlier jobs
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# Download rapidsmpf wheels
# librapidsmpf is a cpp package (like libcudf), rapidsmpf is python package
LIBRAPIDSMPF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RAPIDSMPF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rapidsmpf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-logger "Installing wheels"

# Install libcudf and pylibcudf first
rapids-pip-retry install \
    -v \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

# Install librapidsmpf
rapids-logger "Installing librapidsmpf"
rapids-pip-retry install -v "$(echo "${LIBRAPIDSMPF_WHEELHOUSE}"/librapidsmpf*.whl)"

# Install rapidsmpf
rapids-logger "Installing rapidsmpf"
rapids-pip-retry install -v "$(echo "${RAPIDSMPF_WHEELHOUSE}"/rapidsmpf*.whl)"

# Install cudf_polars with test and experimental extras
rapids-logger "Installing cudf_polars with test and experimental extras"
rapids-generate-pip-constraints py_test_cudf_polars ./constraints.txt
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,experimental]"

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
./ci/run_cudf_polars_with_rapidsmpf_pytests.sh \
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
