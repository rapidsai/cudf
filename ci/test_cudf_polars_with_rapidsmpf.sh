#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# Download libcudf and pylibcudf built in the previous step
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-logger "Installing cudf_polars and its dependencies"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf_polars ./constraints.txt

# Install rapidsmpf from nightly index
rapids-logger "Installing rapidsmpf from nightly"
rapids-pip-retry install \
    -v \
    --extra-index-url=https://pypi.nvidia.com \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    "rapidsmpf-${RAPIDS_PY_CUDA_SUFFIX}>=25.12.0a0,<25.13" \
    "librapidsmpf-${RAPIDS_PY_CUDA_SUFFIX}>=25.12.0a0,<25.13"

# Install cudf_polars with test and experimental extras
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,experimental]" \
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
