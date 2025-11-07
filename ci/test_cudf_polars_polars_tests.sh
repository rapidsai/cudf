#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# Download libcudf and pylibcudf built in the previous step
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-logger "Install libcudf, pylibcudf and cudf_polars"
rapids-pip-retry install \
    -v \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"


TAG=$(python -c 'import polars; print(f"py-{polars.__version__}")')
rapids-logger "Clone polars to ${TAG}"
git clone https://github.com/pola-rs/polars.git --branch "${TAG}" --depth 1

# Install requirements for running polars tests
rapids-logger "Install polars test requirements"
# We don't need to pick up dependencies from polars-cloud, so we remove it.
sed -i '/^polars-cloud$/d' polars/py-polars/requirements-dev.txt
# Deltalake release 1.2.0 contains breaking changes for Polars.
# Tracking issue: https://github.com/pola-rs/polars/issues/24872
sed -i 's/^deltalake>=1.1.4/deltalake>=1.1.4,<1.2.0/' polars/py-polars/requirements-dev.txt
# pyiceberg depends on a non-documented attribute of pydantic.
# AttributeError: 'pydantic_core._pydantic_core.ValidationInfo' object has no attribute 'current_schema_id'
sed -i 's/^pydantic>=2.0.0.*/pydantic>=2.0.0,<2.12.0/' polars/py-polars/requirements-dev.txt
rapids-pip-retry install -r polars/py-polars/requirements-dev.txt -r polars/py-polars/requirements-ci.txt

# shellcheck disable=SC2317
function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

rapids-logger "Run polars tests"
./ci/run_cudf_polars_polars_tests.sh

trap ERR
set -e

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "Running polars test suite FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "Running polars test suite PASSED"
fi
exit ${EXITCODE}
