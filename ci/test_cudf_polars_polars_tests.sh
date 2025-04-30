#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -eou pipefail

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# Download libcudf and pylibcudf built in the previous step
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

set -x
AUDITWHEEL_ARCH="$(arch)"
export AUDITWHEEL_ARCH
AUDITWHEEL_PLAT="manylinux_2_28_$(arch)"
export AUDITWHEEL_PLAT
LIBKVIKIO_WHL="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}-25.6.0a32-py3-none-${AUDITWHEEL_PLAT}.whl"
LIBKVIKIO_TARBALL="kvikio_wheel_cpp_libkvikio_${RAPIDS_PY_CUDA_SUFFIX}_${AUDITWHEEL_ARCH}.tar.gz"
LIBKVIKIO_DIR=$(rapids-get-artifact "ci/kvikio/pull-request/702/7f957eb/${LIBKVIKIO_TARBALL}")
echo "libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://${LIBKVIKIO_DIR}/${LIBKVIKIO_WHL}" >> /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"
set +x


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
# TODO: Remove sed command when polars-cloud supports 1.23
sed -i '/^polars-cloud$/d' polars/py-polars/requirements-dev.txt
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
