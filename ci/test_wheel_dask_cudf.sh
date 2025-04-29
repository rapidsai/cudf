#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
DASK_CUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# Download the cudf, libcudf, and pylibcudf built in the previous step
CUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-logger "Install dask_cudf, cudf, pylibcudf, and test requirements"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_dask_cudf ./constraints.txt

# ci/use_wheels_from_prs.sh
RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBKVIKIO_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact-github kvikio 702 cpp
)
KVIKIO_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="kvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact-github kvikio 702 python
)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat >> ./constraints.txt <<EOF
libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBKVIKIO_WHEELHOUSE}/libkvikio_*.whl)
kvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${KVIKIO_WHEELHOUSE}/kvikio_*.whl)
EOF

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
  -v \
  --constraint ./constraints.txt \
  "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
  "$(echo "${DASK_CUDF_WHEELHOUSE}"/dask_cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
  "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
  "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# Run tests in dask_cudf/tests and dask_cudf/io/tests
rapids-logger "pytest dask_cudf"
pushd python/dask_cudf/dask_cudf
python -m pytest \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  .
popd
