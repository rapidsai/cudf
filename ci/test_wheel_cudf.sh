#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

rapids-logger "Install cudf and test requirements"
PIP_PACKAGE=$(echo ./dist/cudf*.whl | head -n1)
# Use `package[test]` to install latest test dependencies or explicitly install oldest.
if [[ $RAPIDS_DEPENDENCIES != "oldest" ]]; then
    python -m pip install -v ${PIP_PACKAGE}[test]
else
    rapids-dependency-file-generator \
        --output requirements \
        --file-key py_test_cudf \
        --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
      | tee oldest-dependencies.txt

    python -m pip install -v ${PIP_PACKAGE} -r oldest-dependencies.txt
fi


RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"


rapids-logger "pytest pylibcudf"
pushd python/cudf/cudf/pylibcudf_tests
python -m pytest \
  --cache-clear \
  --dist=worksteal \
  .
popd

rapids-logger "pytest cudf"
pushd python/cudf/cudf/tests
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  .
popd
