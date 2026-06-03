#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download cudf_streaming, libcudf_streaming, and pylibcudf built in previous steps
CUDF_STREAMING_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_streaming_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUDF_STREAMING_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_streaming_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf_streaming "${PIP_CONSTRAINT}"

rapids-logger "Install cudf_streaming and its dependencies"

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_STREAMING_WHEELHOUSE}"/cudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
    "$(echo "${LIBCUDF_STREAMING_WHEELHOUSE}"/libcudf_streaming_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

rapids-logger "pytest cudf_streaming"
pushd python/cudf_streaming/cudf_streaming
timeout 30m python -m pytest \
  --cache-clear \
  --numprocesses=8 \
  --dist=worksteal \
  tests
popd
