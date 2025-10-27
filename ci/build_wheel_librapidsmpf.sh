#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

rapids-print-env

rapids-logger "Download cudf wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download libcudf and pylibcudf built in the previous step
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-logger "Install libcudf and pylibcudf wheels"
# Install these first so rapidsmpf can find them
python -m pip install \
    -v \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

rapids-logger "Clone rapidsmpf from main branch"
git clone https://github.com/rapidsai/rapidsmpf.git --branch main --depth 1
cd rapidsmpf

# Use rapidsmpf's existing VERSION file (don't overwrite it)
rapids-logger "rapidsmpf VERSION: $(cat VERSION)"

################################################################################
# Build librapidsmpf wheel (C++ library)
################################################################################
rapids-logger "Building librapidsmpf wheel"

package_name="librapidsmpf"
package_dir="python/librapidsmpf"

rapids-logger "Generating build requirements for ${package_name}"
rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build-librapidsmpf.txt

rapids-logger "Installing build requirements for ${package_name}"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build-librapidsmpf.txt

# Build with '--no-build-isolation', for better sccache hit rate
export PIP_NO_BUILD_ISOLATION=0

# Disable MPI, tests, benchmarks, examples, and NUMA support for wheel build
export SKBUILD_CMAKE_ARGS="-DBUILD_MPI_SUPPORT=OFF;-DBUILD_TESTS=OFF;-DBUILD_BENCHMARKS=OFF;-DBUILD_EXAMPLES=OFF;-DBUILD_NUMA_SUPPORT=OFF"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"
cd "${package_dir}"
rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

rapids-logger "Repairing ${package_name} wheel with auditwheel"
python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    --exclude libucp.so.0 \
    --exclude libucxx.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    dist/*

rapids-logger "List output wheels"
ls -lh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
