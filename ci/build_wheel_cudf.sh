#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_dir="python/cudf"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcudf and pylibcudf wheels from this current build,
# then ensures 'cudf' wheel builds always use the 'libcudf' and 'pylibcudf' just built in the same CI run.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
    PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
    source ./ci/use_upstream_sabi_wheels.sh
else
    PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
fi

LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUDF_WHEELHOUSE}/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"
echo "pylibcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBCUDF_WHEELHOUSE}/pylibcudf_*.whl)" >> "${PIP_CONSTRAINT}"


./ci/build_wheel.sh cudf ${package_dir}

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude libnvcomp.so.* \
    --exclude libkvikio.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
    RAPIDS_PACKAGE_NAME="$(rapids-package-name wheel_python cudf --stable --cuda)"
    export RAPIDS_PACKAGE_NAME
fi
