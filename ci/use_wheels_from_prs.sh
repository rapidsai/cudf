#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Download wheel CI artifacts from dependent PRs
# NOTE: This script must be sourced AFTER rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2106 cpp wheel
)
RMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2106 python wheel
)

cat >> "${PIP_CONSTRAINT}" <<EOF
librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_WHEELHOUSE}/librmm_*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${RMM_WHEELHOUSE}/rmm_*.whl)
EOF
