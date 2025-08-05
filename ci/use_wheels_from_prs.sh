#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

source ci/use_gha_tools_from_branch.sh

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2008 cpp wheel
)
RMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2008 python wheel
)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat > "${PIP_CONSTRAINT}" <<EOF
librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_WHEELHOUSE}/librmm_*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${RMM_WHEELHOUSE}/librmm_*.whl)
EOF
