#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1909 cpp
)
RMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1909 python
)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat > /tmp/constraints.txt <<EOF
librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_WHEELHOUSE}/librmm_*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${RMM_WHEELHOUSE}/rmm_*.whl)
EOF

export PIP_CONSTRAINT=/tmp/constraints.txt
