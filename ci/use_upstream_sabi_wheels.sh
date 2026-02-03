#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then

# download wheels, store the directories holding them in variables
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2241 cpp wheel
)
RMM_WHEELHOUSE=$(
  rapids-get-pr-artifact rmm 2241 python wheel --stable
)
LIBKVIKIO_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact kvikio 920 cpp wheel
)
KVIKIO_WHEELHOUSE=$(
  rapids-get-pr-artifact kvikio 920 python wheel --stable
)

cat > "${PIP_CONSTRAINT}" <<EOF
librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm_*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RMM_WHEELHOUSE}"/rmm_*.whl)
libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio_*.whl)
kvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${KVIKIO_WHEELHOUSE}"/kvikio_*.whl)
EOF

fi
