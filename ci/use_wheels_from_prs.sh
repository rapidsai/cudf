#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
https://github.com/rapidsai/kvikio/pull/942/commits/b71d3beaca46a0ec8d54a948c1c9bca5e4359b53
KVIKIO_COMMIT=b71d3beaca46a0ec8d54a948c1c9bca5e4359b53
LIBKVIKIO_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact kvikio 942 cpp wheel "${KVIKIO_COMMIT}"
)
KVIKIO_WHEELHOUSE=$(
  rapids-get-pr-artifact kvikio 942 python wheel --pkg_name kvikio --stable "${KVIKIO_COMMIT}"
)

RMM_COMMIT=af96977c404565bbb0657c490d407a561cedc3fc
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2270 cpp wheel "${RMM_COMMIT}"
)
RMM_WHEELHOUSE=$(
  rapids-get-pr-artifact rmm 2270 python wheel --pkg_name rmm --stable "${RMM_COMMIT}"
)

RAFT_COMMIT=b7b9c53d7c492937cc9214cbd292039d6551d996
LIBRAFT_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 2971 cpp wheel "${RAFT_COMMIT}"
)
PYLIBRAFT_WHEELHOUSE=$(
  rapids-get-pr-artifact raft 2971 python wheel --pkg_name pylibraft --stable "${RAFT_COMMIT}"
)
RAFT_DASK_WHEELHOUSE=$(
  rapids-get-pr-artifact raft 2971 python wheel --pkg_name raft_dask --stable "${RAFT_COMMIT}"
)

UCXX_COMMIT=b1e5230d5b1ac47a11cc2d421b3af91be50b39e4
DISTRIBUTED_UCXX_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-get-pr-artifact --pkg_name distributed-ucxx ucxx 604 python wheel
)
LIBUCXX_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact ucxx 604 cpp wheel "${UCXX_COMMIT}"
)
UCXX_WHEELHOUSE=$(
  rapids-get-pr-artifact ucxx 604 python wheel --pkg_name ucxx --stable "${UCXX_COMMIT}"
)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat >> "${PIP_CONSTRAINT}" <<EOF
libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio*.whl)
kvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${KVIKIO_WHEELHOUSE}"/kvikio_*.whl)

libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft*.whl)
pylibraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft*.whl)
raft-dask-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RAFT_DASK_WHEELHOUSE}"/raft_dask*.whl)

librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RMM_WHEELHOUSE}"/rmm_*.whl)

distributed-ucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${DISTRIBUTED_UCXX_WHEELHOUSE}"/distributed_ucxx*.whl)
libucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBUCXX_WHEELHOUSE}"/libucxx*.whl)
ucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${UCXX_WHEELHOUSE}"/ucxx*.whl)
EOF
