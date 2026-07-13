#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: Assumes the caller already ran `source rapids-init-pip` (so PIP_CONSTRAINT
# is set) and finished generating its own constraints. This APPENDS to
# PIP_CONSTRAINT, so it must be sourced AFTER `rapids-generate-pip-constraints`
# (which overwrites that file).

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# download wheels, store the directories holding them in variables
LIBRAPIDSMPF_WHEELHOUSE=$(rapids-get-pr-artifact rapidsmpf 1106 cpp wheel)
RAPIDSMPF_WHEELHOUSE=$(rapids-get-pr-artifact rapidsmpf 1106 python wheel)

cat >> "${PIP_CONSTRAINT}" <<EOF
librapidsmpf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAPIDSMPF_WHEELHOUSE}"/librapidsmpf_*.whl)
rapidsmpf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RAPIDSMPF_WHEELHOUSE}"/rapidsmpf_*.whl)
EOF
