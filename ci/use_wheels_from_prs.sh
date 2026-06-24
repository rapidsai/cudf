#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This script assumes the caller has already run `source rapids-init-pip`
# (so PIP_CONSTRAINT is set) and has finished generating any of its own
# constraints. It APPENDS to PIP_CONSTRAINT so it must not be sourced before
# `rapids-generate-pip-constraints`, which overwrites that file.

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# download wheels, store the directories holding them in variables
LIBRAPIDSMPF_WHEELHOUSE=$(rapids-get-pr-artifact rapidsmpf 1108 cpp wheel)
RAPIDSMPF_WHEELHOUSE=$(rapids-get-pr-artifact rapidsmpf 1108 python wheel)

# append to the pip constraints file saying e.g. "whenever you encounter a
# requirement for 'librapidsmpf-cu12', use this wheel"
cat >> "${PIP_CONSTRAINT}" <<EOF
librapidsmpf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAPIDSMPF_WHEELHOUSE}"/librapidsmpf_*.whl)
rapidsmpf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RAPIDSMPF_WHEELHOUSE}"/rapidsmpf_*.whl)
EOF
