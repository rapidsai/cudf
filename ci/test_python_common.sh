#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Common setup steps shared by Python test jobs

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")

# Unpatched MPF fails on empty buffers with new RMM; remove after independent fixes ship.
MPF_PR=1261
MPF_COMMIT=f06149945f934b7be925f2ad2db100767eaec25f
if [[ "${RAPIDS_DEPENDENCIES}" == "oldest" ]]; then
  # Oldest numba needs a fixed MPF build that retains the older UCXX/cuda-core minimum.
  MPF_PR=1258
  MPF_COMMIT=5ce52df90de32c03c449a51619b20d72fe7f3474
fi
MPF_CPP_CHANNEL=$(rapids-get-pr-artifact rapidsmpf "${MPF_PR}" cpp conda "${MPF_COMMIT}")
MPF_PYTHON_CHANNEL=$(rapids-get-pr-artifact rapidsmpf "${MPF_PR}" python conda "${MPF_COMMIT}")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate Python testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

CMD="rapids-dependency-file-generator --output conda"

# Add file-key options for each argument
for KEY in "$@"; do
  CMD="${CMD} --file-key \"${KEY}\""
done

DEPENDENCY_MATRIX="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}"
if [[ -n "${CUDF_EXTRA_DEPENDENCY_MATRIX:-}" ]]; then
  DEPENDENCY_MATRIX="${DEPENDENCY_MATRIX};${CUDF_EXTRA_DEPENDENCY_MATRIX}"
fi

CMD="${CMD} \
  --prepend-channel \"${MPF_CPP_CHANNEL}\" \
  --prepend-channel \"${MPF_PYTHON_CHANNEL}\" \
  --prepend-channel \"${CPP_CHANNEL}\" \
  --prepend-channel \"${PYTHON_CHANNEL}\" \
  --prepend-channel \"${PYTHON_NOARCH_CHANNEL}\" \
  --matrix \"${DEPENDENCY_MATRIX}\""

eval ${CMD} | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${RESULTS_DIR}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env
