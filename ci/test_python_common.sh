#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Common setup steps shared by Python test jobs

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-package-name conda_python cudf --stable --cuda)")
PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github "$(rapids-package-name conda_python cudf --pure --cuda "${RAPIDS_CUDA_VERSION}")")

rapids-logger "Generate Python testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

CMD="rapids-dependency-file-generator --output conda"

# Add file-key options for each argument
for KEY in "$@"; do
  CMD="${CMD} --file-key \"${KEY}\""
done

CMD="${CMD} \
  --prepend-channel \"${CPP_CHANNEL}\" \
  --prepend-channel \"${PYTHON_CHANNEL}\" \
  --prepend-channel \"${PYTHON_NOARCH_CHANNEL}\" \
  --matrix \"cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}\""

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
