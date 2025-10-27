#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_java \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

export CMAKE_GENERATOR=Ninja

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# disable large strings
export LIBCUDF_LARGE_STRINGS_ENABLED=0

rapids-logger "Run Java tests"
pushd java
mvn test -B -DCUDF_JNI_ENABLE_PROFILING=OFF
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
