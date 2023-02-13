#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp

echo "++++++++++++++++++++++++++++++++++++++++++++"
printenv
echo "++++++++++++++++++++++++++++++++++++++++++++"

#UPLOAD_NAME=cpp_cuda${RAPIDS_CUDA_VERSION%%.*}_$(arch).ninja_log
#FILE=$(echo /opt/conda/conda-bld/*libraft-split*/cpp/build/.ninja_log)
#rapids-upload-to-s3 "${UPLOAD_NAME}" "${FILE}"
