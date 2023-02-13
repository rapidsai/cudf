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
printenv | sort
echo "++++++++++++++++++++++++++++++++++++++++++++"
ls -l ${RAPIDS_BMR_DIR}

UPLOAD_NAME=cpp_cuda${RAPIDS_CUDA_VERSION%%.*}_$(arch).ninja.log
FILE=${RAPIDS_BMR_DIR}/ninja.log
rapids-upload-to-s3 "${UPLOAD_NAME}" "${FILE}"

UPLOAD_NAME=cpp_cuda${RAPIDS_CUDA_VERSION%%.*}_$(arch).BuildMetricsReport.html
FILE=${RAPIDS_BMR_DIR}/ninja_log.html
rapids-upload-to-s3 "${UPLOAD_NAME}" "${FILE}"
