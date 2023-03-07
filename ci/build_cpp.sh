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

if [[ -d $RAPIDS_ARTIFACTS_DIR ]]; then
  ls -l ${RAPIDS_ARTIFACTS_DIR}
fi

echo "++++++++++++++++++++++++++++++++++++++++++++"

FILE=${RAPIDS_ARTIFACTS_DIR}/ninja.log
if [[ -f $FILE ]]; then
  echo -e "\x1B[33;1m\x1B[48;5;240m Ninja log for this build available at the following link \x1B[0m"
  UPLOAD_NAME=cpp_cuda${RAPIDS_CUDA_VERSION%%.*}_$(arch).ninja.log
  rapids-upload-to-s3 "${UPLOAD_NAME}" "${FILE}"
fi

echo "++++++++++++++++++++++++++++++++++++++++++++"

FILE=${RAPIDS_ARTIFACTS_DIR}/ninja_log.html
if [[ -f $FILE ]]; then
  echo -e "\x1B[33;1m\x1B[48;5;240m Build Metrics Report for this build available at the following link \x1B[0m"
  UPLOAD_NAME=cpp_cuda${RAPIDS_CUDA_VERSION%%.*}_$(arch).BuildMetricsReport.html
  rapids-upload-to-s3 "${UPLOAD_NAME}" "${FILE}"
fi

echo "++++++++++++++++++++++++++++++++++++++++++++"
