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
#printenv | sort

if [[ -d $RAPIDS_BMR_DIR ]]; then
  ls -l ${RAPIDS_BMR_DIR}
fi

echo $RAPIDS_REF_NAME
echo $RAPIDS_SHA
echo ${RAPIDS_REF_NAME}/${RAPIDS_SHA:0:5}

FILE=${RAPIDS_BMR_DIR}/ninja.log
if [[ -f $FILE ]]; then
  UPLOAD_NAME=cpp_cuda${RAPIDS_CUDA_VERSION%%.*}_$(arch).ninja.log
  rapids-upload-to-s3 "${UPLOAD_NAME}" "${FILE}"
fi

FILE=${RAPIDS_BMR_DIR}/ninja_log.html
if [[ -f $FILE ]]; then
  UPLOAD_NAME=cpp_cuda${RAPIDS_CUDA_VERSION%%.*}_$(arch).BuildMetricsReport.html
  rapids-upload-to-s3 "${UPLOAD_NAME}" "${FILE}"
fi

echo "++++++++++++++++++++++++++++++++++++++++++++"
