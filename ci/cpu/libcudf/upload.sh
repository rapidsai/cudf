#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.

set -e

function upload() {
    echo "UPLOADFILE = ${UPLOADFILE}"
    test -e ${UPLOADFILE}
    source ./ci/cpu/libcudf/upload-anaconda.sh
}

CUDA_REL=${CUDA:0:3}
if [ "${CUDA:0:2}" == '10' ]; then
  # CUDA 10 release
  CUDA_REL=${CUDA:0:4}
fi

if [ "$UPLOAD_LIBCUDF" == "1" ]; then
  # Upload libcudf
  export UPLOADFILE=`conda build conda/recipes/libcudf --output`
  upload
fi

if [ "$UPLOAD_CUDF" == "1" ]; then
  # Upload libcudf_cffi
  export UPLOADFILE=`conda build conda/recipes/libcudf_cffi --python=$PYTHON --output`
  upload
fi