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

if [ "$BUILD_LIBCUDF" == "1" ]; then
  # Upload libcudf
  if [ "$BUILD_ABI" == "1" ]; then
    export UPLOADFILE=`conda build conda/recipes/libcudf -c nvidia/label/cuda${CUDA_REL} -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} -c numba -c conda-forge -c defaults --output`
  else
    export UPLOADFILE=`conda build conda/recipes/libcudf -c nvidia/label/cf201901-cuda${CUDA_REL} -c rapidsai/label/cf201901-cuda${CUDA_REL} -c rapidsai-nightly/label/cf201901-cuda${CUDA_REL} -c numba -c conda-forge/label/cf201901 -c defaults --output`
  fi
  upload
fi

if [ "$BUILD_CFFI" == "1" ]; then
  # Upload libcudf_cffi
  if [ "$BUILD_ABI" == "1" ]; then
    export UPLOADFILE=`conda build conda/recipes/libcudf_cffi -c nvidia/label/cuda${CUDA_REL} -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} -c numba -c conda-forge -c defaults --python=$PYTHON --output`
  else
    export UPLOADFILE=`conda build conda/recipes/libcudf_cffi -c nvidia/label/cf201901-cuda${CUDA_REL} -c rapidsai/label/cf201901-cuda${CUDA_REL} -c rapidsai-nightly/label/cf201901-cuda${CUDA_REL} -c numba -c conda-forge/label/cf201901 -c defaults --python=$PYTHON --output`
  fi
  upload
fi