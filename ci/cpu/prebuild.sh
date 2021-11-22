#!/usr/bin/env bash

# Copyright (c) 2020, NVIDIA CORPORATION.
set -e

ARCH=$(arch)
if [ "${ARCH}" = "x86_64" ]; then
    DEFAULT_CUDA_VER="11.0"
elif [ "${ARCH}" = "aarch64" ]; then
    DEFAULT_CUDA_VER="11.2"
else
    echo "Unsupported arch ${ARCH}"
    exit 1
fi

#Always upload cudf Python package
export UPLOAD_CUDF=1

#Upload libcudf once per CUDA
if [[ "$PYTHON" == "3.7" ]]; then
    export UPLOAD_LIBCUDF=1
else
    export UPLOAD_LIBCUDF=0
fi

# upload cudf_kafka for all versions of Python
if [[ "$CUDA" == "${DEFAULT_CUDA_VER}" ]]; then
    export UPLOAD_CUDF_KAFKA=1
else
    export UPLOAD_CUDF_KAFKA=0
fi

#We only want to upload libcudf_kafka once per python/CUDA combo
if [[ "$PYTHON" == "3.7" ]] && [[ "$CUDA" == "${DEFAULT_CUDA_VER}" ]]; then
    export UPLOAD_LIBCUDF_KAFKA=1
else
    export UPLOAD_LIBCUDF_KAFKA=0
fi

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    #If project flash is not activate, always build both
    export BUILD_LIBCUDF=1
    export BUILD_CUDF=1
fi
