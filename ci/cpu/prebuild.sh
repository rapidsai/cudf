#!/usr/bin/env bash

# Copyright (c) 2020-2022, NVIDIA CORPORATION.
set -e

#Always upload cudf packages
export UPLOAD_CUDF=1
export UPLOAD_LIBCUDF=1
export UPLOAD_CUDF_KAFKA=1

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    #If project flash is not activate, always build both
    export BUILD_LIBCUDF=1
    export BUILD_CUDF=1
fi
