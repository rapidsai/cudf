#!/usr/bin/env bash

# Copyright (c) 2020, NVIDIA CORPORATION.
set -e

#Always upload cudf Python package
export UPLOAD_CUDF=1

#Upload libcudf once per CUDA
if [[ "$PYTHON" == "3.7" ]]; then
    export UPLOAD_LIBCUDF=1
else
    export UPLOAD_LIBCUDF=0
fi

# upload cudf_kafka for all versions of Python
if [[ "$CUDA" == "10.1" ]]; then
    export UPLOAD_CUDF_KAFKA=1
else
    export UPLOAD_CUDF_KAFKA=0
fi

#We only want to upload libcudf_kafka once per python/CUDA combo
if [[ "$PYTHON" == "3.7" ]] && [[ "$CUDA" == "10.1" ]]; then
    export UPLOAD_LIBCUDF_KAFKA=1
else
    export UPLOAD_LIBCUDF_KAFKA=0
fi
