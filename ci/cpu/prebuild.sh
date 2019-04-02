#!/usr/bin/env bash

export BUILD_ABI=1
export BUILD_CFFI=1

#Build cudf once per PYTHON
if [[ "$CUDA" == "9.2" ]]; then
    export BUILD_CUDF=1
else
    export BUILD_CUDF=0
fi

#Build libcudf once per CUDA
if [[ "$PYTHON" == "3.6" ]]; then
    export BUILD_LIBCUDF=1
else
    export BUILD_LIBCUDF=0
fi