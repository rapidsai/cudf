#!/usr/bin/env bash

export BUILD_ABI=1
export BUILD_CUDF=1
export BUILD_CFFI=1

if [[ "$PYTHON" == "3.6" ]]; then
    export BUILD_LIBCUDF=1
else
    export BUILD_LIBCUDF=0
fi