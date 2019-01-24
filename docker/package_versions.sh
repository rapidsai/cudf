#!/usr/bin/env bash
# Usage:
# "./package_versions.sh /cudf/conda/environments/cudf_dev.yml" - Updates package versions in file based on Docker build-args

FILENAME=$1

set_version() {
    sed -i "/\- $1[<>=][^a-Z]*/\- $1=$2/" $FILENAME
}

replace_text() {
    sed -i "/$1/$2/" $FILENAME
}

add_package() {
    sed -i "/$1/a \- $2=$3/" $FILENAME
}

if [ -z "$PYTHON_VERSION" ]; then
    PACKAGE_NAME="python"
    set_version "$PACKAGE_NAME" "$PYTHON_VERSION"
fi

if [ -z "$NUMBA_VERSION" ]; then
    PACKAGE_NAME="numba"
    set_version "$PACKAGE_NAME" "$NUMBA_VERSION"
fi

if [ -z "$PANDAS_VERSION" ]; then
    PACKAGE_NAME="pandas"
    set_version "$PACKAGE_NAME" "$PANDAS_VERSION"
fi

if [ -z "$PYARROW_VERSION" ]; then
    PACKAGE_NAME="pyarrow"
    set_version "$PACKAGE_NAME" "$PYARROW_VERSION"
fi

if [ -z "$CYTHON_VERSION" ]; then
    PACKAGE_NAME="cython"
    set_version "$PACKAGE_NAME" "$CYTHON_VERSION"
fi

if [ -z "$CMAKE_VERSION" ]; then
    PACKAGE_NAME="cmake"
    set_version "$PACKAGE_NAME" "$CMAKE_VERSION"
fi

if [ -z "$CUDA_VERSION" ]; then
    replace_text "nvidia" "nvidia\/label\/cuda$CUDA_VERSION"
    replace_text "rapidsai" "rapidsai\/label\/cuda$CUDA_VERSION"
fi

if [ -z "$NUMPY_VERSION" ]; then
    ABOVE_PACKAGE="pandas"
    PACKAGE_NAME="numpy"
    add_package "$ABOVE_PACKAGE" "$PACKAGE_NAME" "$NUMPY_VERSION"
fi
