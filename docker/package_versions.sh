#!/usr/bin/env bash
# Usage:
# "./package_versions.sh /cudf/conda/environments/cudf_dev.yml" - Updates package versions in file based on Docker build-args

FILENAME=$1

set_version() {
    sed -i "s/\- $1\([<>=][^a-zA-Z]*\)\?$/\- $1=$2/" $FILENAME
}

replace_text() {
    sed -i "s/$1/$2/" $FILENAME
}

add_package() {
    sed -i "s/\- $1\([<>=][^a-zA-Z]*\)\?$/a \- $2=$3/" $FILENAME
}

if [ "$PYTHON_VERSION" ]; then
    PACKAGE_NAME="python"
    set_version "$PACKAGE_NAME" "$PYTHON_VERSION"
fi

if [ "$NUMBA_VERSION" ]; then
    PACKAGE_NAME="numba"
    set_version "$PACKAGE_NAME" "$NUMBA_VERSION"
fi

if [ "$PANDAS_VERSION" ]; then
    PACKAGE_NAME="pandas"
    set_version "$PACKAGE_NAME" "$PANDAS_VERSION"
fi

if [ "$PYARROW_VERSION" ]; then
    PACKAGE_NAME="pyarrow"
    set_version "$PACKAGE_NAME" "$PYARROW_VERSION"
fi

if [ "$CYTHON_VERSION" ]; then
    PACKAGE_NAME="cython"
    set_version "$PACKAGE_NAME" "$CYTHON_VERSION"
fi

if [ "$CMAKE_VERSION" ]; then
    PACKAGE_NAME="cmake"
    set_version "$PACKAGE_NAME" "$CMAKE_VERSION"
fi

if [ "${CUDA_VERSION:0:3}" == "9.2" ]; then
    RAPIDS_CUDA_VERSION="9.2"
elif [ "${CUDA_VERSION:0:4}" == "10.0" ]; then
    RAPIDS_CUDA_VERSION="10.0"
fi

if [ "$RAPIDS_CUDA_VERSION" ]; then
    replace_text "nvidia" "nvidia\/label\/cuda$RAPIDS_CUDA_VERSION"
    replace_text "rapidsai" "rapidsai\/label\/cuda$RAPIDS_CUDA_VERSION"
fi

if [ "$NUMPY_VERSION" ]; then
    ABOVE_PACKAGE="pandas"
    PACKAGE_NAME="numpy"
    add_package "$ABOVE_PACKAGE" "$PACKAGE_NAME" "$NUMPY_VERSION"
fi
