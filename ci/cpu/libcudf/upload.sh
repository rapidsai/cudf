#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.

set -e

function upload() {
    echo "UPLOADFILE = ${UPLOADFILE}"
    test -e ${UPLOADFILE}
    source ./ci/cpu/libcudf/upload-anaconda.sh
}

if [ "$BUILD_LIBCUDF" == "1" ]; then
    # Upload libcudf
    export UPLOADFILE=`conda build conda/recipes/libcudf -c defaults -c conda-forge --output`
    upload
fi

if [ "$BUILD_CFFI" == "1" ]; then
    # Upload libcudf_cffi
    export UPLOADFILE=`conda build conda/recipes/libcudf_cffi -c defaults -c conda-forge --python=${PYTHON} --output`
    upload
fi
