#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.

set -e

function upload() {
    echo "UPLOADFILE = ${UPLOADFILE}"
    test -e ${UPLOADFILE}
    source ./travisci/libgdf/upload-anaconda.sh
}

if [ "$BUILD_LIBGDF" == "1" ]; then
    # Upload libgdf
    export UPLOADFILE=`conda build conda-recipes/libgdf -c defaults -c conda-forge --output`
    upload
fi

if [ "$BUILD_CFFI" == "1" ]; then
    # Upload libgdf_cffi
    export UPLOADFILE=`conda build conda-recipes/libgdf_cffi -c defaults -c conda-forge --python=${PYTHON} --output`
    upload
fi
