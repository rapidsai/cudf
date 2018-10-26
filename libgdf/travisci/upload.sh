#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.

set -e

if [ $BUILD_CFFI == 1 ]; then
    export UPLOADFILE=`conda build conda-recipes/libgdf_cffi -c defaults -c conda-forge --python=${PYTHON} --output`
else
    export UPLOADFILE=`conda build conda-recipes/libgdf -c defaults -c conda-forge --output`
fi

echo "UPLOADFILE = ${UPLOADFILE}"
test -e ${UPLOADFILE}
source ./travisci/upload-anaconda.sh
