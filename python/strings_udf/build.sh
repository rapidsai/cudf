#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.

REPODIR=`git rev-parse --show-toplevel`

BUILD_DIR=${REPODIR}/python/strings_udf/cpp/build

mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

cmake .. -DCONDA_PREFIX=${CONDA_PREFIX} -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}/
make
make install

cd $REPODIR/python/strings_udf/
python setup.py install
