#!/bin/bash
## Copyright (c) 2018, NVIDIA CORPORATION.
## Docs Build for cuDF and nvStrings

set -e

export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export HOME=$WORKSPACE
export WORKSPACE=/rapids/cudf
export LIBCUDF_KERNEL_CACHE_PATH="$HOME/.jitify-cache"
export NIGHTLY_VERSION=$(echo $BRANCH_VERSION | awk -F. '{print $2}')
export PROJECTS=(libcudf cudf nvstrings libnvstrings)

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate rapids
conda install -c anaconda beautifulsoup4 jq
pip install sphinx-markdown-tables


logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list


#libcudf Doxygen build
logger "Build libcudf docs..."
cd $WORKSPACE/cpp/doxygen
doxygen Doxyfile

#cudf Sphinx Build
logger "Build cuDF docs..."
cd $WORKSPACE/docs/cudf
make html

#libnvstrings Doxygen build
logger "Build livnvstrings docs..."
cd $WORKSPACE/cpp/custrings/doxygen
doxygen Doxyfile

#nvstrings Sphinx Build
logger "Build nvstrings docs..."
cd $WORKSPACE/docs/nvstrings
make html

#Move HTML to project folders
mv $WORKSPACE/cpp/doxygen/html/* $WORKSPACE/tmp/libcudf/html/*
mv $WORKSPACE/docs/cudf/build/html/* $WORKSPACE/tmp/cudf/html/*
mv $WORKSPACE/cpp/custrings/doxygen/html/* $WORKSPACE/tmp/libnvstrings/html/*
mv $WORKSPACE/docs/nvstrings/build/html/* $WORKSPACE/tmp/nvstrings/html/*
