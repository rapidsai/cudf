#!/bin/bash

# Copyright (c) 2018, NVIDIA CORPORATION.
#########################################
# cuDF GPU build and test script for CI #
#########################################
set -ex

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Set versions of packages needed to be grabbed
export NVSTRINGS_VERSION=0.7.*
export RMM_VERSION=0.7.*

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install -c rapidsai -c rapidsai-nightly -c nvidia -c conda-forge \
    cudatoolkit=${CUDA_REL} rmm=${RMM_VERSION} nvstrings=${NVSTRINGS_VERSION} \
    sphinx sphinx_rtd_theme numpydoc sphinxcontrib-websupport nbsphinx ipython \
    pandoc=\<2.0.0 recommonmark doxygen

pip install sphinx-markdown-tables
logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcudf and cuDF from source
################################################################################

logger "Build libcudf..."
$WORKSPACE/build.sh clean libcudf cudf

################################################################################
# BUILD - Build docs
################################################################################

#libcudf Doxygen build
logger "Build libcudf docs..."
cd $WORKSPACE/cpp/doxygen
doxygen Doxyfile

rm -rf /data/docs/libcudf/html/*
mv html/* /data/docs/libcudf/html

#cudf Sphinx Build
logger "Build cuDF docs..."
cd $WORKSPACE/docs
make html

rm -rf /data/docs/cudf/html/*
mv build/html/* /data/docs/cudf/html
