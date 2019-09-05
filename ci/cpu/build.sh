#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# cuDF CPU conda build script for CI #
######################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# BUILD - Conda package builds (conda deps: libcudf <- libcudf_cffi <- cudf)
################################################################################

logger "Build conda pkg for libNVStrings..."
source ci/cpu/libnvstrings/build_libnvstrings.sh

logger "Build conda pkg for nvstrings..."
source ci/cpu/nvstrings/build_nvstrings.sh

logger "Build conda pkg for libcudf..."
source ci/cpu/libcudf/build_libcudf.sh

logger "Build conda pkg for cudf..."
source ci/cpu/cudf/build_cudf.sh

logger "Build conda pkg for dask-cudf..."
source ci/cpu/dask-cudf/build_dask_cudf.sh

################################################################################
# UPLOAD - Conda packages
################################################################################

logger "Upload conda pkgs..."
source ci/cpu/upload_anaconda.sh
