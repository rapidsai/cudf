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

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

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

logger "Build conda pkg for libcudf..."
source ci/cpu/libcudf/build_libcudf.sh

logger "Build conda pkg for libcudf_kafka..."
source ci/cpu/libcudf_kafka/build_libcudf_kafka.sh

logger "Build conda pkg for cudf..."
source ci/cpu/cudf/build_cudf.sh

logger "Build conda pkg for dask-cudf..."
source ci/cpu/dask-cudf/build_dask_cudf.sh

logger "Build conda pkg for cudf_kafka..."
source ci/cpu/cudf_kafka/build_cudf_kafka.sh

logger "Build conda pkg for custreamz..."
source ci/cpu/custreamz/build_custreamz.sh
################################################################################
# UPLOAD - Conda packages
################################################################################

logger "Upload conda pkgs..."
source ci/cpu/upload_anaconda.sh
