# Copyright (c) 2018-2019, NVIDIA CORPORATION.

CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release"

# show environment
printenv
# Cleanup local git
git clean -xdf
# Change directory for build process
cd cpp
# Use CMake-based build procedure
mkdir build
cd build
# configure
cmake $CMAKE_COMMON_VARIABLES ..
# build
make -j${PARALLEL_LEVEL} VERBOSE=1 install
