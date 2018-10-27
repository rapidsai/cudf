CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release"


if [ -n "$MACOSX_DEPLOYMENT_TARGET" ]; then
    # C++11 requires 10.9
    # but cudatoolkit 8 is build for 10.11
    export MACOSX_DEPLOYMENT_TARGET=10.11
fi

# Cleanup local git
git clean -xdf
# Change directory for build process
cd libgdf
# Patch tests for CUDA 10 skipping hash_map
## See https://github.com/rapidsai/libgdf/pull/149
if [ ${CUDA:0:4} == '10.0' ]; then
  sed -i '/hash_map/d' ./src/tests/CMakeLists.txt
fi
# Use CMake-based build procedure
mkdir build
cd build
# configure
cmake $CMAKE_COMMON_VARIABLES ..
# build
make -j$CPU_COUNT install
