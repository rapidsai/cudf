#!/usr/bin/env bash

pushd $(pwd)

# Reconfigure libcudf to use debug build
clean-all
configure-cudf-cpp -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_BENCHMARKS=ON -DBUILD_TESTS=ON -j 16

# Rebuild libcudf
# In nvcc, compiler options -E and -MD cannot be used together. However:
# - Adding the device debug symbol (-G) for a selected set of .cu files
#   appears to cause -MD to be added as a compiler option.
# - sccache always forces -E to be appended even when export CCACHE_DISABLE=1
#   is used.
# Given this conundrum, the hacky solution is to remove the use of sccache
# from the build.ninja file, before building libcudf with the device
# debug symbol added.
cd ~/cudf/cpp/build/latest
sed -i 's|LAUNCHER = /usr/bin/sccache|LAUNCHER =|g' build.ninja
ninja -j16 cudf ORC_TEST

popd

# Build Python package
build-pylibcudf-python -j 16
build-cudf-python -j 16