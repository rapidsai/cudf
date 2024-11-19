#!/usr/bin/env bash

pushd $(pwd)

cd ~
clean-all
configure-cudf-cpp -DCMAKE_BUILD_TYPE=Debug -j 16 -DCMAKE_CUDA_ARCHITECTURES=86

cd ~/cudf/cpp/build/pip/cuda-12.5/debug
sed -i 's|LAUNCHER = /usr/bin/sccache|LAUNCHER =|g' build.ninja
ninja -j16 cudf

cd ~/cudf/cpp/examples
./build.sh
popd
