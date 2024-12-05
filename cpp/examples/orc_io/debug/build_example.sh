#!/usr/bin/env bash

pushd $(pwd)

# Build cuDF after source changes
cd ~/cudf/cpp/build/pip/cuda-12.5/debug
ninja -j16 cudf

popd


# Build the example after source changes
cmake --build ~/cudf/cpp/examples/orc_io/build -j 16
