#!/usr/bin/env bash

pushd $(pwd)

cd ~/cudf/cpp/build/latest
ninja -j16 ORC_TEST

popd
