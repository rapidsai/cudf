# Copyright (c) 2020, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory

# show environment
printenv
# Cleanup local git
git clean -xdf
# build libcudf with verbose output
./build.sh -v libcudf_kafka
