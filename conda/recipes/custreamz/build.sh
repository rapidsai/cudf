# Copyright (c) 2020, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory

# show environment
printenv
# Cleanup local git
git clean -xdf
# build custreamz with verbose output
./build.sh -v custreamz
