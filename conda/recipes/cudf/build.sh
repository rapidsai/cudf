# Copyright (c) 2018-2024, NVIDIA CORPORATION.

export PIP_VERBOSE=2

# This assumes the script is executed from the root of the repo directory
./build.sh cudf

unset PIP_VERBOSE
