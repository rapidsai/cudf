#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf/";

cp ./basic/4stock_5day.csv ./basic/build
compute-sanitizer --tool memcheck ./basic/build/basic_example

cp ./nested_types/example.json ./nested_types/build
compute-sanitizer --tool memcheck ./nested_types/build/deduplication

cp ./strings/names.csv ./strings/build
compute-sanitizer --tool memcheck ./strings/build/custom_optimized ./strings/build/names.csv
compute-sanitizer --tool memcheck ./strings/build/custom_prealloc ./strings/build/names.csv
compute-sanitizer --tool memcheck ./strings/build/custom_with_malloc ./strings/build/names.csv

exit ${EXITCODE}
