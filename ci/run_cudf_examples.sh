#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the examples' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf/";

compute-sanitizer --tool memcheck basic_example

compute-sanitizer --tool memcheck deduplication

compute-sanitizer --tool memcheck custom_optimized names.csv
compute-sanitizer --tool memcheck custom_prealloc names.csv
compute-sanitizer --tool memcheck custom_with_malloc names.csv

exit ${EXITCODE}
