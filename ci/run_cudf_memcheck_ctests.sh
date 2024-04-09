#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcudf/";

export GTEST_CUDF_RMM_MODE=cuda
# compute-sanitizer bug 4553815
export LIBCUDF_MEMCHECK_ENABLED=1
for gt in ./*_TEST ; do
  test_name=$(basename ${gt})
  # Run gtests with compute-sanitizer
  if [[ "$test_name" == "ERROR_TEST" ]] || [[ "$test_name" == "STREAM_IDENTIFICATION_TEST" ]]; then
    continue
  fi
  echo "Running compute-sanitizer on $test_name"
  compute-sanitizer --tool memcheck ${gt} "$@"
done
unset GTEST_CUDF_RMM_MODE
unset LIBCUDF_MEMCHECK_ENABLED

exit ${EXITCODE}
