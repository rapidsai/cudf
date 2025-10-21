#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcudf/" || exit

export GTEST_CUDF_RMM_MODE=cuda
export GTEST_BRIEF=1
# compute-sanitizer bug 4553815
export LIBCUDF_MEMCHECK_ENABLED=1
for gt in ./*_TEST ; do
  test_name=$(basename "${gt}")
  # Run gtests with compute-sanitizer
  echo "Running compute-sanitizer on $test_name"
  compute-sanitizer --tool memcheck "${gt}" "$@"
done
unset GTEST_BRIEF
unset GTEST_CUDF_RMM_MODE
unset LIBCUDF_MEMCHECK_ENABLED

exit ${EXITCODE}
