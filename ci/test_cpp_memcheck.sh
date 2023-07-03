#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

source "$(dirname "$0")/test_cpp_common.sh"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run gtests with compute-sanitizer
rapids-logger "Memcheck gtests with rmm_mode=cuda"
export GTEST_CUDF_RMM_MODE=cuda
COMPUTE_SANITIZER_CMD="compute-sanitizer --tool memcheck"
for gt in "$CONDA_PREFIX"/bin/gtests/libcudf/*_TEST ; do
    test_name=$(basename ${gt})
    if [[ "$test_name" == "ERROR_TEST" ]] || [[ "$test_name" == "STREAM_IDENTIFICATION_TEST" ]]; then
        continue
    fi
    echo "Running compute-sanitizer on $test_name"
    ${COMPUTE_SANITIZER_CMD} ${gt} --gtest_output=xml:"${RAPIDS_TESTS_DIR}${test_name}.xml"
done
unset GTEST_CUDF_RMM_MODE

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
