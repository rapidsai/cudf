#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

source "$(dirname "$0")/test_cpp_common.sh"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run gtests with compute-sanitizer
compute-sanitizer --tool memcheck "$CONDA_PREFIX"/bin/gtests/libcudf/PARQUET_TEST --gtest_filter=ParquetWriterNumericTypeTest/3.SingleColumnWithNulls --rmm_mode=cuda
compute-sanitizer --tool memcheck "$CONDA_PREFIX"/bin/gtests/libcudf/ORC_TEST --gtest_filter=OrcWriterNumericTypeTest/6.SingleColumn --rmm_mode=cuda

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
