#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_python_cudf_cccl.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_pylibcudf test_python_cudf

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Verify cuda-cccl compatibility"
rapids-mamba-retry install cuda-cccl
pip check | grep -q numba-cuda && exit 1

rapids-logger "pytest cudf"
timeout 40m ./ci/run_cudf_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=../.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-coverage.xml" \
  --cov-report=term


rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
