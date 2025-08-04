#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

PANDAS_TESTS_BRANCH=${1}
RAPIDS_FULL_VERSION=$(<./VERSION)
rapids-logger "Running Pandas tests using $PANDAS_TESTS_BRANCH branch and rapids-version $RAPIDS_FULL_VERSION"
rapids-logger "PR number: ${RAPIDS_REF_NAME:-"unknown"}"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the cudf, libcudf, and pylibcudf built in the previous step
CUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
  "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test,pandas-tests]" \
  "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
  "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
  --numprocesses 5 \
  --tb=line \
  --disable-warnings \
  -m "not slow and not single_cpu and not db and not network" \
  --max-worker-restart=3 \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas.xml" \
  --dist worksteal \
  --report-log="${PANDAS_TESTS_BRANCH}.json" 2>&1

SUMMARY_FILE_NAME=${PANDAS_TESTS_BRANCH}-${RAPIDS_FULL_VERSION}-results.json
# summarize the results and save them to artifacts:
python python/cudf/cudf/pandas/scripts/summarize-test-results.py --output json pandas-testing/"${PANDAS_TESTS_BRANCH}.json" > "pandas-testing/${SUMMARY_FILE_NAME}"
RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
mv pandas-testing/"${SUMMARY_FILE_NAME}" "${RAPIDS_ARTIFACTS_DIR}"/
rapids-upload-to-s3 "${RAPIDS_ARTIFACTS_DIR}"/"${SUMMARY_FILE_NAME}" "${RAPIDS_ARTIFACTS_DIR}"
rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
