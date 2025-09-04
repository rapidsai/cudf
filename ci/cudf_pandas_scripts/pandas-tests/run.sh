#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Check GPU usage"
nvidia-smi

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

timeout 90m bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
  --numprocesses 6 \
  --tb=line \
  -vv \
  --disable-warnings \
  -m "not slow and not single_cpu and not db and not network" \
  --max-worker-restart=3 \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas.xml" \
  --dist worksteal \
  --report-log="${PANDAS_TESTS_BRANCH}.json" 2>&1

SUMMARY_FILE_NAME=${PANDAS_TESTS_BRANCH}-${RAPIDS_FULL_VERSION}-results.json
# summarize the results and save them to artifacts:
python python/cudf/cudf/pandas/scripts/summarize-test-results.py --output json pandas-testng/"${PANDAS_TESTS_BRANCH}.json" > "./${SUMMARY_FILE_NAME}"
# RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
# mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
# cp ./"${SUMMARY_FILE_NAME}" "${RAPIDS_ARTIFACTS_DIR}"/
# rapids-upload-to-s3 "${RAPIDS_ARTIFACTS_DIR}"/"${SUMMARY_FILE_NAME}" "${RAPIDS_ARTIFACTS_DIR}"
# mv "${RAPIDS_ARTIFACTS_DIR}"/"${SUMMARY_FILE_NAME}" "${RAPIDS_ARTIFACTS_DIR}"/${PANDAS_TESTS_BRANCH}-results.json


# Exit early if running tests for main branch
if [[ "${PANDAS_TESTS_BRANCH}" == "main" ]]; then
    rapids-logger "Exiting early for main branch testing"
    exit ${EXITCODE}
fi

# mv "${RAPIDS_ARTIFACTS_DIR}"/${PANDAS_TESTS_BRANCH}-results.json ./${PANDAS_TESTS_BRANCH}-results.json

# Hard-coded needs to match the version deduced by rapids-upload-artifacts-dir
GH_JOB_NAME="pandas-tests / build"
RAPIDS_FULL_VERSION=$(<./VERSION)
rapids-logger "Github job name: ${GH_JOB_NAME}"
rapids-logger "Rapids version: ${RAPIDS_FULL_VERSION}"


rapids-logger "Fetching latest available results from nightly"

MAIN_RUN_ID=$(gh run list -w "Pandas Test Job" -b branch-25.10 --status success --limit 7 --json databaseId --jq ".[0].databaseId")
gh run download $MAIN_RUN_ID -n main-results.json
ls -al
# gh run download $GITHUB_RUN_ID -n pr-results.json
# Compute the diff and prepare job summary:
python ci/cudf_pandas_scripts/pandas-tests/job-summary.py main-results.json pr-results.json "${RAPIDS_FULL_VERSION}" | tee summary.txt >> "$GITHUB_STEP_SUMMARY"

COMMENT=$(head -1 summary.txt | grep -oP '\d+/\d+ \(\d+\.\d+%\).*?(a decrease by|an increase by) \d+\.\d+%')
echo "$COMMENT"
jq --arg COMMENT "$COMMENT" --arg GH_JOB_NAME "$GH_JOB_NAME" -n \
  '{"context": "Pandas tests",
    "description": $COMMENT,
    "state":"success",
    "job_name": $GH_JOB_NAME}' \
    > gh-status.json

rapids-logger "Last Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
