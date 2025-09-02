#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download the summarized results of running the Pandas tests on both the main
# branch and the PR branch:

# Hard-coded needs to match the version deduced by rapids-upload-artifacts-dir
GH_JOB_NAME="pandas-tests-diff / build"
RAPIDS_FULL_VERSION=$(<./VERSION)
rapids-logger "Github job name: ${GH_JOB_NAME}"
rapids-logger "Rapids version: ${RAPIDS_FULL_VERSION}"

rapids-logger "Fetching latest available results from nightly"

MAIN_RUN_ID=$(gh run list -w "Pandas Test Job" -b branch-25.10 --status success --limit 7 --json databaseId --jq ".[0].databaseId")
gh run download $MAIN_RUN_ID -n main-results.json
gh run download $GITHUB_RUN_ID -n pr-results.json
# Compute the diff and prepare job summary:
python -m pip install pandas tabulate
python ci/cudf_pandas_scripts/pandas-tests/job-summary.py main-results.json pr-results.json "${RAPIDS_FULL_VERSION}" | tee summary.txt >> "$GITHUB_STEP_SUMMARY"

COMMENT=$(head -1 summary.txt | grep -oP '\d+/\d+ \(\d+\.\d+%\).*?(a decrease by|an increase by) \d+\.\d+%')
echo "$COMMENT"
jq --arg COMMENT "$COMMENT" --arg GH_JOB_NAME "$GH_JOB_NAME" -n \
  '{"context": "Pandas tests",
    "description": $COMMENT,
    "state":"success",
    "job_name": $GH_JOB_NAME}' \
    > gh-status.json
