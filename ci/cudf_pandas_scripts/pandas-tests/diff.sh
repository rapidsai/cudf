#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download the summarized results of running the Pandas tests on both the main
# branch and the PR branch:

# Hard-coded needs to match the version deduced by rapids-upload-artifacts-dir
GH_JOB_NAME="pandas-tests-diff / build"
rapids-logger "Github job name: ${GH_JOB_NAME}"

PY_VER="39"
MAIN_ARTIFACT=$(rapids-s3-path)cuda12_$(arch)_py${PY_VER}.main-results.json
PR_ARTIFACT=$(rapids-s3-path)cuda12_$(arch)_py${PY_VER}.pr-results.json

rapids-logger "Fetching latest available results from nightly"
aws s3api list-objects-v2 --bucket rapids-downloads --prefix "nightly/" --query "sort_by(Contents[?ends_with(Key, '_py${PY_VER}.main-results.json')], &LastModified)[::-1].[Key]" --output text > s3_output.txt

read -r COMPARE_ENV < s3_output.txt
export COMPARE_ENV
rapids-logger "Latest available results from nightly: ${COMPARE_ENV}"

aws s3 cp "s3://rapids-downloads/${COMPARE_ENV}" main-results.json
aws s3 cp $PR_ARTIFACT pr-results.json

# Compute the diff and prepare job summary:
python -m pip install pandas tabulate
python ci/cudf_pandas_scripts/pandas-tests/job-summary.py main-results.json pr-results.json | tee summary.txt >> "$GITHUB_STEP_SUMMARY"

COMMENT=$(head -1 summary.txt | grep -oP '\d+/\d+ \(\d+\.\d+%\).*?(a decrease by|an increase by) \d+\.\d+%')
echo "$COMMENT"
jq --arg COMMENT "$COMMENT" --arg GH_JOB_NAME "$GH_JOB_NAME" -n \
  '{"context": "Pandas tests",
    "description": $COMMENT,
    "state":"success",
    "job_name": $GH_JOB_NAME}' \
    > gh-status.json
