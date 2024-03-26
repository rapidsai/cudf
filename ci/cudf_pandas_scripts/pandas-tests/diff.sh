#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download the summarized results of running the Pandas tests on both the main
# branch and the PR branch:

# Hard-coded needs to match the version deduced by rapids-upload-artifacts-dir
MAIN_ARTIFACT=$(rapids-s3-path)cuda12_$(arch)_py310.main-results.json
PR_ARTIFACT=$(rapids-s3-path)cuda12_$(arch)_py39.pr-results.json

rapids-logger "abc"
rapids-logger "abc-1"
aws s3api list-objects-v2 --bucket rapids-downloads --prefix "nightly/" --query "sort_by(Contents[?ends_with(Key, '.main-results.json')], &LastModified)[::-1].[Key]" --output text > s3_output.txt
# aws s3api list-objects-v2 --bucket rapids-downloads --prefix "nightly/" --query 'sort_by(Contents, &LastModified)[*].{Key: Key, LastModified: LastModified}' --output text > s3_output.txt
# aws s3 ls s3://rapids-downloads/nightly/cudf/ --recursive --output text > s3_output.txt
# grep "-results.json" s3_output.txt
cat s3_output.txt
read -r COMPARE_ENV < s3_output.txt
export COMPARE_ENV
rapids-logger "Got ENV: ${COMPARE_ENV}"
rapids-logger "abc-exit"

aws s3 cp "s3://rapids-downloads/${COMPARE_ENV}" main-results.json
aws s3 cp $PR_ARTIFACT pr-results.json

# Compute the diff and prepare job summary:
python -m pip install pandas tabulate
python ci/cudf_pandas_scripts/pandas-tests/job-summary.py main-results.json pr-results.json | tee summary.txt >> "$GITHUB_STEP_SUMMARY"

# COMMENT=$(head -n 10 summary.txt)
# COMMENT=$(cat summary.txt)
# read -r COMMENT < summary.txt
# export COMMENT

# echo "$COMMENT"
# rapids-logger "comment: ${COMMENT}"
cat summary.txt
# rapids-logger "EOF"

# Magic name that the custom-job.yaml workflow reads and re-exports
# echo "job_output=${COMMENT}" >> $GITHUB_OUTPUT
# echo "EOF" >> $GITHUB_ENV
RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
mv summary.txt ${RAPIDS_ARTIFACTS_DIR}/
rapids-upload-to-s3 ${RAPIDS_ARTIFACTS_DIR}/summary.txt "${RAPIDS_ARTIFACTS_DIR}"
ART_URL="$(rapids-s3-path)${RAPIDS_ARTIFACTS_DIR}/summary.txt"
echo "job_output=${ART_URL}" >> $GITHUB_OUTPUT


# https://downloads.rapids.ai/ci/cudf/pull-request/15369/6f62013//__w/cudf/cudf/artifacts/summary.txt

# s3://rapids-downloads/ci/cudf/pull-request/15369/6f62013//__w/cudf/cudf/artifacts/summary.txt
