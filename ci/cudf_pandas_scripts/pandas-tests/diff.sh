#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download the summarized results of running the Pandas tests on both the main
# branch and the PR branch:

# Hard-coded needs to match the version deduced by rapids-upload-artifacts-dir
MAIN_ARTIFACT=$(rapids-s3-path)cuda12_$(arch)_py310.main-results.json
PR_ARTIFACT=$(rapids-s3-path)cuda12_$(arch)_py310.pr-results.json
aws s3 cp $MAIN_ARTIFACT main-results.json
aws s3 cp $PR_ARTIFACT pr-results.json

# Compute the diff and prepare job summary:
python -m pip install pandas tabulate
python ci/cudf_pandas_scripts/pandas-tests/job-summary.py main-results.json pr-results.json | tee summary.txt >> "$GITHUB_STEP_SUMMARY"

COMMENT=$(head -1 summary.txt)

echo "$COMMENT"

# Magic name that the custom-job.yaml workflow reads and re-exports
echo "job_output=${COMMENT}" >> "${GITHUB_OUTPUT}"
