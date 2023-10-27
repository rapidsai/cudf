#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Download the summarized results of running the Pandas tests on both the main
# branch and the PR branch:

# Hard-coded needs to match the version deduced by rapids-upload-artifacts-dir
MAIN_ARTIFACT=$(rapids-s3-path)cuda12_$(arch)_py310.main-results.json
PR_ARTIFACT=$(rapids-s3-path)cuda12_$(arch)_py310.pr-results.json
aws s3 cp $MAIN_ARTIFACT main-results.json
aws s3 cp $PR_ARTIFACT pr-results.json

# Compute the diff and prepare job summary:
python -m pip install pandas tabulate
python ci/xdf_scripts/pandas-tests/job-summary.py main-results.json pr-results.json | tee summary.txt >> "$GITHUB_STEP_SUMMARY"

COMMENT=$(head -1 summary.txt)

echo "$COMMENT"

# Magic name that the custom-job.yaml workflow reads and re-exports
echo "job_output=${COMMENT}" >> "${GITHUB_OUTPUT}"
