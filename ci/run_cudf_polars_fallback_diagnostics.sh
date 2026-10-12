#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

report_dir=$(mktemp -d)
audit_args=()
publish_audit=false
if [[ "${1:-}" == "--audit" || "${1:-}" == "--publish-audit" ]]; then
    [[ "$1" == "--publish-audit" ]] && publish_audit=true
    shift
    gh api --paginate --slurp \
        "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?filter=latest" \
        | jq '[.[].jobs[] | select(.name | startswith("cudf-polars-polars-tests")) | select(any(.steps[]?; .name == "Run tests"))]' \
        > "${report_dir}/expected-jobs"
    audit_args=(--audit-run-list ci/cudf_polars/gpu_test_run_list.json --expected-jobs "${report_dir}/expected-jobs")
fi
output_dir="${1:-cudf-polars-fallback-diagnostics}"

# The shared test workflows upload RAPIDS_ARTIFACTS_DIR even after test failures.
# Restrict downloads to this attempt so reruns cannot reuse earlier diagnostics.
aws s3 cp "$(rapids-s3-path)" "${report_dir}/" --recursive \
    --exclude '*' \
    --include "*.cudf-polars-fallback-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-*.json" \
    --only-show-errors

audit_status=0
python ci/analyze_cudf_polars_fallbacks.py \
    --reports-dir "${report_dir}" \
    --output-dir "${output_dir}" \
    --summary-file "${GITHUB_STEP_SUMMARY}" \
    "${audit_args[@]}" || audit_status=$?

# The shared workflow uploads artifacts only after a successful script step.
# Nightlies publish first, then gate on the uploaded audit in a dependent job.
if ! ${publish_audit}; then
    exit "${audit_status}"
fi
