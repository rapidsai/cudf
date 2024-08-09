#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PANDAS_TESTS_BRANCH=${1}
RAPIDS_FULL_VERSION=$(<./VERSION)
rapids-logger "Running Pandas tests using $PANDAS_TESTS_BRANCH branch and rapids-version $RAPIDS_FULL_VERSION"
rapids-logger "PR number: ${RAPIDS_REF_NAME:-"unknown"}"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-cudf-dep
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./local-cudf-dep

# --- start of section to remove ---#
# TODO: remove this before merging
# use librmm and rmm from https://github.com/rapidsai/rmm/pull/1644
RAPIDS_REPOSITORY=rmm \
RAPIDS_BUILD_TYPE=pull-request \
RAPIDS_REF_NAME=1644 \
RAPIDS_SHA=0701559 \
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" \
    rapids-download-wheels-from-s3 cpp /tmp/local-rmm-dep

RAPIDS_REPOSITORY=rmm \
RAPIDS_BUILD_TYPE=pull-request \
RAPIDS_REF_NAME=1644 \
RAPIDS_SHA=0701559 \
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" \
    rapids-download-wheels-from-s3 python /tmp/local-rmm-dep

echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/local-rmm-dep/librmm_*.whl)" >> /tmp/constraints.txt
echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/local-rmm-dep/rmm_*.whl)" >> /tmp/constraints.txt

export PIP_CONSTRAINT=/tmp/constraints.txt
# --- end of section to remove ---#


python -m pip install "$(echo ./local-cudf-dep/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)"
python -m pip install --find-links $(pwd)/local-cudf-dep "$(echo ./local-cudf-dep/cudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test,pandas-tests]"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
  -n 5 \
  --tb=no \
  -m "not slow" \
  --max-worker-restart=3 \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas.xml" \
  --dist worksteal \
  --report-log=${PANDAS_TESTS_BRANCH}.json 2>&1

SUMMARY_FILE_NAME=${PANDAS_TESTS_BRANCH}-${RAPIDS_FULL_VERSION}-results.json
# summarize the results and save them to artifacts:
python python/cudf/cudf/pandas/scripts/summarize-test-results.py --output json pandas-testing/${PANDAS_TESTS_BRANCH}.json > pandas-testing/${SUMMARY_FILE_NAME}
RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
mv pandas-testing/${SUMMARY_FILE_NAME} ${RAPIDS_ARTIFACTS_DIR}/
rapids-upload-to-s3 ${RAPIDS_ARTIFACTS_DIR}/${SUMMARY_FILE_NAME} "${RAPIDS_ARTIFACTS_DIR}"
