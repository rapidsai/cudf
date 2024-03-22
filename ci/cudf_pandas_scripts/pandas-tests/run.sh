#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PANDAS_TESTS_BRANCH=${1}

rapids-logger "Running Pandas tests using $PANDAS_TESTS_BRANCH branch"
rapids-logger "PR number: ${RAPIDS_REF_NAME:-"unknown"}"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
python -m pip install $(ls ./local-cudf-dep/cudf*.whl)[test,pandas-tests]

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
  -n 10 \
  --tb=no \
  -m "not slow" \
  --max-worker-restart=3 \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas.xml" \
  --dist worksteal \
  --report-log=${PANDAS_TESTS_BRANCH}.json 2>&1

# summarize the results and save them to artifacts:
python python/cudf/cudf/pandas/scripts/summarize-test-results.py --output json pandas-testing/${PANDAS_TESTS_BRANCH}.json > pandas-testing/${PANDAS_TESTS_BRANCH}-results.json
RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
mv pandas-testing/${PANDAS_TESTS_BRANCH}-results.json ${RAPIDS_ARTIFACTS_DIR}/
rapids-upload-to-s3 ${RAPIDS_ARTIFACTS_DIR}/${PANDAS_TESTS_BRANCH}-results.json "${RAPIDS_ARTIFACTS_DIR}"
