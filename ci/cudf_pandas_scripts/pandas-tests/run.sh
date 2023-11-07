#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

PANDAS_TESTS_BRANCH=${1}

rapids-logger "Running Pandas tests using $PANDAS_TESTS_BRANCH branch"
rapids-logger "PR number: $RAPIDS_REF_NAME"


COMMIT=$(git rev-parse HEAD)
WHEEL_NAME="cudf"
if [[ "${PANDAS_TESTS_BRANCH}" == "main" ]]; then
    COMMIT=$(git merge-base HEAD origin/branch-23.10-xdf)
    WHEEL_NAME="${WHEEL_NAME}_${PANDAS_TESTS_BRANCH}"
fi

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${WHEEL_NAME}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
python -m pip install $(ls ./local-cudf-dep/cudf*.whl)[test,pandas_tests]

git checkout $COMMIT

bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
  -n 10 \
  --tb=line \
  --skip-slow \
  --max-worker-restart=3 \
  --import-mode=importlib \
  --report-log=${PANDAS_TESTS_BRANCH}.json 2>&1

# summarize the results and save them to artifacts:
python python/cudf/cudf/pandas/scripts/summarize-test-results.py --output json pandas-testing/${PANDAS_TESTS_BRANCH}.json > pandas-testing/${PANDAS_TESTS_BRANCH}-results.json
RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
mv pandas-testing/${PANDAS_TESTS_BRANCH}-results.json ${RAPIDS_ARTIFACTS_DIR}/
