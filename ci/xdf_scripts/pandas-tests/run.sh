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

PANDAS_TESTS_BRANCH=${1}
XDF_MODE=${2:-"--transparent"}

rapids-logger "Running Pandas tests using $PANDAS_TESTS_BRANCH branch"
rapids-logger "Using ${XDF_MODE} for import overrrides"
rapids-logger "PR number: $RAPIDS_REF_NAME"


COMMIT=$(git rev-parse HEAD)
WHEEL_NAME_PREFIX="cudf_"
if [[ "${PANDAS_TESTS_BRANCH}" == "main" ]]; then
    COMMIT=$(git merge-base HEAD origin/branch-23.10-xdf)
    WHEEL_NAME_PREFIX="cudf_${PANDAS_TESTS_BRANCH}_"
fi

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${WHEEL_NAME_PREFIX}${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
python -m pip install ./local-cudf-dep/cudf*.whl[test,pandas-tests]

git checkout $COMMIT

bash scripts/run-pandas-tests.sh ${XDF_MODE} \
  -n 10 \
  --tb=line \
  --skip-slow \
  --max-worker-restart=3 \
  --import-mode=importlib \
  --report-log=${PANDAS_TESTS_BRANCH}.json 2>&1

# summarize the results and save them to artifacts:
python scripts/summarize-test-results.py --output json pandas-testing/${PANDAS_TESTS_BRANCH}.json > pandas-testing/${PANDAS_TESTS_BRANCH}-results.json
RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
mv pandas-testing/${PANDAS_TESTS_BRANCH}-results.json ${RAPIDS_ARTIFACTS_DIR}/
