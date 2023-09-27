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


if [[ "${PANDAS_TESTS_BRANCH}" == "main" ]]; then
    git checkout main
fi

pip install cudf_cu${RAPIDS_CUDA_VERSION%%.*}
cd python/xdf/
pip install .[test]

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
