#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PANDAS_TESTS_BRANCH=${1}

rapids-logger "Running Pandas tests using $PANDAS_TESTS_BRANCH branch"
rapids-logger "PR number: ${RAPIDS_REF_NAME:-"unknown"}"

# Set the manylinux version used for downloading the wheels so that we test the
# newer ABI wheels on the newer images that support their installation.
# Need to disable pipefail for the head not to fail, see
# https://stackoverflow.com/questions/19120263/why-exit-code-141-with-grep-q
set +o pipefail
glibc_minor_version=$(ldd --version | head -1 | grep -o "[0-9]\.[0-9]\+" | tail -1 | cut -d '.' -f2)
set -o pipefail
manylinux_version="2_17"
if [[ ${glibc_minor_version} -ge 28 ]]; then
    manylinux_version="2_28"
fi
manylinux="manylinux_${manylinux_version}"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_${manylinux}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
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
