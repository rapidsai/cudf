#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_narwhals

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest narwhals"
git clone https://github.com/narwhals-dev/narwhals --depth=1
pushd narwhals || exit 1
rapids-pip-retry install -U -e ".[dev]"

rapids-logger "Check narwhals versions"
python -c "import narwhals; print(narwhals.show_versions())"

rapids-logger "Run narwhals tests for cuDF"
python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-narwhals.xml" \
    -p cudf.testing.narwhals_test_plugin \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=cudf

rapids-logger "Run narwhals tests for cuDF Polars"
NARWHALS_POLARS_GPU=1 python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-narwhals.xml" \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=polars[lazy]

popd || exit 1

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
