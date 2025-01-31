#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_cudf

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest narwhals"
git clone https://github.com/narwhals-dev/narwhals --depth=1
pushd narwhals || exit
pip install -U -e ".[dev]"
python -c "import narwhals; print(narwhals.show_versions())"
python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-narwhals.xml" \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=cudf
popd || exit

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
