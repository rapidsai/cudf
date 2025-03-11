#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1;

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_cudf

# rapids-logger "Check GPU usage"
# nvidia-smi
rapids-print-env
# shellcheck disable=SC2034
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "import cudf"
python -c "import cudf"
