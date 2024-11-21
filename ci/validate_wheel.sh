#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2

cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

pydistcheck \
    --inspect \
    "$(echo ${wheel_dir_relative_path}/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo ${wheel_dir_relative_path}/*.whl)"
