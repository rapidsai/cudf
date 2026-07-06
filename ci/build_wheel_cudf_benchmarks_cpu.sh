#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_dir="python/cudf_benchmarks_cpu"

./ci/build_wheel.sh cudf-benchmarks-cpu ${package_dir}
cp "${package_dir}/dist"/* "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

# cudf-benchmarks-cpu is an unsuffixed, CUDA-agnostic pure wheel (built with
# disable-cuda), so its artifact name carries no CUDA suffix.
RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python cudf-benchmarks-cpu cudf --pure --arch any)"
export RAPIDS_PACKAGE_NAME
