#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/xdf"

./ci/build_wheel.sh xdf ${package_dir} 1

RAPIDS_PY_WHEEL_NAME="xdf" rapids-upload-wheels-to-s3 ${package_dir}/dist
