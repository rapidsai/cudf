#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir=$1

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

cd "${package_dir}"

python -m pip wheel . -w dist -v --no-deps --disable-pip-version-check
