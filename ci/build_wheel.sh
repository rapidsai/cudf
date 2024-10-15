#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

source rapids-configure-sccache
source rapids-date-string

rapids-logger "Generating build requirements"
matrix_selectors="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${underscore_package_name}" \
  --matrix "${matrix_selectors}" \
| tee /tmp/requirements-build.txt

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_rapids_build_${underscore_package_name}" \
  --matrix "${matrix_selectors}" \
| tee -a /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
python -m pip install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

rapids-generate-version > ./VERSION

cd "${package_dir}"

python -m pip wheel \
    -w dist \
    -v \
    --no-build-isolation \
    --no-deps \
    --disable-pip-version-check \
    .
