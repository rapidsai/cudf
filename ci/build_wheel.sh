#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
echo "${version}" > VERSION

if rapids-is-release-build; then
    export RAPIDS_ONLY_RELEASE_DEPS=1
fi

# Need to manually patch the cuda-python version for CUDA 12.
ctk_major=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f1)
if [[ ${ctk_major} == "12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9a]*/cuda-python>=12.0,<13.0a0/g" ${package_dir}/pyproject.toml
fi

cd "${package_dir}"

# For testing, will be removed when the package is released
git clone https://github.com/vyasr/rapids_builder.git
pushd rapids_builder
python -m pip wheel . --no-deps
popd

PIP_FIND_LINKS="file://${PWD}/rapids_builder/" python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check
