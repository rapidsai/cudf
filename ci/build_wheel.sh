#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override="$(rapids-pip-wheel-version ${RAPIDS_DATE_STRING})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

sed -i "s/^version = .*/version = \"${version_override}\"/g" ${pyproject_file}
sed -i "s/^name = .*/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}

# For nightlies we want to ensure that we're pulling in alphas as well. The
# easiest way to do so is to augment the spec with a constraint containing a
# min alpha version that doesn't affect the version bounds but does allow usage
# of alpha versions for that dependency without --pre
alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi

if [[ ${package_name} == "dask_cudf" ]]; then
    sed -r -i "s/cudf==(.*)\"/cudf${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
else
    sed -r -i "s/rmm(.*)\"/rmm${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}
    # ptxcompiler and cubinlinker aren't version constrained
    sed -r -i "s/ptxcompiler\"/ptxcompiler${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
    sed -r -i "s/cubinlinker\"/cubinlinker${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
fi

if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9a]*/cuda-python>=12.0,<13.0a0/g" ${pyproject_file}
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
    sed -i "/ptxcompiler/d" ${pyproject_file}
    sed -i "/cubinlinker/d" ${pyproject_file}
fi

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check
