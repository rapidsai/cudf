#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# Download the cudf built in the previous step
# Set the manylinux version used for downloading the wheels so that we test the
# newer ABI wheels on the newer images that support their installation.
# Need to disable pipefail for the head not to fail, see
# https://stackoverflow.com/questions/19120263/why-exit-code-141-with-grep-q
set +o pipefail
glibc_minor_version=$(ldd --version | head -1 | grep -o "[0-9]\.[0-9]\+" | tail -1 | cut -d '.' -f2)
set -o pipefail
manylinux_version="2_17"
if [[ ${glibc_minor_version} -ge 28 ]]; then
    manylinux_version="2_28"
fi
manylinux="manylinux_${manylinux_version}"

RAPIDS_PY_WHEEL_NAME="cudf_${manylinux}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
python -m pip install --no-deps ./local-cudf-dep/cudf*.whl

# Always install latest dask for testing
python -m pip install git+https://github.com/dask/dask.git@main git+https://github.com/dask/distributed.git@main git+https://github.com/rapidsai/dask-cuda.git@branch-23.12

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/dask_cudf*.whl)[test]

# Run tests in dask_cudf/tests and dask_cudf/io/tests
python -m pytest -n 8 ./python/dask_cudf/dask_cudf/
