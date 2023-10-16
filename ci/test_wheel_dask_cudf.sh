#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# Download the cudf built in the previous step
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
python -m pip install --no-deps ./local-cudf-dep/cudf*.whl

# Always install latest dask for testing
python -m pip install git+https://github.com/dask/dask.git@2023.9.2 git+https://github.com/dask/distributed.git@2023.9.2 git+https://github.com/rapidsai/dask-cuda.git@branch-23.12

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/dask_cudf*.whl)[test]

python -m pytest -n 8 ./python/dask_cudf/dask_cudf/tests
