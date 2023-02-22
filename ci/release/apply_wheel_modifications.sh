#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

# __init__.py versions
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/cudf/cudf/__init__.py
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/dask_cudf/dask_cudf/__init__.py
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/cudf_kafka/cudf_kafka/__init__.py
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/custreamz/custreamz/__init__.py

# setup.py versions
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/cudf/setup.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/dask_cudf/setup.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/cudf_kafka/setup.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/custreamz/setup.py

# cudf setup.py cuda suffixes
sed -i "s/name=\"cudf\"/name=\"cudf${CUDA_SUFFIX}\"/g" python/cudf/setup.py
sed -i "s/rmm/rmm${CUDA_SUFFIX}/g" python/cudf/setup.py
sed -i "s/ptxcompiler/ptxcompiler${CUDA_SUFFIX}/g" python/cudf/setup.py
sed -i "s/cubinlinker/cubinlinker${CUDA_SUFFIX}/g" python/cudf/setup.py

# cudf pyproject.toml cuda suffixes
sed -i "s/rmm/rmm${CUDA_SUFFIX}/g" python/cudf/pyproject.toml

# dask_cudf setup.py cuda suffixes
sed -i "s/name=\"dask-cudf\"/name=\"dask-cudf${CUDA_SUFFIX}\"/g" python/dask_cudf/setup.py
# Need to provide the == to avoid modifying the URL
sed -i "s/\"cudf==/\"cudf${CUDA_SUFFIX}==/g" python/dask_cudf/setup.py
