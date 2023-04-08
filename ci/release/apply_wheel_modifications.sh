#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

# pyproject.toml versions
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/cudf/pyproject.toml
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/dask_cudf/pyproject.toml
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/cudf_kafka/pyproject.toml
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/custreamz/pyproject.toml

# cudf pyproject.toml cuda suffixes
sed -i "s/^name = \"cudf\"/name = \"cudf${CUDA_SUFFIX}\"/g" python/cudf/pyproject.toml
sed -i "s/rmm/rmm${CUDA_SUFFIX}/g" python/cudf/pyproject.toml
sed -i "s/ptxcompiler/ptxcompiler${CUDA_SUFFIX}/g" python/cudf/pyproject.toml
sed -i "s/cubinlinker/cubinlinker${CUDA_SUFFIX}/g" python/cudf/pyproject.toml

# dask_cudf pyproject.toml cuda suffixes
sed -i "s/^name = \"dask_cudf\"/name = \"dask_cudf${CUDA_SUFFIX}\"/g" python/dask_cudf/pyproject.toml
# Need to provide the == to avoid modifying the URL
sed -i "s/\"cudf==/\"cudf${CUDA_SUFFIX}==/g" python/dask_cudf/pyproject.toml
