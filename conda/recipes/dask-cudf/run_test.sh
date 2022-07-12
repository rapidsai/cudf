#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.

set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Importing cudf on arm64 CPU only nodes is currently not working due to a
# difference in reported gpu devices between arm64 and amd64
ARCH=$(arch)

if [ "${ARCH}" = "aarch64" ]; then
  logger "Skipping tests on arm64"
  exit 0
fi

# Install the latest version of dask and distributed
logger "pip install git+https://github.com/dask/distributed.git@main --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git@main" --upgrade --no-deps

logger "pip install git+https://github.com/dask/dask.git@main --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git@main" --upgrade --no-deps

logger "python -c 'import dask_cudf'"
python -c "import dask_cudf"
