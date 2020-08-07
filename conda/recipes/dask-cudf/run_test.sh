#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.

set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Install the master version of dask and distributed
logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps

logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

logger "python -c 'import dask_cudf'"
python -c "import dask_cudf"
