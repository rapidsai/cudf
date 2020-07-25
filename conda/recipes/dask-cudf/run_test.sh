#!/bin/bash

set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Install the master version of dask and distributed
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps

pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

python -c "import dask_cudf"