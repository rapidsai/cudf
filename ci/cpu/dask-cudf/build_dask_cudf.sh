#!/usr/bin/env bash
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

logger "Building dask_cudf"
conda build conda/recipes/dask-cudf --python=$PYTHON
