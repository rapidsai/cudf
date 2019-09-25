#!/usr/bin/env bash
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

logger "Building dask_cudf"
conda build conda/recipes/dask-cudf --python=$PYTHON
