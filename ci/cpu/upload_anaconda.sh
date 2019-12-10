#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

export LIBNVSTRINGS_FILE=`conda build conda/recipes/libnvstrings --output`
export NVSTRINGS_FILE=`conda build conda/recipes/nvstrings --python=$PYTHON --output`
export LIBCUDF_FILE=`conda build conda/recipes/libcudf --output`
export CUDF_FILE=`conda build conda/recipes/cudf --python=$PYTHON --output`
export DASK_CUDF_FILE=`conda build conda/recipes/dask-cudf --python=$PYTHON --output`
export CUSTREAMZ_FILE=`conda build conda/recipes/custreamz --python=$PYTHON --output`

SOURCE_BRANCH=master
CUDA_REL=${CUDA_VERSION%.*}

# Restrict uploads to master branch
if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
  echo "Skipping upload"
  return 0
fi

if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
fi

if [ "$UPLOAD_LIBCUDF" == "1" ]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${LIBNVSTRINGS_FILE}
  echo "Upload libNVStrings"
  echo ${LIBNVSTRINGS_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${LIBNVSTRINGS_FILE}

  test -e ${LIBCUDF_FILE}
  echo "Upload libcudf"
  echo ${LIBCUDF_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${LIBCUDF_FILE}
fi

if [ "$UPLOAD_CUDF" == "1" ]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${NVSTRINGS_FILE}
  echo "Upload nvstrings"
  echo ${NVSTRINGS_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${NVSTRINGS_FILE}

  test -e ${CUDF_FILE}
  echo "Upload cudf"
  echo ${CUDF_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${CUDF_FILE}

  test -e ${DASK_CUDF_FILE}
  echo "Upload dask-cudf"
  echo ${DASK_CUDF_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${DASK_CUDF_FILE}

  test -e ${CUSTREAMZ_FILE}
  echo "Upload custreamz"
  echo ${CUSTREAMZ_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${CUSTREAMZ_FILE}
fi
