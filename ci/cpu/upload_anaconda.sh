#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

export LIBCUDF_FILE=`conda build conda/recipes/libcudf --output`
export LIBCUDF_CFFI_FILE=`conda build conda/recipes/libcudf_cffi --python=$PYTHON --output`
export CUDF_FILE=`conda build conda/recipes/cudf --python=$PYTHON --output`

SOURCE_BRANCH=master
CUDA_REL=${CUDA:0:3}
if [ "${CUDA:0:2}" == '10' ]; then
  # CUDA 10 release
  CUDA_REL=${CUDA:0:4}
fi

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
  LABEL_OPTION="--label main --label cuda${CUDA_REL}"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${LIBCUDF_FILE}
  echo "Upload libcudf"
  echo ${LIBCUDF_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${LIBCUDF_FILE}
fi

if [ "$UPLOAD_CUDF" == "1" ]; then
  LABEL_OPTION="--label main --label cuda9.2 --label cuda10.0"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${LIBCUDF_CFFI_FILE}
  echo "Upload libcudf_cffi"
  echo ${LIBCUDF_CFFI_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${LIBCUDF_CFFI_FILE}

  test -e ${CUDF_FILE}
  echo "Upload cudf"
  echo ${CUDF_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${CUDF_FILE}
fi