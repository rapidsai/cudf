#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

if [ "$BUILD_CUDF" == "1" ]; then
  if [ "$BUILD_ABI" == "1" ]; then
    export UPLOADFILE=`conda build conda/recipes/cudf -c rapidsai -c rapidsai-nightly -c nvidia -c numba -c conda-forge -c defaults --python=$PYTHON --output`
  else
    export UPLOADFILE=`conda build conda/recipes/cudf -c rapidsai/label/cf201901 -c rapidsai-nightly/label/cf201901 -c nvidia/label/cf201901 -c numba -c conda-forge/label/cf201901 -c defaults --python=$PYTHON --output`
  fi

  SOURCE_BRANCH=master

  # Have to label all CUDA versions due to the compatibility to work with any CUDA
  if [ "$LABEL_MAIN" == "1" -a "$BUILD_ABI" == "1" ]; then
    LABEL_OPTION="--label main --label cuda9.2 --label cuda10.0"
  elif [ "$LABEL_MAIN" == "0" -a "$BUILD_ABI" == "1" ]; then
    LABEL_OPTION="--label dev --label cuda9.2 --label cuda10.0"
  elif [ "$LABEL_MAIN" == "1" -a "$BUILD_ABI" == "0" ]; then
    LABEL_OPTION="--label cf201901 --label cf201901-cuda9.2 --label cf201901-cuda10.0"
  elif [ "$LABEL_MAIN" == "0" -a "$BUILD_ABI" == "0" ]; then
    LABEL_OPTION="--label cf201901-dev --label cf201901-cuda9.2 --label cf201901-cuda10.0"
  else
    echo "Unknown label configuration LABEL_MAIN='$LABEL_MAIN' BUILD_ABI='$BUILD_ABI'"
    exit 1
  fi
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${UPLOADFILE}

  # Restrict uploads to master branch
  if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
    echo "Skipping upload"
    return 0
  fi

  if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
  fi

  echo "Upload"
  echo ${UPLOADFILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${UPLOADFILE}
fi