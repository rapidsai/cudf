#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

if [ "$BUILD_CUDF" == "1" ]; then
  export UPLOADFILE=`conda build conda/recipes/cudf -c rapidsai -c nvidia -c numba -c conda-forge -c defaults --python=$PYTHON --output`
  SOURCE_BRANCH=master

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
  anaconda -t ${MY_UPLOAD_KEY} upload -u rapidsai -l main --force ${UPLOADFILE}
fi