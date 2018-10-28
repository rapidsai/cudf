#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

if [ "$BUILD_CUDF" == "1" ]; then
  export UPLOADFILE=`conda build conda-recipes/cudf -c defaults -c conda-forge -c numba -c rapidsai/label/dev --python=$PYTHON --output`
  SOURCE_BRANCH=master

  test -e ${UPLOADFILE}

  # Pull requests or commits to other branches shouldn't upload
  if [ ${TRAVIS_PULL_REQUEST} != false -o ${TRAVIS_BRANCH} != ${SOURCE_BRANCH} ]; then
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