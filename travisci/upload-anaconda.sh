#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh
export UPLOADFILE=`conda build conda-recipes/pygdf -c defaults -c conda-forge -c numba -c gpuopenanalytics/label/dev --python $PYTHON --output`
echo ${UPLOADFILE}
set -e

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
anaconda -t ${MY_UPLOAD_KEY} upload -u gpuopenanalytics -l dev --force ${UPLOADFILE}
