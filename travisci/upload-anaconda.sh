#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

SOURCE_BRANCH=master

export EXTRALABEL="dev_${CUDA:0:3}"
echo "EXTRALABEL=${EXTRALABEL}"

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
if [ ${CUDA:0:3} == '9.0' ]; then
  anaconda -t ${MY_UPLOAD_KEY} upload -u gpuopenanalytics -l dev --force ${UPLOADFILE}
else
  anaconda -t ${MY_UPLOAD_KEY} upload -u gpuopenanalytics -l ${EXTRALABEL} --force ${UPLOADFILE}
fi

