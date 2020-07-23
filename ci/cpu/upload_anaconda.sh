#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

export LIBCUDF_FILE=`conda build conda/recipes/libcudf --output`
export CUDF_FILE=`conda build conda/recipes/cudf --python=$PYTHON --output`
export DASK_CUDF_FILE=`conda build conda/recipes/dask-cudf --python=$PYTHON --output`
export LIBCUDF_KAFKA_FILE=`conda build conda/recipes/libcudf_kafka --output`
export CUDF_KAFKA_FILE=`conda build conda/recipes/cudf_kafka --python=$PYTHON --output`
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

  test -e ${LIBCUDF_FILE}
  echo "Upload libcudf"
  echo ${LIBCUDF_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${LIBCUDF_FILE}
fi

if [ "$UPLOAD_CUDF" == "1" ]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${CUDF_FILE}
  echo "Upload cudf"
  echo ${CUDF_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUDF_FILE}

  test -e ${DASK_CUDF_FILE}
  echo "Upload dask-cudf"
  echo ${DASK_CUDF_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${DASK_CUDF_FILE}

  test -e ${CUSTREAMZ_FILE}
  echo "Upload custreamz"
  echo ${CUSTREAMZ_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUSTREAMZ_FILE}
fi

if [ "$UPLOAD_LIBCUDF_KAFKA" == "1" ]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${LIBCUDF_KAFKA_FILE}
  echo "Upload libcudf_kafka"
  echo ${LIBCUDF_KAFKA_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${LIBCUDF_KAFKA_FILE}
fi

if [ "$UPLOAD_CUDF_KAFKA" == "1" ]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${CUDF_KAFKA_FILE}
  echo "Upload cudf_kafka"
  echo ${CUDF_KAFKA_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUDF_KAFKA_FILE}
fi
