#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

# Setup 'gpuci_retry' for upload retries (results in 4 total attempts)
export GPUCI_RETRY_MAX=3
export GPUCI_RETRY_SLEEP=30

# Set default label options if they are not defined elsewhere
export LABEL_OPTION=${LABEL_OPTION:-"--label main"}

# Skip uploads unless BUILD_MODE == "branch"
if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

# Skip uploads if there is no upload key
if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi

################################################################################
# SETUP - Get conda file output locations
################################################################################

gpuci_logger "Get conda file output locations"
export LIBCUDF_FILE=`conda build conda/recipes/libcudf --output`
export LIBCUDF_KAFKA_FILE=`conda build conda/recipes/libcudf_kafka --output`
export CUDF_FILE=`conda build conda/recipes/cudf --python=$PYTHON --output`
export DASK_CUDF_FILE=`conda build conda/recipes/dask-cudf --python=$PYTHON --output`
export CUDF_KAFKA_FILE=`conda build conda/recipes/cudf_kafka --python=$PYTHON --output`
export CUSTREAMZ_FILE=`conda build conda/recipes/custreamz --python=$PYTHON --output`

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Starting conda uploads"
if [ "$UPLOAD_LIBCUDF" == "1" ]; then
  test -e ${LIBCUDF_FILE}
  echo "Upload libcudf"
  echo ${LIBCUDF_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${LIBCUDF_FILE}
fi

if [ "$UPLOAD_CUDF" == "1" ]; then
  test -e ${CUDF_FILE}
  echo "Upload cudf"
  echo ${CUDF_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUDF_FILE}

  test -e ${DASK_CUDF_FILE}
  echo "Upload dask-cudf"
  echo ${DASK_CUDF_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${DASK_CUDF_FILE}

  test -e ${CUSTREAMZ_FILE}
  echo "Upload custreamz"
  echo ${CUSTREAMZ_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUSTREAMZ_FILE}
fi

if [ "$UPLOAD_LIBCUDF_KAFKA" == "1" ]; then
  test -e ${LIBCUDF_KAFKA_FILE}
  echo "Upload libcudf_kafka"
  echo ${LIBCUDF_KAFKA_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${LIBCUDF_KAFKA_FILE}
fi

if [ "$UPLOAD_CUDF_KAFKA" == "1" ]; then
  test -e ${CUDF_KAFKA_FILE}
  echo "Upload cudf_kafka"
  echo ${CUDF_KAFKA_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUDF_KAFKA_FILE}
fi
