#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh
# Copyright (c) 2020-2022, NVIDIA CORPORATION.

set -e

# Setup 'gpuci_retry' for upload retries (results in 4 total attempts)
export GPUCI_RETRY_MAX=3
export GPUCI_RETRY_SLEEP=30

# Set default label options if they are not defined elsewhere
export LABEL_OPTION=${LABEL_OPTION:-"--label main"}

# Skip uploads unless BUILD_MODE == "branch"
if [ "${BUILD_MODE}" != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

# Skip uploads if there is no upload key
if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Starting conda uploads"
if [[ "$BUILD_LIBCUDF" == "1" && "$UPLOAD_LIBCUDF" == "1" ]]; then
  export LIBCUDF_FILES=$(conda build --no-build-id --croot "${CONDA_BLD_DIR}" conda/recipes/libcudf --output)
  LIBCUDF_FILES=$(echo "$LIBCUDF_FILES" | sed 's/.*libcudf-example.*//') # skip libcudf-example pkg upload
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing --no-progress $LIBCUDF_FILES
fi

if [[ "$BUILD_CUDF" == "1" && "$UPLOAD_CUDF" == "1" ]]; then
  CUDF_FILES=$(conda build --croot "${CONDA_BLD_DIR}" conda/recipes/cudf --python=$PYTHON --output)
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing --no-progress ${CUDF_FILES}
fi
