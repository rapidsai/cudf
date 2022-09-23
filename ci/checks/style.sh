#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
#####################
# cuDF Style Tester #
#####################

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH
LC_ALL=C.UTF-8
LANG=C.UTF-8

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

FORMAT_FILE_URL=https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.12/cmake-format-rapids-cmake.json
export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
mkdir -p $(dirname ${RAPIDS_CMAKE_FORMAT_FILE})
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} ${FORMAT_FILE_URL}

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files
PRE_COMMIT_RETVAL=$?

# Check for copyright headers in the files modified currently
COPYRIGHT=`python ci/checks/copyright.py --git-modified-only 2>&1`
COPYRIGHT_RETVAL=$?

# Output results if failure otherwise show pass
if [ "$COPYRIGHT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check\n\n"
  echo -e "$COPYRIGHT"
fi

RETVALS=(
  $PRE_COMMIT_RETVAL $COPYRIGHT_RETVAL
)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
