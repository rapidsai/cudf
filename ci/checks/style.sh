#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#####################
# cuDF Style Tester #
#####################

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH
LC_ALL=C.UTF-8
LANG=C.UTF-8

# Activate common conda env
source activate gdf

# Run isort and get results/return code
ISORT=`isort --check-only python/**/*.py`
ISORT_RETVAL=$?

# Run black and get results/return code
BLACK=`black --check python`
BLACK_RETVAL=$?

# Run flake8 and get results/return code
FLAKE=`flake8 --config=python/.flake8 python`
FLAKE_RETVAL=$?

# Run flake8-cython and get results/return code
FLAKE_CYTHON=`flake8 --config=python/.flake8.cython`
FLAKE_CYTHON_RETVAL=$?

# Run clang-format and check for a consistent code format
CLANG_FORMAT=`python cpp/scripts/run-clang-format.py 2>&1`
CLANG_FORMAT_RETVAL=$?

# Output results if failure otherwise show pass
if [ "$ISORT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: isort style check; begin output\n\n"
  echo -e "$ISORT"
  echo -e "\n\n>>>> FAILED: isort style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: isort style check\n\n"
fi

if [ "$BLACK_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: black style check; begin output\n\n"
  echo -e "$BLACK"
  echo -e "\n\n>>>> FAILED: black style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: black style check\n\n"
fi

if [ "$FLAKE_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

if [ "$FLAKE_CYTHON_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: flake8-cython style check; begin output\n\n"
  echo -e "$FLAKE_CYTHON"
  echo -e "\n\n>>>> FAILED: flake8-cython style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8-cython style check\n\n"
fi

if [ "$CLANG_FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: clang format check; begin output\n\n"
  echo -e "$CLANG_FORMAT"
  echo -e "\n\n>>>> FAILED: clang format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: clang format check\n\n"
fi

# Run header meta.yml check and get results/return code
HEADER_META=`ci/checks/headers_test.sh`
HEADER_META_RETVAL=$?
echo -e "$HEADER_META"

RETVALS=($ISORT_RETVAL $BLACK_RETVAL $FLAKE_RETVAL $FLAKE_CYTHON_RETVAL $CLANG_FORMAT_RETVAL $HEADER_META_RETVAL)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
