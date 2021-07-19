#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

# Run isort-cudf and get results/return code
ISORT_CUDF=`isort python/cudf --check-only --settings-path=python/cudf/setup.cfg 2>&1`
ISORT_CUDF_RETVAL=$?

# Run isort-cudf-kafka and get results/return code
ISORT_CUDF_KAFKA=`isort python/cudf_kafka --check-only --settings-path=python/cudf_kafka/setup.cfg 2>&1`
ISORT_CUDF_KAFKA_RETVAL=$?

# Run isort-custreamz and get results/return code
ISORT_CUSTREAMZ=`isort python/custreamz --check-only --settings-path=python/custreamz/setup.cfg 2>&1`
ISORT_CUSTREAMZ_RETVAL=$?

# Run isort-dask-cudf and get results/return code
ISORT_DASK_CUDF=`isort python/dask_cudf --check-only --settings-path=python/dask_cudf/setup.cfg 2>&1`
ISORT_DASK_CUDF_RETVAL=$?

# Run black and get results/return code
BLACK=`black --check python 2>&1`
BLACK_RETVAL=$?

# Run flake8 and get results/return code
FLAKE=`flake8 --config=python/.flake8 python 2>&1`
FLAKE_RETVAL=$?

# Run flake8-cython and get results/return code
FLAKE_CYTHON=`flake8 --config=python/.flake8.cython 2>&1`
FLAKE_CYTHON_RETVAL=$?

# Run mypy and get results/return code
MYPY_CUDF=`mypy --config=python/cudf/setup.cfg python/cudf/cudf 2>&1`
MYPY_CUDF_RETVAL=$?

# Run pydocstyle and get results/return code
PYDOCSTYLE=`pydocstyle --config=python/.flake8 python 2>&1`
PYDOCSTYLE_RETVAL=$?

# Run clang-format and check for a consistent code format
CLANG_FORMAT=`python cpp/scripts/run-clang-format.py 2>&1`
CLANG_FORMAT_RETVAL=$?

# Output results if failure otherwise show pass
if [ "$ISORT_CUDF_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: isort-cudf style check; begin output\n\n"
  echo -e "$ISORT_CUDF"
  echo -e "\n\n>>>> FAILED: isort-cudf style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: isort-cudf style check\n\n"
fi

if [ "$ISORT_CUDF_KAFKA_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: isort-cudf-kafka style check; begin output\n\n"
  echo -e "$ISORT_CUDF_KAFKA"
  echo -e "\n\n>>>> FAILED: isort-cudf-kafka style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: isort-cudf-kafka style check\n\n"
fi

if [ "$ISORT_CUSTREAMZ_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: isort-custreamz style check; begin output\n\n"
  echo -e "$ISORT_CUSTREAMZ"
  echo -e "\n\n>>>> FAILED: isort-custreamz style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: isort-custreamz style check\n\n"
fi

if [ "$ISORT_DASK_CUDF_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: isort-dask-cudf style check; begin output\n\n"
  echo -e "$ISORT_DASK_CUDF"
  echo -e "\n\n>>>> FAILED: isort-dask-cudf style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: isort-dask-cudf style check\n\n"
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

if [ "$MYPY_CUDF_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: mypy style check; begin output\n\n"
  echo -e "$MYPY_CUDF"
  echo -e "\n\n>>>> FAILED: mypy style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: mypy style check\n\n"
fi

if [ "$PYDOCSTYLE_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: pydocstyle style check; begin output\n\n"
  echo -e "$PYDOCSTYLE"
  echo -e "\n\n>>>> FAILED: pydocstyle style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: pydocstyle style check\n\n"
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

RETVALS=(
  $ISORT_CUDF_RETVAL $ISORT_CUDF_KAFKA_RETVAL $ISORT_CUSTREAMZ_RETVAL $ISORT_DASK_CUDF_RETVAL
  $BLACK_RETVAL $FLAKE_RETVAL $FLAKE_CYTHON_RETVAL $PYDOCSTYLE_RETVAL $CLANG_FORMAT_RETVAL
  $HEADER_META_RETVAL $MYPY_CUDF_RETVAL
)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
