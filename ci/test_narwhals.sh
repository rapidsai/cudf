#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_narwhals

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest narwhals"
git clone https://github.com/narwhals-dev/narwhals --depth=1 -b stable
pushd narwhals || exit 1
rapids-pip-retry install -U -e . pytest-env hypothesis

rapids-logger "Check narwhals versions"
python -c "import narwhals; print(narwhals.show_versions())"

rapids-logger "Run narwhals tests for cuDF"
python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-narwhals.xml" \
    -p cudf.testing.narwhals_test_plugin \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=cudf

# Narwhals needs to allow for tests to be run for any backend without needing the other
# backends installed. See https://github.com/rapidsai/cudf/pull/18297#issuecomment-2730310885
TEST_THAT_NEED_NARWHALS_FIX=" \
test_eager_only_sqlframe or \
test_series_only_sqlframe \
"

# Temporarily skipping these tests in 25.04 and will unskip them in 25.06, which will support Polars 1.26.
# Will also prioritize https://github.com/rapidsai/cudf/issues/18191, which will switch us to testing against
# Narwhals tags instead of the "stable" branch for 25.06. That change will allow us to require all
# Narwhals tests to pass consistently for supported versions.
TEMPORARILY_SKIP=" \
test_rolling_std_expr_lazy_ungrouped or \
test_rolling_var_expr_lazy_ungrouped \
"

rapids-logger "Run narwhals tests for cuDF Polars"
NARWHALS_POLARS_GPU=1 python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-narwhals.xml" \
    -k "not ( \
        ${TEST_THAT_NEED_NARWHALS_FIX} or \
        ${TEMPORARILY_SKIP} \
    )" \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=polars[lazy]

rapids-logger "Run narwhals tests for cuDF Pandas"

# test_is_finite_expr & test_is_finite_series: https://github.com/rapidsai/cudf/issues/18257
# test_maybe_convert_dtypes_pandas: https://github.com/rapidsai/cudf/issues/14149
TESTS_THAT_NEED_CUDF_FIX=" \
test_is_finite_expr or \
test_is_finite_series or \
test_maybe_convert_dtypes_pandas \
"

# test_array_dunder_with_copy: https://github.com/rapidsai/cudf/issues/18248#issuecomment-2719234741
# test_to_arrow & test_to_arrow_with_nulls: https://github.com/rapidsai/cudf/issues/18248#issuecomment-2719254791
# test_pandas_object_series: https://github.com/rapidsai/cudf/issues/18248#issuecomment-2719180627
TESTS_TO_ALWAYS_SKIP=" \
test_array_dunder_with_copy or \
test_to_arrow or \
test_to_arrow_with_nulls or \
test_pandas_object_series \
"

NARWHALS_DEFAULT_CONSTRUCTORS=pandas python -m pytest \
    -p cudf.pandas \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas-narwhals.xml" \
    -k "not ( \
        ${TESTS_THAT_NEED_CUDF_FIX} or \
        ${TESTS_TO_ALWAYS_SKIP} or \
        ${TEST_THAT_NEED_NARWHALS_FIX} \
    )" \
    --numprocesses=8 \
    --dist=worksteal

popd || exit 1

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
