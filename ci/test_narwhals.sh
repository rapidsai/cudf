#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

source rapids-init-pip

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_narwhals

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest narwhals"
NARWHALS_VERSION=$(python -c "import narwhals; print(narwhals.__version__)")
git clone https://github.com/narwhals-dev/narwhals.git --depth=1 -b "v${NARWHALS_VERSION}" narwhals
pushd narwhals || exit 1
rapids-pip-retry install -U -e .

rapids-logger "Check narwhals versions"
python -c "import narwhals; print(narwhals.show_versions())"

rapids-logger "Run narwhals tests for cuDF"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-narwhals.xml" \
    -p xdist \
    -p env \
    -p no:pytest_benchmark \
    -p cudf.testing.narwhals_test_plugin \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=cudf

# test_dtypes: With cudf.pandas loaded, to_pandas() preserves Arrow dtypes like list and struct, so pandas
# columns aren't object anymore. The test expects object, causing a mismatch.
# test_nan: Narwhals expect this test to fail, but as of polars 1.30 we raise a RuntimeError,
# not polars ComputeError. So the test is looking for the wrong error and fails.
TESTS_THAT_NEED_NARWHALS_FIX_FOR_CUDF_POLARS=" \
test_dtypes or \
test_nan \
"

rapids-logger "Run narwhals tests for cuDF Polars"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NARWHALS_POLARS_GPU=1 python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-narwhals.xml" \
    -p xdist \
    -p env \
    -p no:pytest_benchmark \
    -k "not ( \
        ${TESTS_THAT_NEED_NARWHALS_FIX_FOR_CUDF_POLARS} \
    )" \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=polars[lazy]

rapids-logger "Run narwhals tests for cuDF Pandas"

# test_is_finite_expr & test_is_finite_series: https://github.com/rapidsai/cudf/issues/18257
# test_maybe_convert_dtypes_pandas: https://github.com/rapidsai/cudf/issues/14149
# test_log_dtype_pandas: cudf is promoting the type to float64
# test_len_over_2369: It fails during fallback. The error is 'DataFrame' object has no attribute 'to_frame'
TESTS_THAT_NEED_CUDF_FIX=" \
test_is_finite_expr or \
test_is_finite_series or \
test_maybe_convert_dtypes_pandas or \
test_log_dtype_pandas or \
test_len_over_2369 \
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

# test_dtypes: With cudf.pandas loaded, to_pandas() preserves Arrow dtypes like list and struct, so pandas
# columns aren't object anymore. The test expects object, causing a mismatch.
TESTS_THAT_NEED_NARWHALS_FIX_FOR_CUDF_PANDAS=" \
test_dtypes \
"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NARWHALS_DEFAULT_CONSTRUCTORS=pandas python -m pytest \
    -p cudf.pandas \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas-narwhals.xml" \
    -p xdist \
    -p env \
    -p no:pytest_benchmark \
    -k "not ( \
        ${TESTS_THAT_NEED_CUDF_FIX} or \
        ${TESTS_TO_ALWAYS_SKIP} or \
        ${TESTS_THAT_NEED_NARWHALS_FIX_FOR_CUDF_PANDAS} \
    )" \
    --numprocesses=8 \
    --dist=worksteal

popd || exit 1

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
