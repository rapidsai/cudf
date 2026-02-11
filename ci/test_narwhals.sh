#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

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
pushd narwhals
rapids-pip-retry install -U -e .

rapids-logger "Check narwhals versions"
python -c "import narwhals; print(narwhals.show_versions())"

# test_to_numpy[cudf]: Passes as of https://github.com/rapidsai/cudf/pull/19923
# test_fill_null_strategies_with_limit_as_none[cudf]: Narwhals passes inplace=None instead of a bool
# test_fill_null_series_limit_as_none[cudf]: Narwhals passes inplace=None instead of a bool
# test_cast_decimal_to_native[cudf-*]: Decimal casting issues
# test_categorical[cudf], test_categorical_as_str[cudf-*], test_cast[cudf], test_cast_series[cudf], test_cast_to_enum_vmain[cudf]: Categorical type handling issues - TypeError: unhashable type
# test_iter_columns[cudf]: Column iteration not implemented - NotImplementedError
# test_getitem_boolean_columns[cudf]: Boolean column indexing not implemented
# test_check_row_order[cudf-False], test_self_equal[cudf]: Testing/comparison issues with categorical columns
# test_schema_from_pandas_like[cudf]: Schema differences with categorical columns
# test_to_datetime[cudf], test_to_datetime_series[cudf], test_to_datetime_infer_fmt[cudf-*], test_to_datetime_series_infer_fmt[cudf-*]: String to datetime conversion issues
# test_with_columns_dtypes_single_row[cudf]: Type handling issue with single row
# test_first_last_different_orders[cudf]: Multiple order_by in groupby not supported by narwhals pandas-like backend
# test_series_rfloordiv_by_zero[cudf-0], test_expr_rfloordiv_by_zero[cudf-0]: xpass(strict) - now passing but tests expect failure
TESTS_THAT_NEED_NARWHALS_FIX_FOR_CUDF=" \
test_to_numpy[cudf] or \
test_fill_null_strategies_with_limit_as_none[cudf] or \
test_fill_null_series_limit_as_none[cudf] or \
(test_cast_decimal_to_native and cudf) or \
(test_categorical_as_str and cudf) or \
test_categorical[cudf] or \
test_cast[cudf] or \
test_cast_series[cudf] or \
test_cast_to_enum_vmain[cudf] or \
test_iter_columns[cudf] or \
test_getitem_boolean_columns[cudf] or \
test_check_row_order[cudf-False] or \
test_self_equal[cudf] or \
test_schema_from_pandas_like[cudf] or \
(test_to_datetime_infer_fmt and cudf) or \
(test_to_datetime and cudf) or \
(test_to_datetime_series and cudf) or \
(test_to_datetime_series_infer_fmt and cudf) or \
test_with_columns_dtypes_single_row[cudf] or \
test_first_last_different_orders[cudf] or \
test_series_rfloordiv_by_zero[cudf-0] or \
test_expr_rfloordiv_by_zero[cudf-0] \
"

rapids-logger "Run narwhals tests for cuDF"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    timeout 15m \
    python -m pytest \
    --cache-clear \
    -p xdist \
    -p env \
    -p no:pytest_benchmark \
    -p cudf.testing.narwhals_test_plugin \
    -k "not ( \
        ${TESTS_THAT_NEED_NARWHALS_FIX_FOR_CUDF} \
    )" \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=cudf

# test_datetime[polars[lazy]]: Fixed in the next narwhals release >2.0.1
# test_nan[polars[lazy]]: Passes as of https://github.com/rapidsai/cudf/pull/19742
# test_to_datetime_tz_aware[polars[lazy]-None]: Fixed in the Narwhals version that supports polars 1.33.1
# test_nested_structures[polars[lazy]-value0|1|3|4|6|7]: List/tuple literals without nested lists
# test_series_from_iterable[pandas-*]: Pandas-specific test that shouldn't run with polars constructor
TESTS_THAT_NEED_NARWHALS_FIX_FOR_CUDF_POLARS=" \
test_datetime[polars[lazy]] or \
test_nan[polars[lazy]] or \
test_to_datetime_tz_aware[polars[lazy]-None] or \
test_truncate[polars[lazy]-1ns-expected0] or \
test_truncate_multiples[polars[lazy]-2ns-expected0] or \
((test_nested_structures and polars and lazy) and (value0 or value1 or value3 or value4 or value6 or value7)) or \
(test_series_from_iterable and pandas) \
"

rapids-logger "Run narwhals tests for cuDF Polars"
CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE=805306368 \
CUDF_POLARS__EXECUTOR__FALLBACK_MODE=silent \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NARWHALS_POLARS_GPU=1 \
    timeout 15m \
    python -m pytest \
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
# test_all_ignore_nulls, test_allh_kleene, and test_anyh_kleene: https://github.com/rapidsai/cudf/issues/19417
# test_offset_by_date_pandas: https://github.com/rapidsai/cudf/issues/19418
# test_select_boolean_cols and test_select_boolean_cols_multi_group_by: https://github.com/rapidsai/cudf/issues/19421
# test_to_datetime_pd_preserves_pyarrow_backend_dtype: https://github.com/rapidsai/cudf/issues/19422
TESTS_THAT_NEED_CUDF_FIX=" \
test_is_finite_expr or \
test_is_finite_series or \
test_maybe_convert_dtypes_pandas or \
test_log_dtype_pandas or \
test_len_over_2369 or \
test_all_ignore_nulls or \
test_allh_kleene or \
test_anyh_kleene or \
test_offset_by_date_pandas or \
test_select_boolean_cols or \
test_select_boolean_cols_multi_group_by or \
test_to_datetime_pd_preserves_pyarrow_backend_dtype \
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
# test_get_dtype_backend: We now preserve arrow extension dtypes
# (e.g. bool[pyarrow], duration[ns][pyarrow]).
# test_explode_multiple_cols[pandas-l1-more_columns0-expected0] matches pandas now so needs a skip in the test
# test_series_from_iterable[*-cudf.pandas.fast_slow_proxy._FunctionProxy-*]: FunctionProxy issues with series_from_iterable
# test_group_by_agg_first/last[pandas-*]: first/last aggregation with order_by not working correctly in cudf.pandas
# test_first_expr_in_group_by[pandas]: Related to first/last aggregation issues
# test_any_value_group_by[pandas-False]: Non-deterministic any_value in groupby
# test_over_when_then_aggregation_partition_by[pandas-*]: Window function issues with when/then
# test_pandas_pyarrow_dtypes, test_nested_dtypes_dask, test_schema_from_pandas_like[pandas]: PyArrow dtype handling differences
# test_top_k[pandas]: top_k ordering mismatch
# test_get_series[pandas-0-expected0]: List.get operation issue
# test_sqrt_dtype_pandas[*]: sqrt with nulls returns 0.0 instead of NA in cudf.pandas
# test_check_row_order_nested_only[pandas]: Nested dtype row ordering check not raising NotImplementedError as expected
# test_cast_string: String casting issues with cudf.pandas
# test_contains_case_insensitive[pandas], test_contains_series_case_insensitive[pandas]: String contains with case_insensitive returns False instead of None for nulls
TESTS_THAT_NEED_NARWHALS_FIX_FOR_CUDF_PANDAS=" \
test_dtypes or \
test_explode_multiple_cols or \
(test_get_dtype_backend and pyarrow and (pandas or modin)) or \
test_series_from_iterable or \
(test_group_by_agg_first and pandas) or \
(test_group_by_agg_last and pandas) or \
test_first_expr_in_group_by[pandas] or \
test_any_value_group_by[pandas-False] or \
(test_over_when_then_aggregation_partition_by and pandas) or \
test_pandas_pyarrow_dtypes or \
test_nested_dtypes_dask or \
test_schema_from_pandas_like[pandas] or \
test_top_k[pandas] or \
test_get_series[pandas-0-expected0] or \
test_sqrt_dtype_pandas or \
test_check_row_order_nested_only[pandas] or \
test_cast_string or \
(test_contains_case_insensitive and pandas) or \
(test_contains_series_case_insensitive and pandas) \
"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NARWHALS_DEFAULT_CONSTRUCTORS=pandas \
    timeout 15m \
    python -m pytest \
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

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
