# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running polars test suite setting GPU engine as default."""

from __future__ import annotations

from functools import partialmethod
from typing import TYPE_CHECKING

import pytest

import polars

if TYPE_CHECKING:
    from collections.abc import Mapping


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add plugin-specific options."""
    group = parser.getgroup(
        "cudf-polars", "Plugin to set GPU as default engine for polars tests"
    )
    group.addoption(
        "--cudf-polars-no-fallback",
        action="store_true",
        help="Turn off fallback to CPU when running tests (default use fallback)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Enable use of this module as a pytest plugin to enable GPU collection."""
    no_fallback = config.getoption("--cudf-polars-no-fallback")
    collect = polars.LazyFrame.collect
    engine = polars.GPUEngine(raise_on_fail=no_fallback)
    polars.LazyFrame.collect = partialmethod(collect, engine=engine)
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*GPU engine does not support streaming or background collection",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*Query execution with GPU not possible",
    )


EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/unit/io/test_csv.py::test_compressed_csv": "Need to determine if file is compressed",
    "tests/unit/io/test_csv.py::test_read_csv_only_loads_selected_columns": "Memory usage won't be correct due to GPU",
    "tests/unit/io/test_delta.py::test_scan_delta_version": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_scan_delta_relative": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_read_delta_version": "Need to expose hive partitioning",
    "tests/unit/io/test_lazy_count_star.py::test_count_compressed_csv_18057": "Need to determine if file is compressed",
    "tests/unit/io/test_lazy_csv.py::test_scan_csv_slice_offset_zero": "Integer overflow in sliced read",
    "tests/unit/io/test_lazy_parquet.py::test_dsl2ir_cached_metadata[False]": "cudf-polars doesn't use metadata read by rust preprocessing",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_is_in_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_different_schema[False]": "Needs cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-columns]": "Correctly raises but different error",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-row_groups]": "Correctly raises but different error",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-prefiltered]": "Correctly raises but different error",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-none]": "Correctly raises but different error",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_mismatch_panic_17067[False]": "Needs cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_ignores_dtype_mismatch_for_non_projected_columns_19249[False-False]": "Needs some variant of cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_ignores_dtype_mismatch_for_non_projected_columns_19249[True-False]": "Needs some variant of cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_slice_pushdown_non_zero_offset[False]": "Thrift data not handled correctly/slice pushdown wrong?",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_unaligned_schema_read[False]": "Incomplete handling of projected reads with mismatching schemas, cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_unaligned_schema_read_dtype_mismatch[False]": "Different exception raised, but correctly raises an exception",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_unaligned_schema_read_missing_cols_from_first[False]": "Different exception raised, but correctly raises an exception",
    "tests/unit/io/test_parquet.py::test_read_parquet_only_loads_selected_columns_15098": "Memory usage won't be correct due to GPU",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-False-none]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-False-none]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-False-prefiltered]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-False-prefiltered]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-False-row_groups]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-False-row_groups]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-False-columns]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-False-columns]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-True-none]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-True-none]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-True-prefiltered]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-True-prefiltered]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-True-row_groups]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-True-row_groups]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection0-True-columns]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_allow_missing_columns[projection1-True-columns]": "Mismatching column read cudf#16394",
    "tests/unit/io/test_scan.py::test_scan[single-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_limit[single-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_filter[single-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[single-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_limit_and_filter[single-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_and_limit[single-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_and_filter[single-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_limit_and_filter[single-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan[glob-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_limit[glob-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_filter[glob-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[glob-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_limit_and_filter[glob-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_and_limit[glob-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_and_filter[glob-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_limit_and_filter[glob-csv-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_limit[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_filter[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_limit_and_filter[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_and_limit[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_and_filter[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_limit_and_filter[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_filter_and_limit[glob-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_limit[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_filter[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_limit_and_filter[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_and_limit[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_and_filter[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_limit_and_filter[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_filter_and_limit[single-parquet-async]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_scan_include_file_name[False-scan_parquet-write_parquet]": "Need to add include_file_path to IR",
    "tests/unit/io/test_scan.py::test_scan_include_file_name[False-scan_csv-write_csv]": "Need to add include_file_path to IR",
    "tests/unit/io/test_scan.py::test_scan_include_file_name[False-scan_ndjson-write_ndjson]": "Need to add include_file_path to IR",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[gpu]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[engine1]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_lazyframe.py::test_round[dtype1-123.55-1-123.6]": "Rounding midpoints is handled incorrectly",
    "tests/unit/lazyframe/test_lazyframe.py::test_cast_frame": "Casting that raises not supported on GPU",
    "tests/unit/lazyframe/test_lazyframe.py::test_lazy_cache_hit": "Debug output on stderr doesn't match",
    "tests/unit/operations/aggregation/test_aggregations.py::test_duration_function_literal": "Broadcasting inside groupby-agg not supported",
    "tests/unit/operations/aggregation/test_aggregations.py::test_sum_empty_and_null_set": "libcudf sums column of all nulls to null, not zero",
    "tests/unit/operations/aggregation/test_aggregations.py::test_binary_op_agg_context_no_simplify_expr_12423": "groupby-agg of just literals should not produce collect_list",
    "tests/unit/operations/aggregation/test_aggregations.py::test_nan_inf_aggregation": "treatment of nans and nulls together is different in libcudf and polars in groupby-agg context",
    "tests/unit/operations/arithmetic/test_list_arithmetic.py::test_list_arithmetic_values[func0-func0-none]": "cudf-polars doesn't nullify division by zero",
    "tests/unit/operations/arithmetic/test_list_arithmetic.py::test_list_arithmetic_values[func0-func1-none]": "cudf-polars doesn't nullify division by zero",
    "tests/unit/operations/arithmetic/test_list_arithmetic.py::test_list_arithmetic_values[func0-func2-none]": "cudf-polars doesn't nullify division by zero",
    "tests/unit/operations/arithmetic/test_list_arithmetic.py::test_list_arithmetic_values[func0-func3-none]": "cudf-polars doesn't nullify division by zero",
    "tests/unit/operations/arithmetic/test_list_arithmetic.py::test_list_arithmetic_values[func1-func0-none]": "cudf-polars doesn't nullify division by zero",
    "tests/unit/operations/arithmetic/test_list_arithmetic.py::test_list_arithmetic_values[func1-func1-none]": "cudf-polars doesn't nullify division by zero",
    "tests/unit/operations/arithmetic/test_list_arithmetic.py::test_list_arithmetic_values[func1-func2-none]": "cudf-polars doesn't nullify division by zero",
    "tests/unit/operations/arithmetic/test_list_arithmetic.py::test_list_arithmetic_values[func1-func3-none]": "cudf-polars doesn't nullify division by zero",
    "tests/unit/operations/test_abs.py::test_abs_duration": "Need to raise for unsupported uops on timelike values",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input7-expected7-Float32-Float32]": "Mismatching dtypes, needs cudf#15852",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input10-expected10-Date-output_dtype10]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input11-expected11-input_dtype11-output_dtype11]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input12-expected12-input_dtype12-output_dtype12]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_mean_by_dtype[input13-expected13-input_dtype13-output_dtype13]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input7-expected7-Float32-Float32]": "Mismatching dtypes, needs cudf#15852",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input10-expected10-Date-output_dtype10]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input11-expected11-input_dtype11-output_dtype11]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input12-expected12-input_dtype12-output_dtype12]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input13-expected13-input_dtype13-output_dtype13]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input14-expected14-input_dtype14-output_dtype14]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input15-expected15-input_dtype15-output_dtype15]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input16-expected16-input_dtype16-output_dtype16]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_binary_agg_with_literal": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_group_by.py::test_aggregated_scalar_elementwise_15602": "Unsupported boolean function/dtype combination in groupby-agg",
    "tests/unit/operations/test_group_by.py::test_schemas[data1-expr1-expected_select1-expected_gb1]": "Mismatching dtypes, needs cudf#15852",
    "tests/unit/operations/test_join.py::test_cross_join_slice_pushdown": "Need to implement slice pushdown for cross joins",
    "tests/unit/sql/test_cast.py::test_cast_errors[values0-values::uint8-conversion from `f64` to `u64` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values1-values::uint4-conversion from `i64` to `u32` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values2-values::int1-conversion from `i64` to `i8` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values5-values::int4-conversion from `str` to `i32` failed]": "Cast raises, but error user receives is wrong",
    "tests/unit/sql/test_miscellaneous.py::test_read_csv": "Incorrect handling of missing_is_null in read_csv",
    "tests/unit/sql/test_wildcard_opts.py::test_select_wildcard_errors": "Raises correctly but with different exception",
    "tests/unit/streaming/test_streaming_io.py::test_parquet_eq_statistics": "Debug output on stderr doesn't match",
    "tests/unit/test_cse.py::test_cse_predicate_self_join": "Debug output on stderr doesn't match",
    "tests/unit/test_empty.py::test_empty_9137": "Mismatching dtypes, needs cudf#15852",
    "tests/unit/test_errors.py::test_error_on_empty_group_by": "Incorrect exception raised",
    # Maybe flaky, order-dependent?
    "tests/unit/test_projections.py::test_schema_full_outer_join_projection_pd_13287": "Order-specific result check, query is correct but in different order",
    "tests/unit/test_queries.py::test_group_by_agg_equals_zero_3535": "libcudf sums all nulls to null, not zero",
}


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark known failing tests."""
    if config.getoption("--cudf-polars-no-fallback"):
        # Don't xfail tests if running without fallback
        return
    for item in items:
        if item.nodeid in EXPECTED_FAILURES:
            item.add_marker(pytest.mark.xfail(reason=EXPECTED_FAILURES[item.nodeid]))
