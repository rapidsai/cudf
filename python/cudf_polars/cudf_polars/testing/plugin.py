# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
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
    if no_fallback:
        collect = polars.LazyFrame.collect
        engine = polars.GPUEngine(raise_on_fail=no_fallback)
        # https://github.com/python/mypy/issues/2427
        polars.LazyFrame.collect = partialmethod(collect, engine=engine)  # type: ignore[method-assign,assignment]
    else:
        polars.Config.set_engine_affinity("gpu")
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*GPU engine does not support streaming or background collection",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*Query execution with GPU not possible",
    )


EXPECTED_FAILURES: Mapping[str, str | tuple[str, bool]] = {
    "tests/unit/io/test_csv.py::test_compressed_csv": "Need to determine if file is compressed",
    "tests/unit/io/test_csv.py::test_read_csv_only_loads_selected_columns": "Memory usage won't be correct due to GPU",
    "tests/unit/io/test_delta.py::test_scan_delta_version": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_scan_delta_relative": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_read_delta_version": "Need to expose hive partitioning",
    "tests/unit/io/test_lazy_count_star.py::test_count_compressed_csv_18057": "Need to determine if file is compressed",
    "tests/unit/io/test_lazy_count_star.py::test_count_parquet[small.parquet-4]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_count_star.py::test_count_parquet[foods*.parquet-54]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_csv.py::test_scan_csv_slice_offset_zero": "Integer overflow in sliced read",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_is_in_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_key_order_22645": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_finish_callback[io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_finish_callback[io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_finish_callback[io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df1-a-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df2-sorts2-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df2-sorts2-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df2-sorts2-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df2-sorts2-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df3-b-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df3-b-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df3-b-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df3-b-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df4-sorts4-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df4-sorts4-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df4-sorts4-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df4-sorts4-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_finish_callback[io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df0-a-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df0-a-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df0-a-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df0-a-io_type3]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df1-a-io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df1-a-io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory_sort_by[df1-a-io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_ignores_dtype_mismatch_for_non_projected_columns_19249[False-False]": "Needs some variant of cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_scan_parquet_ignores_dtype_mismatch_for_non_projected_columns_19249[True-False]": "Needs some variant of cudf#16394",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_unaligned_schema_read[False]": "Incomplete handling of projected reads with mismatching schemas, cudf#16394",
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
    "tests/unit/io/test_parquet.py::test_scan_parquet_filter_statistics_load_missing_column_21391": "Mismatching column read cudf#16394",
    "tests/unit/io/test_parquet.py::test_field_overwrites_metadata": "cannot serialize in-memory sink target.",
    "tests/unit/io/test_parquet_field_overwrites.py::test_required_flat": "cannot serialize in-memory sink target.",
    "tests/unit/io/test_parquet_field_overwrites.py::test_required_list[dtype0]": "cannot serialize in-memory sink target.",
    "tests/unit/io/test_parquet_field_overwrites.py::test_required_list[dtype1]": "cannot serialize in-memory sink target.",
    "tests/unit/io/test_parquet_field_overwrites.py::test_required_struct": "cannot serialize in-memory sink target.",
    "tests/unit/io/test_write.py::test_write_async[read_parquet-write_parquet]": "Need to add include_file_path to IR",
    "tests/unit/io/test_write.py::test_write_async[<lambda>-write_csv]": "Need to add include_file_path to IR",
    "tests/unit/io/test_write.py::test_write_async[read_parquet-<lambda>]": "Need to add include_file_path to IR",
    "tests/unit/io/test_write.py::test_write_async[<lambda>-<lambda>0]": "Need to add include_file_path to IR",
    "tests/unit/io/test_write.py::test_write_async[<lambda>-<lambda>2]": "Need to add include_file_path to IR",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[gpu]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[engine1]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_lazyframe.py::test_round[dtype1-123.55-1-123.6]": "Rounding midpoints is handled incorrectly",
    "tests/unit/lazyframe/test_lazyframe.py::test_cast_frame": "Casting that raises not supported on GPU",
    "tests/unit/lazyframe/test_lazyframe.py::test_lazy_cache_hit": "Debug output on stderr doesn't match",
    "tests/unit/operations/aggregation/test_aggregations.py::test_binary_op_agg_context_no_simplify_expr_12423": "groupby-agg of just literals should not produce collect_list",
    "tests/unit/operations/aggregation/test_aggregations.py::test_nan_inf_aggregation": "treatment of nans and nulls together is different in libcudf and polars in groupby-agg context",
    "tests/unit/operations/test_abs.py::test_abs_duration": "Need to raise for unsupported uops on timelike values",
    "tests/unit/operations/test_group_by.py::test_group_by_shorthand_quantile": "libcudf quantiles are round to nearest ties to even, polars quantiles are round to nearest ties away from zero",
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
    "tests/unit/operations/test_group_by.py::test_group_by_lit_series": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_join.py::test_cross_join_slice_pushdown": "Need to implement slice pushdown for cross joins",
    "tests/unit/operations/test_rolling.py::test_rolling_group_by_empty_groups_by_take_6330": "Ordering difference, might be polars bug",
    "tests/unit/sql/test_cast.py::test_cast_errors[values0-values::uint8-conversion from `f64` to `u64` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values1-values::uint4-conversion from `i64` to `u32` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values2-values::int1-conversion from `i64` to `i8` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values5-values::int4-conversion from `str` to `i32` failed]": "Cast raises, but error user receives is wrong",
    "tests/unit/sql/test_miscellaneous.py::test_read_csv": "Incorrect handling of missing_is_null in read_csv",
    "tests/unit/sql/test_wildcard_opts.py::test_select_wildcard_errors": "Raises correctly but with different exception",
    "tests/unit/test_cse.py::test_cse_predicate_self_join": "Debug output on stderr doesn't match",
    "tests/unit/test_cse.py::test_nested_cache_no_panic_16553": "Needs https://github.com/rapidsai/cudf/issues/18630",
    "tests/unit/test_errors.py::test_error_on_empty_group_by": "Incorrect exception raised",
    "tests/unit/test_predicates.py::test_predicate_pushdown_split_pushable": "Casting that raises not supported on GPU",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type0]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type1]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type2]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan.py::test_async_read_21945[scan_type3]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan_row_deletion.py::test_scan_row_deletion_skips_file_with_all_rows_deleted": "The test intentionally corrupts the parquet file, so we cannot read the row count from the header.",
    "tests/unit/io/test_multiscan.py::test_multiscan_row_index[scan_csv-write_csv-csv]": "Debug output on stderr doesn't match",
    "tests/unit/functions/range/test_linear_space.py::test_linear_space_date": "Needs https://github.com/pola-rs/polars/issues/23020",
    "tests/unit/sql/test_temporal.py::test_implicit_temporal_strings[dt IN ('1960-01-07','2077-01-01','2222-02-22')-expected15]": "Needs https://github.com/pola-rs/polars/issues/23020",
    "tests/unit/sql/test_operators.py::test_in_not_in[dt NOT IN ('1950-12-24', '1997-07-05')]": "Needs https://github.com/pola-rs/polars/issues/23020",
    "tests/unit/sql/test_operators.py::test_in_not_in[dt IN ('2020-10-10', '2077-03-18')]": "Needs https://github.com/pola-rs/polars/issues/23020",
    "tests/unit/datatypes/test_struct.py::test_struct_agg_all": "Needs nested list[struct] support",
    "tests/unit/constructors/test_structs.py::test_constructor_non_strict_schema_17956": "Needs nested list[struct] support",
    "tests/unit/io/test_delta.py::test_read_delta_arrow_map_type": "Needs nested list[struct] support",
    "tests/unit/lazyframe/test_collect_schema.py::test_collect_schema_parametric": "https://github.com/pola-rs/polars/issues/23214",
    "tests/unit/datatypes/test_struct.py::test_struct_null_cast": "pylibcudf.Scalar does not support struct scalars",
    "tests/unit/datatypes/test_struct.py::test_struct_outer_nullability_zip_18119": "pylibcudf.Scalar does not support struct scalars",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-columns]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-row_groups]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-prefiltered]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-none]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-columns]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-row_groups]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-prefiltered]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-none]": "allow_missing_columns argument in read_parquet not translated in IR",
}


TESTS_TO_SKIP: Mapping[str, str] = {
    "tests/unit/operations/test_profile.py::test_profile_with_cse": "Shape assertion won't match",
    # On Ubuntu 20.04, the tzdata package contains a bunch of symlinks
    # for obsolete timezone names. However, the chrono_tz package that
    # polars uses doesn't read /usr/share/zoneinfo, instead packaging
    # the current zoneinfo database from IANA. Consequently, when this
    # hypothesis-generated test runs and generates timezones from the
    # available zoneinfo-reported timezones, we can get an error from
    # polars that the requested timezone is unknown.
    # Since this is random, just skip it, rather than xfailing.
    "tests/unit/lazyframe/test_serde.py::test_lf_serde_roundtrip_binary": "chrono_tz doesn't have all tzdata symlink names",
    # Tests performance difference of CPU engine
    "tests/unit/operations/test_join.py::test_join_where_eager_perf_21145": "Tests performance bug in CPU engine",
    # The test may segfault with the legacy streaming engine. We should
    # remove this skip when all polars tests use the new streaming engine.
    "tests/unit/streaming/test_streaming_group_by.py::test_streaming_group_by_literal[1]": "May segfault w/the legacy streaming engine",
    # Fails in CI, but passes locally
    "tests/unit/streaming/test_streaming.py::test_streaming_streamable_functions": "RuntimeError: polars_python::sql::PySQLContext is unsendable, but is being dropped on another thread",
    # Remove when polars supports Pydantic V3
    "tests/unit/constructors/test_constructors.py::test_init_structured_objects": "pydantic deprecation warning",
    "tests/unit/constructors/test_constructors.py::test_init_pydantic_2x": "pydantic deprecation warning",
    "tests/unit/constructors/test_constructors.py::test_init_structured_objects_nested[_TestFooPD-_TestBarPD-_TestBazPD]": "pydantic deprecation warning",
    "tests/unit/series/test_series.py::test_init_structured_objects": "pydantic deprecation warning",
    "tests/unit/series/test_describe.py::test_series_describe_float": "https://github.com/rapidsai/cudf/issues/19324",
    "tests/unit/series/test_describe.py::test_series_describe_int": "https://github.com/rapidsai/cudf/issues/19324",
    "tests/unit/streaming/test_streaming.py::test_streaming_apply": "https://github.com/pola-rs/polars/issues/22558",
}


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark known failing tests."""
    if config.getoption("--cudf-polars-no-fallback"):
        # Don't xfail tests if running without fallback
        return
    for item in items:
        if (reason := TESTS_TO_SKIP.get(item.nodeid, None)) is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
        elif (entry := EXPECTED_FAILURES.get(item.nodeid, None)) is not None:
            if isinstance(entry, tuple):
                # the second entry in the tuple is the condition to xfail on
                reason, condition = entry
                item.add_marker(
                    pytest.mark.xfail(
                        condition=condition,
                        reason=reason,
                    ),
                )
            else:
                item.add_marker(pytest.mark.xfail(reason=entry))
