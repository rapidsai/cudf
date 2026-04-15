# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running polars test suite setting GPU engine as default."""

from __future__ import annotations

from functools import partialmethod
from typing import TYPE_CHECKING

import pytest

import polars

from cudf_polars.utils.config import Runtime, StreamingFallbackMode

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


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
    group.addoption(
        "--executor",
        action="store",
        default="in-memory",
        choices=("in-memory", "streaming"),
        help="Executor to use for GPUEngine.",
    )
    group.addoption(
        "--blocksize-mode",
        action="store",
        default="default",
        choices=("small", "default"),
        help=(
            "Blocksize to use for 'streaming' executor. Set to 'small' "
            "to run most tests with multiple partitions."
        ),
    )
    group.addoption(
        "--runtime",
        action="store",
        default="tasks",
        choices=("tasks", "rapidsmpf"),
        help="Runtime to use for the 'streaming' executor.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Enable use of this module as a pytest plugin to enable GPU collection."""
    no_fallback = config.getoption("--cudf-polars-no-fallback")
    executor = config.getoption("--executor")
    blocksize_mode = config.getoption("--blocksize-mode")
    runtime = config.getoption("--runtime")
    if no_fallback:
        collect = polars.LazyFrame.collect
        engine = polars.GPUEngine(raise_on_fail=no_fallback)
        # https://github.com/python/mypy/issues/2427
        polars.LazyFrame.collect = partialmethod(collect, engine=engine)  # type: ignore[method-assign, assignment]
    elif executor == "in-memory":
        collect = polars.LazyFrame.collect
        engine = polars.GPUEngine(executor=executor)
        polars.LazyFrame.collect = partialmethod(collect, engine=engine)  # type: ignore[method-assign, assignment]
    elif executor == "streaming" and blocksize_mode == "small":
        executor_options: dict[str, Any] = {}
        executor_options["max_rows_per_partition"] = 4
        executor_options["target_partition_size"] = 10
        # We expect many tests to fall back, so silence the warnings
        executor_options["fallback_mode"] = StreamingFallbackMode.SILENT
        executor_options["runtime"] = Runtime[runtime.upper()]
        collect = polars.LazyFrame.collect
        engine = polars.GPUEngine(executor=executor, executor_options=executor_options)
        polars.LazyFrame.collect = partialmethod(collect, engine=engine)  # type: ignore[method-assign, assignment]
    else:
        # run with streaming executor and default blocksize
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
    "tests/unit/io/test_csv.py::test_read_csv_only_loads_selected_columns": "Memory usage won't be correct due to GPU",
    "tests/unit/io/test_delta.py::test_scan_delta_version": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_scan_delta_relative": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_read_delta_version": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_scan_delta_schema_evolution_nested_struct_field_19915": "Need to expose hive partitioning",
    "tests/unit/io/test_delta.py::test_scan_delta_nanosecond_timestamp": "polars generates the wrong schema: https://github.com/pola-rs/polars/issues/23949",
    "tests/unit/io/test_delta.py::test_scan_delta_nanosecond_timestamp_nested": "polars generates the wrong schema: https://github.com/pola-rs/polars/issues/23949",
    "tests/unit/io/test_lazy_count_star.py::test_count_parquet[small.parquet-4]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_count_star.py::test_count_parquet[foods*.parquet-54]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_is_in_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_statistics": "Debug output on stderr doesn't match",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type0]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type1]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type2]": "partition sinks not yet supported in standard engine.",
    "tests/unit/io/test_partition.py::test_partition_to_memory[io_type3]": "partition sinks not yet supported in standard engine.",
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
    "tests/unit/io/test_sink.py::test_collect_all_lazy": "SinkMultiple not supported by InMemory CPU Engine, which we fallback to. See pola-rs/polars/pull/26537",
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
    "tests/unit/io/test_parquet.py::test_binary_offset_roundtrip": "binary offset type unsupported",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[gpu]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_engine_selection.py::test_engine_import_error_raises[engine1]": "Expect this to pass because cudf-polars is installed",
    "tests/unit/lazyframe/test_lazyframe.py::test_round[dtype2-123.55-1-123.6]": "libcudf HALF_EVEN rounding bug for Float64 with decimal_places > 0. See https://github.com/rapidsai/cudf/issues/21319",
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
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input14-expected14-input_dtype14-output_dtype14]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input15-expected15-input_dtype15-output_dtype15]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_median_by_dtype[input16-expected16-input_dtype16-output_dtype16]": "Unsupported groupby-agg for a particular dtype",
    "tests/unit/operations/test_group_by.py::test_group_by_binary_agg_with_literal": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_group_by.py::test_group_by_lit_series": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_group_by.py::test_group_by_series_lit_22103[False]": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_group_by.py::test_group_by_series_lit_22103[True]": "Incorrect broadcasting of literals in groupby-agg",
    "tests/unit/operations/test_join.py::test_cross_join_slice_pushdown": "Need to implement slice pushdown for cross joins",
    # We match the behavior of the polars[cpu] streaming engine (it makes doesn't make any ordering guarantees either when maintain_order is none).
    # But this test does because the test is run with the polars[cpu] in-memory engine, which still preserves the order of the left dataframe
    # when maintain order is none.
    "tests/unit/operations/test_join.py::test_join_preserve_order_left": "polars[gpu] makes no ordering guarantees when maintain_order is none",
    # TODO: As of polars 1.34, the column names for left and right came in unaligned, which causes the dtypes to mismatch when calling plc.replace.replace_nulls
    # Need to investigate what changed in polars
    "tests/unit/operations/test_join.py::test_join_coalesce_column_order_23177": "Misaligned left/right column names left and right tables in join op",
    "tests/unit/operations/namespaces/string/test_pad.py::test_str_zfill_unicode_not_respected": "polars doesn't add zeros for unicode characters.",
    "tests/unit/sql/test_cast.py::test_cast_errors[values0-values::uint8-conversion from `f64` to `u64` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values1-values::uint4-conversion from `i64` to `u32` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values2-values::int1-conversion from `i64` to `i8` failed]": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_cast.py::test_cast_errors[values5-values::int4-conversion from `str` to `i32` failed]": "Cast raises, but error user receives is wrong",
    "tests/unit/lazyframe/test_predicates.py::test_predicate_pushdown_split_pushable": "Casting that raises not supported on GPU",
    "tests/unit/sql/test_miscellaneous.py::test_read_csv": "Incorrect handling of missing_is_null in read_csv",
    "tests/unit/lazyframe/test_cse.py::test_cse_predicate_self_join[False]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_scan_row_deletion.py::test_scan_row_deletion_skips_file_with_all_rows_deleted": "The test intentionally corrupts the parquet file, so we cannot read the row count from the header.",
    "tests/unit/io/test_multiscan.py::test_multiscan_row_index[scan_csv-write_csv-csv]": "Debug output on stderr doesn't match",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-columns]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-row_groups]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-prefiltered]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[True-none]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-columns]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-row_groups]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-prefiltered]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_schema_arg[False-none]": "allow_missing_columns argument in read_parquet not translated in IR",
    "tests/unit/io/test_multiscan.py::test_multiscan_row_index[scan_csv-write_csv]": "CSV multiscan with row_index and no row limit is not yet supported.",
    "tests/unit/operations/namespaces/test_binary.py::test_binary_compounded_literal_aggstate_24460": "List literal loses nesting in gather: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_gather_get_on_literal_24101[lit1-0-False]": "List literal loses nesting in gather: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_gather_get_on_literal_24101[lit1-idx1-False]": "List literal loses nesting in gather: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_gather_get_on_literal_24101[lit1-idx2-False]": "List literal loses nesting in gather: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_head_tail_on_literal_24102[lit1-1-False]": "List literal loses nesting in head/tail: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_head_tail_on_literal_24102[lit1-len1-False]": "List literal loses nesting in head/tail: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_slice_on_literal_23999[lit1-0-0-False]": "List literal loses nesting in slice: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_slice_on_literal_23999[lit1-0-len1-False]": "List literal loses nesting in slice: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_slice_on_literal_23999[lit1-offset1-0-False]": "List literal loses nesting in slice: cudf#19610",
    "tests/unit/operations/test_slice.py::test_schema_slice_on_literal_23999[lit1-offset1-len1-False]": "List literal loses nesting in slice: cudf#19610",
    "tests/unit/functions/test_concat.py::test_concat_horizontally_strict": "polars doesnt hand us the hconcat options. Fixed in 1.39.",
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
    "tests/unit/operations/namespaces/list/test_list.py::test_list_struct_field_perf": "Tests CPU Engine perf",
    "tests/benchmark/test_with_columns.py::test_with_columns_quadratic_19503": "Tests performance bug in CPU engine",
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
    # New iceberg release causes this test to fail. We can remove in the next polars version bump: https://github.com/rapidsai/cudf/pull/19912
    "tests/unit/io/test_iceberg.py::test_fill_missing_fields_with_identity_partition_values[False]": "https://github.com/pola-rs/polars/pull/24456",
    "tests/unit/operations/test_rolling.py::test_rolling_agg_bad_input_types[str]": "https://github.com/rapidsai/cudf/issues/20551",
    "tests/unit/operations/test_group_by_dynamic.py::test_group_by_dynamic_agg_bad_input_types[str]": "TODO: Need to investigate why this fails in CI but passes locally. We should fallback to CPU for group_by_dynamic",
    "tests/unit/expr/test_exprs.py::test_exp_log1p[Float16-Float16]": "Flaky test: Small floating-point precision differences in exp/log1p results",
    # TODO: Investigate why these tests fail in CI but pass locally.
    "tests/unit/io/test_delta.py::test_scan_delta_extract_table_statistics_df": "schemas mismatch: dtypes different",
    "tests/unit/io/test_partition.py::test_sink_partitioned_no_columns_in_file_25535[scan_parquet-sink_parquet]": "Incorrect row count. Related to https://github.com/rapidsai/cudf/issues/21428",
    "tests/unit/operations/test_group_by.py::test_unique_head_tail_26429[0]": "ZeroDivisionError: division by zero",
}


STREAMING_ONLY_EXPECTED_FAILURES: Mapping[str, str] = {
    # Add tests that are expected to fail with the streaming executor
}

RAPIDSMPF_TESTS_TO_SKIP: Mapping[str, str] = {
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q1": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q2": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q3": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q4": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q5": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q7": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_group_by.py::test_groupby_h2oai_q10": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_join_where.py::test_single_inequality": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_join_where.py::test_non_strict_inequalities": "Too slow with --blocksize-mode small",
    "tests/benchmark/test_join_where.py::test_strict_inequalities": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_partition.py::test_partition_approximate_size": "Too slow for CI",
    "tests/unit/io/test_lazy_parquet.py::test_parquet_many_row_groups_12297": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan[single-parquet-async]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan[single-parquet-sync]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[glob-parquet-async]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[glob-parquet-sync]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[single-parquet-async]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_filter[single-parquet-sync]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[glob-parquet-async]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[glob-parquet-sync]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[single-parquet-async]": "Takes >60 seconds to run locally",
    "tests/unit/io/test_scan.py::test_scan_with_filter_and_limit[single-parquet-sync]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[glob-parquet-async]": "Takes >60 seconds to run locally",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[glob-parquet-sync]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[single-parquet-async]": "Too slow with --blocksize-mode small",
    "tests/unit/io/test_scan.py::test_scan_with_row_index_projected_out[single-parquet-sync]": "Too slow with --blocksize-mode small",
    "tests/unit/lazyframe/test_optimizations.py::test_collapse_joins_combinations": "Too slow for CI",
    "tests/unit/operations/test_slice.py::test_slice_slice_pushdown": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-10432-False]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-10432-True]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-10432-False]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-10432-True]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-10432-False]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-10432-True]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Categorical-10432-True]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Categorical-10432-False]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[String-1056-False]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Boolean-1056-False]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_first_last_big[Int32-1056-False]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_group_by_lit_series": "Incorrect broadcasting of literals in groupby-agg but failure leads to segfaults https://github.com/rapidsai/cudf/issues/22138",
    "tests/unit/operations/test_group_by.py::test_group_by_series_partitioned": "https://github.com/rapidsai/cudf/issues/22072 but failure leads to segfaults https://github.com/rapidsai/cudf/issues/22138",
    "tests/unit/operations/test_group_by.py::test_overflow_mean_partitioned_group_by_5194[Int32]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_overflow_mean_partitioned_group_by_5194[UInt32]": "Too slow with --blocksize-mode small",
    "tests/unit/operations/test_group_by.py::test_partitioned_group_by_chunked": "https://github.com/rapidsai/cudf/issues/22072 but failure leads to segfaults https://github.com/rapidsai/cudf/issues/22138",
    "tests/unit/streaming/test_streaming_sort.py::test_streaming_sort_varying_order_and_dtypes[sort_by0]": "Too slow for CI",
}

RAPIDSMPF_ONLY_EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/unit/functions/range/test_linear_space.py::test_linear_space_num_samples_expr": "https://github.com/rapidsai/cudf/issues/22072",
    "tests/unit/interop/test_interop.py::test_0_width_df_roundtrip": "https://github.com/rapidsai/cudf/issues/21644",
    "tests/unit/lazyframe/test_cse.py::test_cse_10441": "https://github.com/rapidsai/cudf/issues/21645",
    "tests/unit/lazyframe/test_cse.py::test_cse_10452": "https://github.com/rapidsai/cudf/issues/21645",
    "tests/unit/lazyframe/test_cse.py::test_cse_expr_selection_context": "https://github.com/rapidsai/cudf/issues/21645",
    "tests/unit/lazyframe/test_cse.py::test_cse_non_scalar_length_mismatch_17732": "https://github.com/rapidsai/cudf/issues/21645",
    "tests/unit/lazyframe/test_projections.py::test_join_projection_pushdown_struct_field_as_key_24446": "https://github.com/rapidsai/cudf/issues/22105",
    "tests/unit/operations/test_group_by.py::test_group_by_unique_parametric[n_unique-True-True]": "https://github.com/rapidsai/cudf/issues/21641",
    "tests/unit/operations/test_group_by.py::test_unique_head_tail_26429[4]": "https://github.com/rapidsai/cudf/issues/22075",
    "tests/unit/operations/test_join.py::test_empty_outer_join_22206": "https://github.com/rapidsai/cudf/issues/22084",
    "tests/unit/operations/test_join.py::test_join_numeric_key_upcast_15338[True-dtypes20]": "https://github.com/rapidsai/cudf/issues/22085",
    "tests/unit/operations/test_join.py::test_join_numeric_key_upcast_15338[False-dtypes14]": "https://github.com/rapidsai/cudf/issues/22085",
    "tests/unit/operations/test_join.py::test_join_numeric_key_upcast_15338[False-dtypes40]": "https://github.com/rapidsai/cudf/issues/22085",
    "tests/unit/operations/test_join.py::test_join_numeric_key_upcast_15338[False-dtypes44]": "https://github.com/rapidsai/cudf/issues/22085",
    "tests/unit/operations/test_join.py::test_join_numeric_key_upcast_order": "https://github.com/rapidsai/cudf/issues/22085",
    "tests/unit/operations/test_join.py::test_semi_anti_join": "https://github.com/rapidsai/cudf/issues/22049",
    "tests/unit/operations/test_shift.py::test_shift_expr": "https://github.com/rapidsai/cudf/issues/21645",
    "tests/unit/operations/test_shift.py::test_shift_fill_value": "https://github.com/rapidsai/cudf/issues/21645",
    "tests/unit/operations/test_slice.py::test_slice_pushdown_literal_projection_14349": "https://github.com/rapidsai/cudf/issues/22072",
    "tests/unit/operations/test_top_k.py::test_top_k_non_elementwise_by_24163": "https://github.com/rapidsai/cudf/issues/22074",
    "tests/unit/sql/test_joins.py::test_cross_join_unnest_from_cte": "https://github.com/rapidsai/cudf/issues/22073",
    "tests/unit/sql/test_joins.py::test_join_on_mixed_expression_conditions[df10-df20-df1.category = df2.category AND (df1.code * 2) = df2.code_doubled-df1.name, df1.code, df2.type-expected0-schema0]": "https://github.com/rapidsai/cudf/issues/22085 (or similar)",
    "tests/unit/sql/test_joins.py::test_join_on_mixed_expression_conditions[df11-df21-df1.id = df2.id AND LOWER(df1.name) = df2.match-df1.id, df1.name, df2.match-expected1-schema1]": "https://github.com/rapidsai/cudf/issues/22085 (or similar)",
    "tests/unit/sql/test_joins.py::test_join_on_expression_with_literals[df10-df20-df1.id = df2.id AND df1.multiplier * 5 = df2.base AND df1.category = 'A'-df1.id, df1.multiplier, df2.base-expected0-schema0]": "https://github.com/rapidsai/cudf/issues/22085 (or similar)",
    "tests/unit/sql/test_qualify.py::test_qualify_constraints[above_avg]": "https://github.com/rapidsai/cudf/issues/22050",
    "tests/unit/sql/test_qualify.py::test_qualify_constraints[equals_max]": "https://github.com/rapidsai/cudf/issues/22050",
    "tests/unit/sql/test_qualify.py::test_qualify_constraints[compound_expr]": "https://github.com/rapidsai/cudf/issues/22050",
    "tests/unit/sql/test_qualify.py::test_qualify_distinct": "https://github.com/rapidsai/cudf/issues/22050",
    "tests/unit/sql/test_qualify.py::test_qualify_matches_all_rows[sum_window]": "https://github.com/rapidsai/cudf/issues/22050",
    "tests/unit/sql/test_qualify.py::test_qualify_matches_all_rows[count_window]": "https://github.com/rapidsai/cudf/issues/22050",
    "tests/unit/sql/test_qualify.py::test_qualify_multiple_clauses": "https://github.com/rapidsai/cudf/issues/22050",
    "tests/unit/sql/test_qualify.py::test_qualify_with_internal_cumulative_sum": "https://github.com/rapidsai/cudf/issues/22050",
    "tests/unit/sql/test_qualify.py::test_qualify_with_where_clause": "https://github.com/rapidsai/cudf/issues/22050",
}


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark known failing tests."""
    if config.getoption("--cudf-polars-no-fallback"):
        # Don't xfail tests if running without fallback
        return
    with_rapidsmpf = config.getoption("--runtime") == "rapidsmpf"
    with_streaming = config.getoption("--executor") == "streaming"
    for item in items:
        if (reason := TESTS_TO_SKIP.get(item.nodeid, None)) is not None or (
            with_rapidsmpf
            and (reason := RAPIDSMPF_TESTS_TO_SKIP.get(item.nodeid, None)) is not None
        ):
            item.add_marker(pytest.mark.skip(reason=reason))
        elif (
            with_rapidsmpf
            and (r_reason := RAPIDSMPF_ONLY_EXPECTED_FAILURES.get(item.nodeid, None))
            is not None
        ):
            item.add_marker(pytest.mark.xfail(reason=r_reason))
        elif (
            with_streaming
            and (s_reason := STREAMING_ONLY_EXPECTED_FAILURES.get(item.nodeid, None))
            is not None
        ):
            # Also sets --runtime=rapidsmpf also sets --executor=streaming, so check last
            item.add_marker(pytest.mark.xfail(reason=s_reason))
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
