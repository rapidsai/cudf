# Copyright (c) 2024, NVIDIA CORPORATION.
import io

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from pyarrow.parquet import read_table
from utils import assert_table_and_meta_eq, make_source

import pylibcudf as plc
from pylibcudf.expressions import (
    ASTOperator,
    ColumnNameReference,
    ColumnReference,
    Literal,
    Operation,
)

# Shared kwargs to pass to make_source
_COMMON_PARQUET_SOURCE_KWARGS = {"format": "parquet"}


@pytest.mark.parametrize("columns", [None, ["col_int64", "col_bool"]])
def test_read_parquet_basic(
    table_data, binary_source_or_sink, nrows_skiprows, columns
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    res = plc.io.parquet.read_parquet(
        plc.io.SourceInfo([source]),
        nrows=nrows,
        skip_rows=skiprows,
        columns=columns,
    )

    if columns is not None:
        pa_table = pa_table.select(columns)

    # Adapt to nrows/skiprows
    pa_table = pa_table.slice(
        offset=skiprows, length=nrows if nrows != -1 else None
    )

    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)


@pytest.mark.parametrize(
    "pa_filters,plc_filters",
    [
        (
            pc.field("col_int64") >= 10,
            Operation(
                ASTOperator.GREATER_EQUAL,
                ColumnNameReference("col_int64"),
                Literal(plc.interop.from_arrow(pa.scalar(10))),
            ),
        ),
        (
            (pc.field("col_int64") >= 10) & (pc.field("col_double") < 0),
            Operation(
                ASTOperator.LOGICAL_AND,
                Operation(
                    ASTOperator.GREATER_EQUAL,
                    ColumnNameReference("col_int64"),
                    Literal(plc.interop.from_arrow(pa.scalar(10))),
                ),
                Operation(
                    ASTOperator.LESS,
                    ColumnNameReference("col_double"),
                    Literal(plc.interop.from_arrow(pa.scalar(0.0))),
                ),
            ),
        ),
        (
            (pc.field(0) == 10),
            Operation(
                ASTOperator.EQUAL,
                ColumnReference(0),
                Literal(plc.interop.from_arrow(pa.scalar(10))),
            ),
        ),
    ],
)
def test_read_parquet_filters(
    table_data, binary_source_or_sink, pa_filters, plc_filters
):
    _, pa_table = table_data

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    plc_table_w_meta = plc.io.parquet.read_parquet(
        plc.io.SourceInfo([source]), filters=plc_filters
    )
    exp = read_table(source, filters=pa_filters)
    assert_table_and_meta_eq(
        exp, plc_table_w_meta, check_field_nullability=False
    )


# TODO: Test these options
# list row_groups = None,
# ^^^ This one is not tested since it's not in pyarrow/pandas, deprecate?
# bool convert_strings_to_categories = False,
# bool use_pandas_metadata = True


@pytest.mark.parametrize(
    "compression",
    [
        plc.io.types.CompressionType.NONE,
        plc.io.types.CompressionType.GZIP,
    ],
)
@pytest.mark.parametrize(
    "stats_level",
    [
        plc.io.types.StatisticsFreq.STATISTICS_NONE,
        plc.io.types.StatisticsFreq.STATISTICS_COLUMN,
    ],
)
@pytest.mark.parametrize("int96_timestamps", [True, False])
@pytest.mark.parametrize("write_v2_headers", [True, False])
@pytest.mark.parametrize(
    "dictionary_policy",
    [
        plc.io.types.DictionaryPolicy.ADAPTIVE,
        plc.io.types.DictionaryPolicy.NEVER,
    ],
)
@pytest.mark.parametrize("utc_timestamps", [True, False])
@pytest.mark.parametrize("write_arrow_schema", [True, False])
@pytest.mark.parametrize(
    "partitions",
    [None, [plc.io.types.PartitionInfo.from_start_and_num(0, 10)]],
)
@pytest.mark.parametrize("column_chunks_file_paths", [None, ["tmp.parquet"]])
@pytest.mark.parametrize("row_group_size_bytes", [None, 100])
@pytest.mark.parametrize("row_group_size_rows", [None, 1])
@pytest.mark.parametrize("max_page_size_bytes", [None, 100])
@pytest.mark.parametrize("max_page_size_rows", [None, 1])
@pytest.mark.parametrize("max_dictionary_size", [None, 100])
def test_write_parquet(
    table_data,
    compression,
    stats_level,
    int96_timestamps,
    write_v2_headers,
    dictionary_policy,
    utc_timestamps,
    write_arrow_schema,
    partitions,
    column_chunks_file_paths,
    row_group_size_bytes,
    row_group_size_rows,
    max_page_size_bytes,
    max_page_size_rows,
    max_dictionary_size,
):
    plc_table, _ = table_data
    table_meta = plc.io.types.TableInputMetadata(plc_table)
    sink = plc.io.SinkInfo([io.BytesIO()])
    user_data = [{"foo": "{'bar': 'baz'}"}]
    options = (
        plc.io.parquet.ParquetWriterOptions.builder(sink, plc_table)
        .metadata(table_meta)
        .key_value_metadata(user_data)
        .compression(compression)
        .stats_level(stats_level)
        .int96_timestamps(int96_timestamps)
        .write_v2_headers(write_v2_headers)
        .dictionary_policy(dictionary_policy)
        .utc_timestamps(utc_timestamps)
        .write_arrow_schema(write_arrow_schema)
        .build()
    )
    if partitions is not None:
        options.set_partitions(partitions)
    if column_chunks_file_paths is not None:
        options.set_column_chunks_file_paths(column_chunks_file_paths)
    if row_group_size_bytes is not None:
        options.set_row_group_size_bytes(row_group_size_bytes)
    if row_group_size_rows is not None:
        options.set_row_group_size_rows(row_group_size_rows)
    if max_page_size_bytes is not None:
        options.set_max_page_size_bytes(max_page_size_bytes)
    if max_page_size_rows is not None:
        options.set_max_page_size_rows(max_page_size_rows)
    if max_dictionary_size is not None:
        options.set_max_dictionary_size(max_dictionary_size)

    result = plc.io.parquet.write_parquet(options)
    assert isinstance(result, plc.io.parquet.BufferArrayFromVector)
