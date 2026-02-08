# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import io

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from pyarrow.parquet import read_table
from utils import (
    assert_table_and_meta_eq,
    get_bytes_from_source,
    make_source,
    synchronize_stream,
)

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.stream import Stream

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


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("column_names", [None, ["col_int64", "col_bool"]])
@pytest.mark.parametrize("column_indices", [None, [2, 0]])
@pytest.mark.parametrize("source_strategy", ["inline", "set_source"])
def test_read_parquet_basic(
    table_data,
    binary_source_or_sink,
    nrows_skiprows,
    column_names,
    column_indices,
    stream,
    source_strategy,
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    source_info = plc.io.SourceInfo([source])
    options = plc.io.parquet.ParquetReaderOptions.builder(
        source_info if source_strategy == "inline" else plc.io.SourceInfo([])
    ).build()

    if source_strategy == "set_source":
        options.set_source(source_info)

    if nrows > -1:
        options.set_num_rows(nrows)
    if skiprows != 0:
        options.set_skip_rows(skiprows)
    if column_names is not None:
        options.set_column_names(column_names)
    elif column_indices is not None:
        options.set_column_indices(column_indices)

    res = plc.io.parquet.read_parquet(options, stream)

    if column_names is not None:
        pa_table = pa_table.select(column_names)
    elif column_indices is not None:
        column_names = [pa_table.column_names[idx] for idx in column_indices]
        pa_table = pa_table.select(column_names)

    # Adapt to nrows/skiprows
    pa_table = pa_table.slice(
        offset=skiprows, length=nrows if nrows > -1 else None
    )

    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)

    # No filtering done
    assert res.num_row_groups_after_stats_filter is None
    assert res.num_row_groups_after_bloom_filter is None


@pytest.mark.parametrize("if_prune_rowgroup,result", [(True, 0), (False, 1)])
def test_read_parquet_filters_metadata(tmp_path, if_prune_rowgroup, result):
    col_list = list(range(1, 10))
    min_element = min(col_list)
    max_element = max(col_list)
    tbl1 = pa.Table.from_pydict({"a": col_list})
    path1 = tmp_path / "tbl1.parquet"
    pa.parquet.write_table(tbl1, path1)
    source = plc.io.SourceInfo([path1])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    if if_prune_rowgroup:
        # Prune the only row group since the filter aims to find elements larger than the max
        filter = Operation(
            ASTOperator.GREATER,
            ColumnNameReference("a"),
            Literal(plc.Scalar.from_arrow(pa.scalar(max_element))),
        )
    else:
        # No real pruning
        filter = Operation(
            ASTOperator.GREATER,
            ColumnNameReference("a"),
            Literal(plc.Scalar.from_arrow(pa.scalar(min_element))),
        )
    options.set_filter(filter)
    plc_table_w_meta = plc.io.parquet.read_parquet(options)
    assert (
        plc_table_w_meta.num_input_row_groups == 1
    )  # Input has only one rowgroup
    assert plc_table_w_meta.num_row_groups_after_stats_filter == result


@pytest.mark.parametrize(
    "pa_filters,plc_filters",
    [
        (
            pc.field("col_int64") >= 10,
            Operation(
                ASTOperator.GREATER_EQUAL,
                ColumnNameReference("col_int64"),
                Literal(plc.Scalar.from_arrow(pa.scalar(10))),
            ),
        ),
        (
            (pc.field("col_int64") >= 10) & (pc.field("col_double") < 0),
            Operation(
                ASTOperator.LOGICAL_AND,
                Operation(
                    ASTOperator.GREATER_EQUAL,
                    ColumnNameReference("col_int64"),
                    Literal(plc.Scalar.from_arrow(pa.scalar(10))),
                ),
                Operation(
                    ASTOperator.LESS,
                    ColumnNameReference("col_double"),
                    Literal(plc.Scalar.from_arrow(pa.scalar(0.0))),
                ),
            ),
        ),
        (
            (pc.field(0) == 10),
            Operation(
                ASTOperator.EQUAL,
                ColumnReference(0),
                Literal(plc.Scalar.from_arrow(pa.scalar(10))),
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

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([source])
    ).build()
    options.set_filter(plc_filters)

    plc_table_w_meta = plc.io.parquet.read_parquet(options)
    exp = read_table(source, filters=pa_filters)
    assert_table_and_meta_eq(
        exp, plc_table_w_meta, check_field_nullability=False
    )


class FooSpan:
    def __init__(self, owner):
        # Keep the owning object alive
        self._data = owner

    @property
    def ptr(self):
        return self._data.ptr

    @property
    def size(self):
        return self._data.size


@pytest.mark.parametrize("num_buffers", [1, 2])
@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("column_names", [None, ["col_int64", "col_bool"]])
@pytest.mark.parametrize("column_indices", [None, [2, 0]])
@pytest.mark.parametrize("use_foo_span", [False, True])
def test_read_parquet_from_device_buffers(
    table_data,
    binary_source_or_sink,
    nrows_skiprows,
    stream,
    column_names,
    column_indices,
    num_buffers,
    use_foo_span,
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    # Load data from source
    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    rmm_buf = DeviceBuffer.to_device(
        get_bytes_from_source(source), plc.utils._get_stream(stream)
    )
    buf = FooSpan(rmm_buf) if use_foo_span else rmm_buf

    synchronize_stream(stream)

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([buf] * num_buffers)
    ).build()
    if nrows > -1:
        options.set_num_rows(nrows)
    if skiprows != 0:
        options.set_skip_rows(skiprows)
    if column_names is not None:
        options.set_column_names(column_names)
    elif column_indices is not None:
        options.set_column_indices(column_indices)

    res = plc.io.parquet.read_parquet(options, stream)

    expected = (
        pa_table
        if num_buffers == 1
        else pa.concat_tables([pa_table] * num_buffers)
    )
    if column_names is not None:
        expected = expected.select(column_names)
    elif column_indices is not None:
        column_names = [expected.column_names[idx] for idx in column_indices]
        expected = expected.select(column_names)

    expected = expected.slice(skiprows, nrows if nrows > -1 else None)

    assert_table_and_meta_eq(expected, res, check_field_nullability=False)


# TODO: Test these options
# list row_groups = None,
# ^^^ This one is not tested since it's not in pyarrow/pandas, deprecate?
# bool convert_strings_to_categories = False,
# bool use_pandas_metadata = True


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("write_v2_headers", [True, False])
@pytest.mark.parametrize("utc_timestamps", [True, False])
@pytest.mark.parametrize("write_arrow_schema", [True, False])
@pytest.mark.parametrize(
    "partitions",
    [None, [plc.io.types.PartitionInfo(0, 10)]],
)
@pytest.mark.parametrize("column_chunks_file_paths", [None, ["tmp.parquet"]])
@pytest.mark.parametrize("row_group_size_bytes", [None, 1024])
@pytest.mark.parametrize("row_group_size_rows", [None, 1])
@pytest.mark.parametrize("max_page_size_bytes", [None, 1024])
@pytest.mark.parametrize("max_page_size_rows", [None, 1])
@pytest.mark.parametrize("max_dictionary_size", [None, 100])
def test_write_parquet(
    table_data,
    write_v2_headers,
    utc_timestamps,
    write_arrow_schema,
    partitions,
    column_chunks_file_paths,
    row_group_size_bytes,
    row_group_size_rows,
    max_page_size_bytes,
    max_page_size_rows,
    max_dictionary_size,
    stream,
):
    _, pa_table = table_data
    if len(pa_table) == 0 and partitions is not None:
        pytest.skip("https://github.com/rapidsai/cudf/issues/17361")
    plc_table = plc.Table.from_arrow(pa_table)
    table_meta = plc.io.types.TableInputMetadata(plc_table)
    sink = plc.io.SinkInfo([io.BytesIO()])
    user_data = [{"foo": "{'bar': 'baz'}"}]
    compression = plc.io.types.CompressionType.SNAPPY
    stats_level = plc.io.types.StatisticsFreq.STATISTICS_COLUMN
    dictionary_policy = plc.io.types.DictionaryPolicy.ADAPTIVE
    options = (
        plc.io.parquet.ParquetWriterOptions.builder(sink, plc_table)
        .metadata(table_meta)
        .key_value_metadata(user_data)
        .compression(compression)
        .stats_level(stats_level)
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

    result = plc.io.parquet.write_parquet(options, stream)

    synchronize_stream(stream)

    assert isinstance(result, memoryview)


@pytest.mark.parametrize("use_jit_filter", [False, True])
@pytest.mark.parametrize(
    "pa_filter,plc_filter",
    [
        (
            pc.field("col_int64") >= 10,
            Operation(
                ASTOperator.GREATER_EQUAL,
                ColumnNameReference("col_int64"),
                Literal(plc.Scalar.from_arrow(pa.scalar(10, type=pa.int64()))),
            ),
        ),
        (
            pc.field("col_str") == "foo",
            Operation(
                ASTOperator.EQUAL,
                ColumnNameReference("col_str"),
                Literal(
                    plc.Scalar.from_arrow(pa.scalar("foo", type=pa.string()))
                ),
            ),
        ),
    ],
)
def test_read_parquet_filters_jit(
    binary_source_or_sink,
    pa_filter,
    plc_filter,
    use_jit_filter,
):
    pa_table = pa.table(
        {
            "col_int64": pa.array([6, 0, 2, 2], type=pa.int64()),
            "col_str": pa.array(
                ["bar", "foo", "baz", "foo"], type=pa.string()
            ),
        }
    )

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    options = (
        plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo([source])
        )
        .use_jit_filter(use_jit_filter)
        .build()
    )
    options.set_filter(plc_filter)

    assert options.is_enabled_use_jit_filter() is use_jit_filter

    got = plc.io.parquet.read_parquet(options)
    expect = read_table(source, filters=pa_filter)

    assert_table_and_meta_eq(
        expect,
        got,
        check_field_nullability=False,
    )
