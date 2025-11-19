# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import (
    _convert_types,
    assert_table_and_meta_eq,
    get_bytes_from_source,
    make_source,
)

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.stream import Stream

import pylibcudf as plc

_COMMON_ORC_SOURCE_KWARGS = {"format": "orc"}


def _cast_unsigned_to_int64(table):
    _, new_fields = _convert_types(
        table, pa.types.is_unsigned_integer, pa.int64()
    )
    return table.cast(pa.schema(new_fields))


def _drop_nested_columns_if_skipping(table, skiprows):
    if skiprows <= 0:
        return table
    cols_to_drop = [
        field.name for field in table.schema if pa.types.is_nested(field.type)
    ]
    return table.drop(cols_to_drop)


def _build_orc_reader_options(
    source_info, nrows=None, skiprows=None, columns=None
):
    options = plc.io.orc.OrcReaderOptions.builder(source_info).build()
    if nrows is not None and nrows >= 0:
        options.set_num_rows(nrows)
    if skiprows is not None and skiprows >= 0:
        options.set_skip_rows(skiprows)
    if columns:
        options.set_columns(columns)
    return options


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("columns", [None, ["col_int64", "col_bool"]])
@pytest.mark.parametrize("source_strategy", ["inline", "set_source"])
def test_read_orc_basic(
    table_data,
    binary_source_or_sink,
    nrows_skiprows,
    columns,
    stream,
    source_strategy,
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    pa_table = _drop_nested_columns_if_skipping(pa_table, skiprows)
    pa_table = _cast_unsigned_to_int64(pa_table)

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_ORC_SOURCE_KWARGS
    )
    source_info = plc.io.types.SourceInfo([source])
    options = _build_orc_reader_options(
        source_info
        if source_strategy == "inline"
        else plc.io.types.SourceInfo([]),
        nrows=nrows,
        skiprows=skiprows,
        columns=columns,
    )

    if source_strategy == "set_source":
        options.set_source(source_info)

    res = plc.io.orc.read_orc(options, stream)

    if columns is not None:
        pa_table = pa_table.select(columns)

    pa_table = pa_table.slice(
        offset=skiprows, length=nrows if nrows != -1 else None
    )

    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)


@pytest.mark.parametrize("num_buffers", [1, 2])
@pytest.mark.parametrize("stream", [None, Stream()])
def test_read_orc_from_device_buffers(
    table_data, binary_source_or_sink, num_buffers, stream
):
    _, pa_table = table_data

    pa_table = _cast_unsigned_to_int64(pa_table)

    source = make_source(binary_source_or_sink, pa_table, format="orc")

    buf = DeviceBuffer.to_device(get_bytes_from_source(source))

    options = plc.io.orc.OrcReaderOptions.builder(
        plc.io.types.SourceInfo([buf] * num_buffers)
    ).build()

    result = plc.io.orc.read_orc(options, stream)

    expected = (
        pa_table
        if num_buffers == 1
        else pa.concat_tables([pa_table] * num_buffers)
    )
    res = plc.io.types.TableWithMetadata(
        result.tbl,
        [(name, []) for name in expected.column_names],
    )

    assert_table_and_meta_eq(expected, res, check_field_nullability=False)


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize(
    "compression",
    [
        plc.io.types.CompressionType.NONE,
        plc.io.types.CompressionType.SNAPPY,
    ],
)
@pytest.mark.parametrize(
    "statistics",
    [
        plc.io.types.StatisticsFreq.STATISTICS_NONE,
        plc.io.types.StatisticsFreq.STATISTICS_COLUMN,
    ],
)
@pytest.mark.parametrize("stripe_size_bytes", [None, 65536])
@pytest.mark.parametrize("stripe_size_rows", [None, 512])
@pytest.mark.parametrize("row_index_stride", [None, 512])
def test_roundtrip_pa_table(
    compression,
    statistics,
    stripe_size_bytes,
    stripe_size_rows,
    row_index_stride,
    tmp_path,
    stream,
):
    pa_table = pa.table({"a": [1.0, 2.0, None], "b": [True, None, False]})
    plc_table = plc.Table.from_arrow(pa_table)

    tmpfile_name = tmp_path / "test.orc"

    sink = plc.io.SinkInfo([str(tmpfile_name)])

    tbl_meta = plc.io.types.TableInputMetadata(plc_table)
    user_data = {"a": "", "b": ""}
    options = (
        plc.io.orc.OrcWriterOptions.builder(sink, plc_table)
        .metadata(tbl_meta)
        .key_value_metadata(user_data)
        .compression(compression)
        .enable_statistics(statistics)
        .build()
    )
    if stripe_size_bytes is not None:
        options.set_stripe_size_bytes(stripe_size_bytes)
    if stripe_size_rows is not None:
        options.set_stripe_size_rows(stripe_size_rows)
    if row_index_stride is not None:
        options.set_row_index_stride(row_index_stride)

    plc.io.orc.write_orc(options, stream)

    read_table = pa.orc.read_table(str(tmpfile_name))

    res = plc.io.types.TableWithMetadata(
        plc.Table.from_arrow(read_table),
        [(name, []) for name in pa_table.schema.names],
    )

    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)
