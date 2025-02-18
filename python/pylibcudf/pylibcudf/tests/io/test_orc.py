# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import _convert_types, assert_table_and_meta_eq, make_source

import pylibcudf as plc

# Shared kwargs to pass to make_source
_COMMON_ORC_SOURCE_KWARGS = {"format": "orc"}


@pytest.mark.parametrize("columns", [None, ["col_int64", "col_bool"]])
def test_read_orc_basic(
    table_data, binary_source_or_sink, nrows_skiprows, columns
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    # ORC reader doesn't support skip_rows for nested columns
    if skiprows > 0:
        colnames_to_drop = []
        for i in range(len(pa_table.schema)):
            field = pa_table.schema.field(i)

            if pa.types.is_nested(field.type):
                colnames_to_drop.append(field.name)
        pa_table = pa_table.drop(colnames_to_drop)
    # ORC doesn't support unsigned ints
    # let's cast to int64
    _, new_fields = _convert_types(
        pa_table, pa.types.is_unsigned_integer, pa.int64()
    )
    pa_table = pa_table.cast(pa.schema(new_fields))

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_ORC_SOURCE_KWARGS
    )

    options = plc.io.orc.OrcReaderOptions.builder(
        plc.io.types.SourceInfo([source])
    ).build()
    if nrows >= 0:
        options.set_num_rows(nrows)
    if skiprows >= 0:
        options.set_skip_rows(skiprows)
    if columns is not None and len(columns) > 0:
        options.set_columns(columns)

    res = plc.io.orc.read_orc(options)

    if columns is not None:
        pa_table = pa_table.select(columns)

    # Adapt to nrows/skiprows
    pa_table = pa_table.slice(
        offset=skiprows, length=nrows if nrows != -1 else None
    )

    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)


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
):
    pa_table = pa.table({"a": [1.0, 2.0, None], "b": [True, None, False]})
    plc_table = plc.interop.from_arrow(pa_table)

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

    plc.io.orc.write_orc(options)

    read_table = pa.orc.read_table(str(tmpfile_name))

    res = plc.io.types.TableWithMetadata(
        plc.interop.from_arrow(read_table),
        [(name, []) for name in pa_table.schema.names],
    )

    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)
