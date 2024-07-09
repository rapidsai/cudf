# Copyright (c) 2024, NVIDIA CORPORATION.
import io

import pandas as pd
import pyarrow as pa
import pytest
from utils import (
    _convert_numeric_types_to_floating,
    assert_table_and_meta_eq,
    make_source,
)

import cudf._lib.pylibcudf as plc
from cudf._lib.pylibcudf.io.types import CompressionType

# Shared kwargs to pass to make_source
_COMMON_CSV_SOURCE_KWARGS = {
    "format": "csv",
    "index": False,
}


@pytest.fixture(params=[True, False])
def column_names(table_data, request):
    """
    Parametrized fixture returning column names (or None).
    Useful for testing col_names argument in read_csv
    """
    if request.param:
        _, pa_table = table_data
        return pa_table.column_names
    return None


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_read_csv_basic(
    table_data,
    source_or_sink,
    compression_type,
    column_names,
    nrows,
    skiprows,
    delimiter,
):
    if compression_type in {
        # Not supported by libcudf
        CompressionType.XZ,
        CompressionType.ZSTD,
        # Not supported by pandas
        # TODO: find a way to test these
        CompressionType.SNAPPY,
        CompressionType.BROTLI,
        CompressionType.LZ4,
        CompressionType.LZO,
        CompressionType.ZLIB,
    }:
        pytest.skip("unsupported compression type by pandas/libcudf")

    _, pa_table = table_data

    # can't compress non-binary data with pandas
    if isinstance(source_or_sink, io.StringIO):
        compression_type = CompressionType.NONE

    if len(pa_table) > 0:
        # Drop the string column for now, since it contains ints
        # (which won't roundtrip since they are not quoted by python csv writer)
        # also uint64 will get confused for int64
        pa_table = pa_table.drop_columns(
            [
                "col_string",
                "col_uint64",
                # Nested types don't work by default
                "col_list<item: int64>",
                "col_list<item: list<item: int64>>",
                "col_struct<v: int64 not null>",
                "col_struct<a: int64 not null, b_struct: struct<b: double not null> not null>",
            ]
        )
        if column_names is not None:
            column_names = pa_table.column_names

    source = make_source(
        source_or_sink,
        pa_table,
        compression=compression_type,
        sep=delimiter,
        **_COMMON_CSV_SOURCE_KWARGS,
    )

    res = plc.io.csv.read_csv(
        plc.io.SourceInfo([source]),
        delimiter=delimiter,
        compression=compression_type,
        col_names=column_names,
        nrows=nrows,
        skiprows=skiprows,
    )

    # Adjust table for nrows/skiprows
    pa_table = pa_table.slice(
        offset=skiprows, length=nrows if nrows != -1 else None
    )

    assert_table_and_meta_eq(
        pa_table,
        res,
        check_types=False if len(pa_table) == 0 else True,
        check_names=False if skiprows > 0 and column_names is None else True,
    )


# Note: make sure chunk size is big enough so that dtype inference
# infers correctly
@pytest.mark.parametrize("chunk_size", [4204, 6000])
def test_read_csv_byte_range(table_data, chunk_size):
    _, pa_table = table_data
    if len(pa_table) == 0:
        # pandas writes nothing when we have empty table
        # and header=None
        pytest.skip("Don't test empty table case")
    source = io.BytesIO()
    source = make_source(
        source, pa_table, header=False, **_COMMON_CSV_SOURCE_KWARGS
    )
    tbls_w_meta = []
    for chunk_start in range(0, len(source.getbuffer()), chunk_size):
        tbls_w_meta.append(
            plc.io.csv.read_csv(
                plc.io.SourceInfo([source]),
                byte_range_offset=chunk_start,
                byte_range_size=chunk_start + chunk_size,
                header=-1,
                col_names=pa_table.column_names,
            )
        )
    if isinstance(source, io.IOBase):
        source.seek(0)
    exp = pd.read_csv(source, names=pa_table.column_names, header=None)
    tbls = []
    for tbl_w_meta in tbls_w_meta:
        if tbl_w_meta.tbl.num_rows() > 0:
            tbls.append(plc.interop.to_arrow(tbl_w_meta.tbl))
    full_tbl = pa.concat_tables(tbls)

    full_tbl_plc = plc.io.TableWithMetadata(
        plc.interop.from_arrow(full_tbl),
        tbls_w_meta[0].column_names(include_children=True),
    )
    assert_table_and_meta_eq(pa.Table.from_pandas(exp), full_tbl_plc)


def test_read_csv_dtypes(table_data, source_or_sink):
    # Simple test for dtypes where we read in
    # all numeric data as floats
    _, pa_table = table_data

    # Drop the string column for now, since it contains ints
    # (which won't roundtrip since they are not quoted by python csv writer)
    # also uint64 will get confused for int64
    pa_table = pa_table.drop_columns(
        [
            "col_string",
            "col_uint64",
            # Nested types don't work by default
            "col_list<item: int64>",
            "col_list<item: list<item: int64>>",
            "col_struct<v: int64 not null>",
            "col_struct<a: int64 not null, b_struct: struct<b: double not null> not null>",
        ]
    )
    source = make_source(
        source_or_sink,
        pa_table,
        **_COMMON_CSV_SOURCE_KWARGS,
    )

    dtypes, new_fields = _convert_numeric_types_to_floating(pa_table)
    # Extract the dtype out of the (name, type, child_types) tuple
    # (read_csv doesn't support this format since it doesn't support nested columns)
    dtypes = [dtype for _, dtype, _ in dtypes]

    new_schema = pa.schema(new_fields)

    res = plc.io.csv.read_csv(plc.io.SourceInfo([source]), dtypes=dtypes)
    new_table = pa_table.cast(new_schema)

    assert_table_and_meta_eq(new_table, res)


# TODO: test these
# str prefix = "",
# bool mangle_dupe_cols = True,
# list usecols = None,
# size_type skipfooter = 0,
# size_type header = 0,
# str lineterminator = "\n",
# str thousands = None,
# str decimal = ".",
# str comment = None,
# bool delim_whitespace = False,
# bool skipinitialspace = False,
# bool skip_blank_lines = True,
# quote_style quoting = quote_style.MINIMAL,
# str quotechar = '"',
# bool doublequote = True,
# bool detect_whitespace_around_quotes = False,
# list parse_dates = None,
# object dtypes = None,
# list true_values = None,
# list false_values = None,
# list na_values = None,
# bool keep_default_na = True,
# bool na_filter = True,
# bool dayfirst = False,
