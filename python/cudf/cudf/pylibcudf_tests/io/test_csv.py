# Copyright (c) 2024, NVIDIA CORPORATION.
import io

import pytest
from utils import COMPRESSION_TYPE_TO_PANDAS, assert_table_and_meta_eq

import cudf._lib.pylibcudf as plc
from cudf._lib.pylibcudf.io.types import CompressionType


# TODO: de-dupe with make_json_source
def make_csv_source(path_or_buf, pa_table, **kwargs):
    """
    Uses pandas to write a pyarrow Table to a JSON file.
    The caller is responsible for making sure that no arguments
    unsupported by pandas are passed in.
    """
    df = pa_table.to_pandas()
    mode = "w"
    if "compression" in kwargs:
        kwargs["compression"] = COMPRESSION_TYPE_TO_PANDAS[
            kwargs["compression"]
        ]
        if kwargs["compression"] is not None:
            mode = "wb"
    df.to_csv(path_or_buf, index=False, mode=mode, **kwargs)
    if isinstance(path_or_buf, io.IOBase):
        path_or_buf.seek(0)
    return path_or_buf


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


def test_read_csv_basic(
    table_data, source_or_sink, compression_type, column_names, nrows, skiprows
):
    if compression_type in {
        # Not supported by libcudf
        CompressionType.SNAPPY,
        CompressionType.XZ,
        CompressionType.ZSTD,
        # Not supported by pandas
        # TODO: find a way to test these
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

    source = make_csv_source(
        source_or_sink, pa_table, compression=compression_type
    )

    res = plc.io.csv.read_csv(
        plc.io.SourceInfo([source]),
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


# TODO: test these
# size_t byte_range_offset = 0,
# size_t byte_range_size = 0,
# str prefix = "",
# bool mangle_dupe_cols = True,
# list usecols = None,
# size_type skipfooter = 0,
# size_type header = 0,
# str lineterminator = "\n",
# str delimiter = None,
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
# list parse_hex = None,
# object dtypes = None,
# list true_values = None,
# list false_values = None,
# list na_values = None,
# bool keep_default_na = True,
# bool na_filter = True,
# bool dayfirst = False,
# DataType timestamp_type = DataType(type_id.EMPTY)
