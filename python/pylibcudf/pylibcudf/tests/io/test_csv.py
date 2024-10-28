# Copyright (c) 2024, NVIDIA CORPORATION.
import io
import os
from io import StringIO

import pandas as pd
import pyarrow as pa
import pytest
from utils import (
    _convert_types,
    assert_table_and_meta_eq,
    make_source,
    write_source_str,
)

import pylibcudf as plc
from pylibcudf.io.types import CompressionType

# Shared kwargs to pass to make_source
_COMMON_CSV_SOURCE_KWARGS = {
    "format": "csv",
    "index": False,
}


@pytest.fixture(scope="module")
def csv_table_data(table_data):
    """
    Like the table_data but with nested types dropped
    since the CSV reader can't handle that
    uint64 is also dropped since it can get confused with int64
    """
    _, pa_table = table_data
    pa_table = pa_table.drop_columns(
        [
            "col_uint64",
            "col_list<item: int64>",
            "col_list<item: list<item: int64>>",
            "col_struct<v: int64 not null>",
            "col_struct<a: int64 not null, b_struct: struct<b: double not null> not null>",
        ]
    )
    return plc.interop.from_arrow(pa_table), pa_table


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_read_csv_basic(
    csv_table_data,
    source_or_sink,
    text_compression_type,
    nrows_skiprows,
    delimiter,
):
    _, pa_table = csv_table_data
    compression_type = text_compression_type
    nrows, skiprows = nrows_skiprows

    # can't compress non-binary data with pandas
    if isinstance(source_or_sink, io.StringIO):
        compression_type = CompressionType.NONE

    source = make_source(
        source_or_sink,
        pa_table,
        compression=compression_type,
        sep=delimiter,
        **_COMMON_CSV_SOURCE_KWARGS,
    )

    # Rename the table (by reversing the names) to test names argument
    pa_table = pa_table.rename_columns(pa_table.column_names[::-1])
    column_names = pa_table.column_names

    # Adapt to nrows/skiprows
    pa_table = pa_table.slice(
        offset=skiprows, length=nrows if nrows != -1 else None
    )

    res = plc.io.csv.read_csv(
        plc.io.SourceInfo([source]),
        delimiter=delimiter,
        compression=compression_type,
        col_names=column_names,
        nrows=nrows,
        skiprows=skiprows,
    )

    assert_table_and_meta_eq(
        pa_table,
        res,
        check_types_if_empty=False,
        check_names=False if skiprows > 0 and column_names is None else True,
    )


# Note: make sure chunk size is big enough so that dtype inference
# infers correctly
@pytest.mark.parametrize("chunk_size", [1000, 5999])
def test_read_csv_byte_range(table_data, chunk_size, tmp_path):
    _, pa_table = table_data
    if len(pa_table) == 0:
        # pandas writes nothing when we have empty table
        # and header=None
        pytest.skip("Don't test empty table case")
    source = f"{tmp_path}/a.csv"
    source = make_source(
        source, pa_table, header=False, **_COMMON_CSV_SOURCE_KWARGS
    )
    file_size = os.stat(source).st_size
    tbls_w_meta = []
    for segment in range((file_size + chunk_size - 1) // chunk_size):
        tbls_w_meta.append(
            plc.io.csv.read_csv(
                plc.io.SourceInfo([source]),
                byte_range_offset=segment * chunk_size,
                byte_range_size=chunk_size,
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


@pytest.mark.parametrize("usecols", [None, ["col_int64", "col_bool"], [0, 1]])
def test_read_csv_dtypes(csv_table_data, source_or_sink, usecols):
    # Simple test for dtypes where we read in
    # all numeric data as floats
    _, pa_table = csv_table_data

    source = make_source(
        source_or_sink,
        pa_table,
        **_COMMON_CSV_SOURCE_KWARGS,
    )
    # Adjust table for usecols
    if usecols is not None:
        pa_table = pa_table.select(usecols)

    dtypes, new_fields = _convert_types(
        pa_table,
        lambda t: (pa.types.is_unsigned_integer(t) or pa.types.is_integer(t)),
        pa.float64(),
    )
    # Extract the dtype out of the (name, type, child_types) tuple
    # (read_csv doesn't support this format since it doesn't support nested columns)
    dtypes = {name: dtype for name, dtype, _ in dtypes}

    new_schema = pa.schema(new_fields)

    res = plc.io.csv.read_csv(
        plc.io.SourceInfo([source]), dtypes=dtypes, usecols=usecols
    )
    new_table = pa_table.cast(new_schema)

    assert_table_and_meta_eq(new_table, res)


@pytest.mark.parametrize("skip_blanks", [True, False])
@pytest.mark.parametrize("decimal, quotechar", [(".", "'"), ("_", '"')])
@pytest.mark.parametrize("lineterminator", ["\n", "\r\n"])
def test_read_csv_parse_options(
    source_or_sink, decimal, quotechar, skip_blanks, lineterminator
):
    lines = [
        "# first comment line",
        "# third comment line",
        "1,2,3,4_4,'z'",
        '4,5,6,5_5,""',
        "7,8,9,9_87,'123'",
        "# last comment line",
        "1,1,1,10_11,abc",
    ]
    buffer = lineterminator.join(lines)

    write_source_str(source_or_sink, buffer)

    plc_table_w_meta = plc.io.csv.read_csv(
        plc.io.SourceInfo([source_or_sink]),
        comment="#",
        decimal=decimal,
        skip_blank_lines=skip_blanks,
        quotechar=quotechar,
    )
    df = pd.read_csv(
        StringIO(buffer),
        comment="#",
        decimal=decimal,
        skip_blank_lines=skip_blanks,
        quotechar=quotechar,
    )
    assert_table_and_meta_eq(pa.Table.from_pandas(df), plc_table_w_meta)


@pytest.mark.parametrize("na_filter", [True, False])
@pytest.mark.parametrize("na_values", [["n/a"], ["NV_NAN"]])
@pytest.mark.parametrize("keep_default_na", [True, False])
def test_read_csv_na_values(
    source_or_sink, na_filter, na_values, keep_default_na
):
    lines = ["a,b,c", "n/a,NaN,NV_NAN", "1.0,2.0,3.0"]
    buffer = "\n".join(lines)

    write_source_str(source_or_sink, buffer)

    plc_table_w_meta = plc.io.csv.read_csv(
        plc.io.SourceInfo([source_or_sink]),
        na_filter=na_filter,
        na_values=na_values if na_filter else None,
        keep_default_na=keep_default_na,
    )
    df = pd.read_csv(
        StringIO(buffer),
        na_filter=na_filter,
        na_values=na_values if na_filter else None,
        keep_default_na=keep_default_na,
    )
    assert_table_and_meta_eq(pa.Table.from_pandas(df), plc_table_w_meta)


@pytest.mark.parametrize("header", [0, 10, -1])
def test_read_csv_header(csv_table_data, source_or_sink, header):
    _, pa_table = csv_table_data

    source = make_source(
        source_or_sink,
        pa_table,
        **_COMMON_CSV_SOURCE_KWARGS,
    )

    plc_table_w_meta = plc.io.csv.read_csv(
        plc.io.SourceInfo([source]), header=header
    )
    if header > 0:
        if header < len(pa_table):
            names_row = pa_table.take([header - 1]).to_pylist()[0].values()
            pa_table = pa_table.slice(header)
            col_names = [str(name) for name in names_row]
            pa_table = pa_table.rename_columns(col_names)
        else:
            pa_table = pa.table([])
    elif header < 0:
        # neg header means use user-provided names (in this case nothing)
        # (the original column names are now data)
        tbl_dict = pa_table.to_pydict()
        new_tbl_dict = {}
        for i, (name, vals) in enumerate(tbl_dict.items()):
            str_vals = [str(val) for val in vals]
            new_tbl_dict[str(i)] = [name] + str_vals
        pa_table = pa.table(new_tbl_dict)

    assert_table_and_meta_eq(
        pa_table,
        plc_table_w_meta,
        check_types_if_empty=False,
    )


# TODO: test these
# str prefix = "",
# bool mangle_dupe_cols = True,
# size_type skipfooter = 0,
# str thousands = None,
# bool delim_whitespace = False,
# bool skipinitialspace = False,
# quote_style quoting = quote_style.MINIMAL,
# bool doublequote = True,
# bool detect_whitespace_around_quotes = False,
# list parse_dates = None,
# list true_values = None,
# list false_values = None,
# bool dayfirst = False,
