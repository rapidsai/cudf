# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import csv
import gzip
import os
import shutil
from collections import OrderedDict
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import read_csv
from cudf.tests.utils import assert_eq


def make_numeric_dataframe(nrows, dtype):
    df = pd.DataFrame()
    df["col1"] = np.arange(nrows, dtype=dtype)
    df["col2"] = np.arange(1, 1 + nrows, dtype=dtype)
    return df


def make_datetime_dataframe(include_non_standard=False):
    df = pd.DataFrame()
    df["col1"] = np.array(
        [
            "31/10/2010",
            "05/03/2001",
            "20/10/1994",
            "18/10/1990",
            "1/1/1970",
            "2016-04-30T01:02:03.000",
            "2038-01-19 03:14:07",
        ]
    )
    df["col2"] = np.array(
        [
            "18/04/1995",
            "14 / 07 / 1994",
            "07/06/2006",
            "16/09/2005",
            "2/2/1970",
            "2007-4-30 1:6:40.000PM",
            "2038-01-19 03:14:08",
        ]
    )
    if include_non_standard:
        # Last column contains non-standard date formats
        df["col3"] = np.array(
            [
                "1 Jan",
                "2 January 1994",
                "Feb 2002",
                "31-01-2000",
                "1-1-1996",
                "15-May-2009",
                "21-Dec-3262",
            ]
        )
    return df


def make_numpy_mixed_dataframe():
    df = pd.DataFrame()
    df["Integer"] = np.array([2345, 11987, 9027, 9027])
    df["Date"] = np.array(
        ["18/04/1995", "14/07/1994", "07/06/2006", "16/09/2005"]
    )
    df["Float"] = np.array([9.001, 8.343, 6, 2.781])
    df["Integer2"] = np.array([2345, 106, 2088, 789277])
    # Category is not yet supported from libcudf
    # df["Category"] = np.array(["M", "F", "F", "F"])
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


def make_all_numeric_dataframe():
    df = pd.DataFrame()

    gdf_dtypes = [
        "float",
        "float32",
        "double",
        "float64",
        "int8",
        "short",
        "int16",
        "int",
        "int32",
        "long",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]

    np_dtypes = [
        np.float32,
        np.float32,
        np.float64,
        np.float64,
        np.int8,
        np.int16,
        np.int16,
        np.int32,
        np.int32,
        np.int64,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]

    for i in range(len(gdf_dtypes)):
        df[gdf_dtypes[i]] = np.arange(10, dtype=np_dtypes[i])

    return (
        df,
        OrderedDict(zip(gdf_dtypes, gdf_dtypes)),
        OrderedDict(zip(gdf_dtypes, np_dtypes)),
    )


def make_all_numeric_extremes_dataframe():
    # integers 0,+1,-1,min,max
    # float 0.0, -0.0,+1,-1,min,max, nan, esp, espneg, tiny, [-ve values]
    df, gdf_dtypes, pdf_dtypes = make_all_numeric_dataframe()
    df = pd.DataFrame()

    for gdf_dtype in gdf_dtypes:
        np_type = pdf_dtypes[gdf_dtype]
        if np.issubdtype(np_type, np.integer):
            itype = np.iinfo(np_type)
            extremes = [0, +1, -1, itype.min, itype.max]
            df[gdf_dtype] = np.array(extremes * 4, dtype=np_type)[:20]
        else:
            ftype = np.finfo(np_type)
            extremes = [
                0.0,
                -0.0,
                +1,
                -1,
                np.nan,
                -np.nan,
                # ftype.min, # TODO enable after fixing truncation issue #6235
                # ftype.max, # TODO enable after fixing truncation issue #6235
                np_type(np.inf),
                -np_type(np.inf),
                ftype.eps,
                ftype.epsneg,
                ftype.tiny,
                -ftype.eps,
                -ftype.epsneg,
                -ftype.tiny,
            ]
            df[gdf_dtype] = np.array(extremes * 4, dtype=np_type)[:20]
    return (
        df,
        gdf_dtypes,
        pdf_dtypes,
    )


@pytest.fixture
def path_or_buf(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_path_or_buf.csv")
    df = make_numeric_dataframe(10, np.int32)

    df.to_csv(fname, index=False, header=False)
    buffer = df.to_csv(index=False, header=False)

    def _make_path_or_buf(src):
        if src == "filepath":
            return str(fname)
        if src == "pathobj":
            return fname
        if src == "bytes_io":
            return BytesIO(buffer.encode())
        if src == "string_io":
            return StringIO(buffer)
        if src == "url":
            return Path(fname).as_uri()

        raise ValueError("Invalid source type")

    yield _make_path_or_buf


dtypes = [np.float64, np.float32, np.int64, np.int32, np.uint64, np.uint32]
dtypes_dict = {"1": np.float64, "2": np.float32, "3": np.int64, "4": np.int32}
nelem = [5, 25, 100]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("nelem", nelem)
def test_csv_reader_numeric_data(dtype, nelem, tmpdir):

    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file1.csv")

    df = make_numeric_dataframe(nelem, dtype)
    df.to_csv(fname, index=False, header=False)

    dtypes = [df[k].dtype for k in df.columns]
    out = read_csv(str(fname), names=list(df.columns.values), dtype=dtypes)

    assert len(out.columns) == len(df.columns)
    assert_eq(df, out)


@pytest.mark.parametrize("parse_dates", [["date2"], [0], ["date1", 1, "bad"]])
def test_csv_reader_datetime(parse_dates):
    df = make_datetime_dataframe(include_non_standard=True)
    buffer = df.to_csv(index=False, header=False)

    gdf = read_csv(
        StringIO(buffer),
        names=["date1", "date2", "bad"],
        parse_dates=parse_dates,
        dayfirst=True,
    )
    pdf = pd.read_csv(
        StringIO(buffer),
        names=["date1", "date2", "bad"],
        parse_dates=parse_dates,
        dayfirst=True,
    )

    assert_eq(gdf, pdf)


@pytest.mark.parametrize("pandas_arg", [{"delimiter": "|"}, {"sep": "|"}])
@pytest.mark.parametrize("cudf_arg", [{"sep": "|"}, {"delimiter": "|"}])
def test_csv_reader_mixed_data_delimiter_sep(tmpdir, pandas_arg, cudf_arg):

    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file3.csv")

    df = make_numpy_mixed_dataframe()
    df.to_csv(fname, sep="|", index=False, header=False)

    gdf1 = read_csv(
        str(fname),
        # Category is not yet supported from libcudf
        # names=["1", "2", "3", "4", "5", "6", "7"],
        # dtype=[
        #    "int64", "date", "float64", "int64", "category", "str", "bool"
        # ],
        names=["1", "2", "3", "4", "5", "6"],
        dtype=["int64", "date", "float64", "uint64", "str", "bool"],
        dayfirst=True,
        **cudf_arg,
    )
    gdf2 = read_csv(
        str(fname),
        # Category is not yet supported from libcudf
        # names=["1", "2", "3", "4", "5", "6", "7"],
        # dtype=[
        #    "int64", "date", "float64", "int64", "category", "str", "bool"
        # ],
        names=["1", "2", "3", "4", "5", "6"],
        dtype=["int64", "date", "float64", "uint64", "str", "bool"],
        dayfirst=True,
        **pandas_arg,
    )

    pdf = pd.read_csv(
        fname,
        # Category is not yet supported from libcudf
        # names=["1", "2", "3", "4", "5", "6", "7"],
        names=["1", "2", "3", "4", "5", "6"],
        parse_dates=[1],
        dayfirst=True,
        **pandas_arg,
    )

    assert len(gdf1.columns) == len(pdf.columns)
    assert len(gdf2.columns) == len(pdf.columns)
    assert_eq(gdf1, gdf2)


@pytest.mark.parametrize("use_list", [False, True])
def test_csv_reader_dtype_list(use_list):
    df = make_numeric_dataframe(10, dtype=np.float32)
    buffer = df.to_csv(index=False, header=False)

    # PANDAS doesn't list but cudf does (treated as implied ordered dict)
    # Select first column's dtype if non-list; expect the same dtype for all
    if use_list:
        dtypes = [df[k].dtype for k in df.columns]
    else:
        dtypes = df[df.columns[0]].dtype

    gdf = read_csv(StringIO(buffer), dtype=dtypes, names=df.columns)

    assert_eq(gdf, df)


@pytest.mark.parametrize("use_names", [False, True])
def test_csv_reader_dtype_dict(use_names):
    # Save with the column header if not explicitly specifying a list of names
    df, gdf_dtypes, pdf_dtypes = make_all_numeric_dataframe()
    buffer = df.to_csv(index=False, header=(not use_names))

    gdf_names = list(gdf_dtypes.keys()) if use_names else None
    pdf_names = list(pdf_dtypes.keys()) if use_names else None

    gdf = read_csv(StringIO(buffer), dtype=gdf_dtypes, names=gdf_names)
    pdf = pd.read_csv(StringIO(buffer), dtype=pdf_dtypes, names=pdf_names)

    assert_eq(gdf, pdf)


@pytest.mark.parametrize("use_names", [True])
def test_csv_reader_dtype_extremes(use_names):
    # Save with the column header if not explicitly specifying a list of names
    df, gdf_dtypes, pdf_dtypes = make_all_numeric_extremes_dataframe()
    buffer = df.to_csv(index=False, header=(not use_names))

    gdf_names = list(gdf_dtypes.keys()) if use_names else None
    pdf_names = list(pdf_dtypes.keys()) if use_names else None

    gdf = read_csv(StringIO(buffer), dtype=gdf_dtypes, names=gdf_names)
    pdf = pd.read_csv(StringIO(buffer), dtype=pdf_dtypes, names=pdf_names)

    assert_eq(gdf, pdf)


def test_csv_reader_skiprows_skipfooter(tmpdir):

    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file5.csv")

    df = make_numpy_mixed_dataframe()
    df.to_csv(
        fname, columns=["Integer", "Date", "Float"], index=False, header=False
    )

    # Using engine='python' to eliminate pandas warning of using python engine.
    df_out = pd.read_csv(
        fname,
        names=["1", "2", "3"],
        parse_dates=[1],
        dayfirst=True,
        skiprows=1,
        skipfooter=1,
        engine="python",
    )
    out = read_csv(
        str(fname),
        names=["1", "2", "3"],
        dtype=["int64", "date", "float64"],
        skiprows=1,
        skipfooter=1,
        dayfirst=True,
    )

    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)
    assert_eq(df_out, out)


def test_csv_reader_negative_vals(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file6.csv")

    names = ["0", "1", "2"]
    dtypes = ["float32", "float32", "float32"]
    lines = [
        ",".join(names),
        "-181.5060,-185.37000,-3",
        "-127.6300,-230.54600,-9",
    ]

    with open(str(fname), "w") as fp:
        fp.write("\n".join(lines))

    zero = [-181.5060, -127.6300]
    one = [-185.370, -230.54600]
    two = [-3, -9]

    df = read_csv(str(fname), names=names, dtype=dtypes, skiprows=1)

    np.testing.assert_allclose(zero, df["0"].to_array())
    np.testing.assert_allclose(one, df["1"].to_array())
    np.testing.assert_allclose(two, df["2"].to_array())


def test_csv_reader_strings(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file7.csv")

    names = ["text", "int"]
    dtypes = ["str", "int"]
    lines = [",".join(names), "a,0", "b,0", "c,0", "d,0"]

    with open(str(fname), "w") as fp:
        fp.write("\n".join(lines))

    df = read_csv(
        str(fname),
        names=names,
        dtype=dtypes,
        skiprows=1,
        decimal=".",
        thousands="'",
    )

    assert len(df.columns) == 2
    assert df["text"].dtype == np.dtype("object")
    assert df["int"].dtype == np.dtype("int32")
    assert df["text"][0] == "a"
    assert df["text"][1] == "b"
    assert df["text"][2] == "c"
    assert df["text"][3] == "d"


def test_csv_reader_strings_quotechars(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file8.csv")

    names = ["text", "int"]
    dtypes = ["str", "int"]
    lines = [",".join(names), '"a,\n",0', '"b ""c"" d",0', "e,0", '"f,,!.,",0']

    with open(str(fname), "w") as fp:
        fp.write("\n".join(lines))

    df = read_csv(
        str(fname),
        names=names,
        dtype=dtypes,
        skiprows=1,
        quotechar='"',
        quoting=1,
    )

    assert len(df.columns) == 2
    assert df["text"].dtype == np.dtype("object")
    assert df["int"].dtype == np.dtype("int32")
    assert df["text"][0] == "a,\n"
    assert df["text"][1] == 'b "c" d'
    assert df["text"][2] == "e"
    assert df["text"][3] == "f,,!.,"


def test_csv_reader_usecols_int_char(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file10.csv")
    df = make_numpy_mixed_dataframe()
    df.to_csv(
        fname,
        columns=["Integer", "Date", "Float", "Integer2"],
        index=False,
        header=False,
    )

    df_out = pd.read_csv(fname, usecols=[0, 1, 3])
    out = read_csv(fname, usecols=[0, 1, 3])

    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)
    assert_eq(df_out, out, check_names=False)


def test_csv_reader_mangle_dupe_cols(tmpdir):
    buffer = "abc,ABC,abc,abcd,abc\n1,2,3,4,5\n"

    # Default: mangle_dupe_cols=True
    pd_df = pd.read_csv(StringIO(buffer))
    cu_df = read_csv(StringIO(buffer))
    assert_eq(cu_df, pd_df)

    # Pandas does not support mangle_dupe_cols=False
    cu_df = read_csv(StringIO(buffer), mangle_dupe_cols=False)
    # check that the dupe columns were removed
    assert len(cu_df.columns) == 3
    np.testing.assert_array_equal(cu_df["abc"].to_array(), [1])


def test_csv_reader_float_decimal(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file12.csv")

    names = ["basic_32", "basic_64", "round", "decimal_only", "precision"]
    dtypes = ["float32", "float64", "float64", "float32", "float64"]
    lines = [
        ";".join(names),
        "1,2;1234,5678;12345;0,123;-73,98007199999998",
        "3,4;3456,7890;67890;,456;1,7976931348623157e+307",
        "5,6e0;0,5679e2;1,2e10;0,07e-001;0,0",
    ]

    with open(str(fname), "w") as fp:
        fp.write("\n".join(lines))

    basic_32_ref = [1.2, 3.4, 5.6]
    basic_64_ref = [1234.5678, 3456.7890, 56.79]
    round_ref = [12345, 67890, 12000000000]
    decimal_only_ref = [0.123, 0.456, 0.007]
    precision_ref = [-73.98007199999998, 1.7976931348623157e307, 0.0]

    df = read_csv(
        str(fname),
        names=names,
        dtype=dtypes,
        skiprows=1,
        delimiter=";",
        decimal=",",
    )

    np.testing.assert_allclose(basic_32_ref, df["basic_32"].to_array())
    np.testing.assert_allclose(basic_64_ref, df["basic_64"].to_array())
    np.testing.assert_allclose(round_ref, df["round"].to_array())
    np.testing.assert_allclose(decimal_only_ref, df["decimal_only"].to_array())
    np.testing.assert_allclose(precision_ref, df["precision"].to_array())


def test_csv_reader_NaN_values():

    names = dtypes = ["float32"]
    empty_cells = '\n""\n  \n "" \n'
    default_na_cells = (
        "#N/A\n#N/A N/A\n#NA\n-1.#IND\n"
        "-1.#QNAN\n-NaN\n-nan\n1.#IND\n"
        "1.#QNAN\nN/A\nNA\nNULL\n"
        "NaN\nn/a\nnan\nnull\n"
    )
    custom_na_cells = "NV_NAN\nNotANumber\n"
    all_cells = empty_cells + default_na_cells + custom_na_cells
    custom_na_values = ["NV_NAN", "NotANumber"]

    # test default NA values. empty cells should also yield NaNs
    all_nan = read_csv(
        StringIO(default_na_cells + empty_cells), names=names, dtype=dtypes
    )
    assert all(np.isnan(all_nan.to_pandas()["float32"]))

    # custom NA values
    all_nan = read_csv(
        StringIO(all_cells),
        names=names,
        dtype=dtypes,
        na_values=custom_na_values,
    )
    assert all(np.isnan(all_nan.to_pandas()["float32"]))

    # data type detection should evaluate the column to int8 (all nulls)
    df_int8 = read_csv(
        StringIO(default_na_cells + custom_na_cells),
        header=None,
        na_values=custom_na_values,
    )
    assert df_int8.dtypes[0] == "int8"
    assert all(df_int8["0"][idx] is None for idx in range(len(df_int8["0"])))

    # data type detection should evaluate the column to object;
    # for data type detection, cells need to be completely empty,
    # but some cells in empty_cells contain blank characters and quotes
    df_obj = read_csv(
        StringIO(all_cells), header=None, na_values=custom_na_values
    )
    assert df_obj.dtypes[0] == np.dtype("object")


def test_csv_reader_thousands(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file13.csv")

    names = dtypes = [
        "float32",
        "float64",
        "int32",
        "int64",
        "uint32",
        "uint64",
    ]
    lines = [
        ",".join(names),
        "1'234.5, 1'234.567, 1'234'567, 1'234'567'890,\
                1'234'567, 1'234'567'890",
        "12'345.6, 123'456.7, 12'345, 123'456'789, 12'345, 123'456'789",
    ]

    with open(str(fname), "w") as fp:
        fp.write("\n".join(lines))

    f32_ref = [1234.5, 12345.6]
    f64_ref = [1234.567, 123456.7]
    int32_ref = [1234567, 12345]
    int64_ref = [1234567890, 123456789]
    uint32_ref = [1234567, 12345]
    uint64_ref = [1234567890, 123456789]

    df = read_csv(
        str(fname), names=names, dtype=dtypes, skiprows=1, thousands="'"
    )

    np.testing.assert_allclose(f32_ref, df["float32"].to_array())
    np.testing.assert_allclose(f64_ref, df["float64"].to_array())
    np.testing.assert_allclose(int32_ref, df["int32"].to_array())
    np.testing.assert_allclose(int64_ref, df["int64"].to_array())
    np.testing.assert_allclose(uint32_ref, df["uint32"].to_array())
    np.testing.assert_allclose(uint64_ref, df["uint64"].to_array())


def test_csv_reader_buffer_strings():

    names = ["text", "int"]
    dtypes = ["str", "int"]
    lines = [",".join(names), "a,0", "b,0", "c,0", "d,0"]

    buffer = "\n".join(lines)

    df = read_csv(StringIO(buffer), names=names, dtype=dtypes, skiprows=1)
    assert len(df.columns) == 2
    assert df["text"].dtype == np.dtype("object")
    assert df["int"].dtype == np.dtype("int32")
    assert df["text"][0] == "a"
    assert df["text"][1] == "b"
    assert df["text"][2] == "c"
    assert df["text"][3] == "d"

    df2 = read_csv(
        BytesIO(str.encode(buffer)), names=names, dtype=dtypes, skiprows=1
    )
    assert len(df2.columns) == 2
    assert df2["text"].dtype == np.dtype("object")
    assert df2["int"].dtype == np.dtype("int32")
    assert df2["text"][0] == "a"
    assert df2["text"][1] == "b"
    assert df2["text"][2] == "c"
    assert df2["text"][3] == "d"


@pytest.mark.parametrize(
    "ext, out_comp, in_comp",
    [
        (".geez", "gzip", "gzip"),
        (".beez", "bz2", "bz2"),
        (".gz", "gzip", "infer"),
        (".bz2", "bz2", "infer"),
        (".data", None, "infer"),
        (".txt", None, None),
        ("", None, None),
    ],
)
def test_csv_reader_compression(tmpdir, ext, out_comp, in_comp):

    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_compression" + ext)

    df = make_numpy_mixed_dataframe()
    df.to_csv(fname, index=False, header=False, compression=out_comp)

    gdf = read_csv(fname, names=list(df.columns.values), compression=in_comp)
    pdf = pd.read_csv(
        fname, names=list(df.columns.values), compression=in_comp
    )

    assert_eq(gdf, pdf)


@pytest.mark.parametrize(
    "names, dtypes, data, trues, falses",
    [
        (
            ["A", "B"],
            ["bool", "bool"],
            "True,True\nFalse,False\nTrue,False",
            None,
            None,
        ),
        (
            ["A", "B"],
            ["int32", "int32"],
            "True,1\nFalse,2\nTrue,3",
            None,
            None,
        ),
        (
            ["A", "B"],
            ["int32", "int32"],
            "YES,1\nno,2\nyes,3\nNo,4\nYes,5",
            ["yes", "Yes", "YES"],
            ["no", "NO", "No"],
        ),
        (["A", "B"], ["int32", "int32"], "foo,bar\nbar,foo", ["foo"], ["bar"]),
        (["x", "y"], None, "True,1\nFalse,0", None, None),
    ],
)
def test_csv_reader_bools(tmpdir, names, dtypes, data, trues, falses):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file11.csv")

    lines = [",".join(names), data]

    with open(str(fname), "w") as fp:
        fp.write("\n".join(lines))

    # Usage of true_values and false_values makes that column into bool type
    df_out = pd.read_csv(
        fname,
        names=names,
        skiprows=1,
        dtype=(dtypes[0] if dtypes else None),
        true_values=trues,
        false_values=falses,
    )

    out = read_csv(
        fname,
        names=names,
        dtype=dtypes,
        skiprows=1,
        true_values=trues,
        false_values=falses,
    )

    assert_eq(df_out, out)


def test_csv_quotednumbers(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file12.csv")

    names = ["integer", "decimal"]
    dtypes = ["int32", "float32"]
    lines = [
        ",".join(names),
        '1,"3.14"',
        '"2","300"',
        '"3",10101.0101',
        '4,"6.28318"',
    ]

    with open(str(fname), "w") as fp:
        fp.write("\n".join(lines))

    integer_ref = [1, 2, 3, 4]
    decimal_ref = [3.14, 300, 10101.0101, 6.28318]

    df1 = read_csv(str(fname), names=names, dtype=dtypes, skiprows=1)
    df2 = read_csv(str(fname), names=names, dtype=dtypes, skiprows=1)

    assert len(df2.columns) == 2
    np.testing.assert_allclose(integer_ref, df1["integer"].to_array())
    np.testing.assert_allclose(decimal_ref, df1["decimal"].to_array())
    np.testing.assert_allclose(integer_ref, df2["integer"].to_array())
    np.testing.assert_allclose(decimal_ref, df2["decimal"].to_array())


def test_csv_reader_nrows(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file14.csv")

    names = ["int1", "int2"]
    dtypes = ["int32", "int32"]

    rows = 4000000
    read_rows = (rows * 3) // 4
    skip_rows = (rows - read_rows) // 2
    sample_skip = 1000

    with open(str(fname), "w") as fp:
        fp.write(",".join(names) + "\n")
        for i in range(rows):
            fp.write(str(i) + ", " + str(2 * i) + " \n")

    # with specified names
    df = read_csv(
        str(fname),
        names=names,
        dtype=dtypes,
        skiprows=skip_rows + 1,
        nrows=read_rows,
    )
    assert df.shape == (read_rows, 2)
    for row in range(0, read_rows // sample_skip, sample_skip):
        assert df["int1"][row] == row + skip_rows
        assert df["int2"][row] == 2 * (row + skip_rows)
    assert df["int2"][read_rows - 1] == 2 * (read_rows - 1 + skip_rows)

    # with column name inference
    df = read_csv(
        str(fname), dtype=dtypes, skiprows=skip_rows + 1, nrows=read_rows
    )
    assert df.shape == (read_rows, 2)
    assert str(skip_rows) in list(df)[0]
    assert str(2 * skip_rows) in list(df)[1]
    for row in range(0, read_rows // sample_skip, sample_skip):
        assert df[list(df)[0]][row] == row + skip_rows + 1
        assert df[list(df)[1]][row] == 2 * (row + skip_rows + 1)
    assert df[list(df)[1]][read_rows - 1] == 2 * (read_rows + skip_rows)

    # nrows larger than the file
    df = read_csv(str(fname), dtype=dtypes, nrows=rows * 2)
    assert df.shape == (rows, 2)
    for row in range(0, rows // sample_skip, sample_skip):
        assert df["int1"][row] == row
        assert df["int2"][row] == 2 * row
    assert df["int2"][rows - 1] == 2 * (rows - 1)

    # nrows + skiprows larger than the file
    df = read_csv(
        str(fname), dtype=dtypes, nrows=read_rows, skiprows=read_rows
    )
    assert df.shape == (rows - read_rows, 2)

    # nrows equal to zero
    df = read_csv(str(fname), dtype=dtypes, nrows=0)
    assert df.shape == (0, 2)

    # with both skipfooter and nrows - should throw
    with pytest.raises(ValueError):
        read_csv(str(fname), nrows=read_rows, skipfooter=1)


def test_csv_reader_gzip_compression_strings(tmpdir):
    fnamebase = tmpdir.mkdir("gdf_csv")
    fname = fnamebase.join("tmp_csvreader_file15.csv")
    fnamez = fnamebase.join("tmp_csvreader_file15.csv.gz")

    names = ["text", "int"]
    dtypes = ["str", "int"]
    lines = [",".join(names), "a,0", "b,0", "c,0", "d,0"]

    with open(str(fname), "w") as fp:
        fp.write("\n".join(lines))

    with open(str(fname), "rb") as f_in, gzip.open(str(fnamez), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    df = read_csv(
        str(fnamez),
        names=names,
        dtype=dtypes,
        skiprows=1,
        decimal=".",
        thousands="'",
        compression="gzip",
    )

    assert len(df.columns) == 2
    assert df["text"].dtype == np.dtype("object")
    assert df["int"].dtype == np.dtype("int32")
    assert df["text"][0] == "a"
    assert df["text"][1] == "b"
    assert df["text"][2] == "c"
    assert df["text"][3] == "d"


@pytest.mark.parametrize("skip_rows", [0, 2, 4])
@pytest.mark.parametrize("header_row", [0, 2])
def test_csv_reader_skiprows_header(skip_rows, header_row):

    names = ["float_point", "integer"]
    dtypes = ["float64", "int64"]
    lines = [
        ",".join(names),
        "1.2, 1",
        "2.3, 2",
        "3.4, 3",
        "4.5, 4",
        "5.6, 5",
        "6.7, 6",
    ]
    buffer = "\n".join(lines)

    cu_df = read_csv(
        StringIO(buffer), dtype=dtypes, skiprows=skip_rows, header=header_row
    )
    pd_df = pd.read_csv(
        StringIO(buffer), skiprows=skip_rows, header=header_row
    )

    assert cu_df.shape == pd_df.shape
    assert list(cu_df.columns.values) == list(pd_df.columns.values)


def test_csv_reader_dtype_inference():
    names = ["float_point", "integer"]
    lines = [
        ",".join(names),
        "1.2,1",
        "2.3,2",
        "3.4,3",
        "4.5,4",
        "5.6,5",
        "6.7,6",
    ]
    buffer = "\n".join(lines)
    cu_df = read_csv(StringIO(buffer))
    pd_df = pd.read_csv(StringIO(buffer))

    assert cu_df.shape == pd_df.shape
    assert list(cu_df.columns.values) == list(pd_df.columns.values)


def test_csv_reader_dtype_inference_whitespace():
    names = ["float_point", "integer"]
    lines = [
        ",".join(names),
        "  1.2,    1",
        "2.3,2    ",
        "  3.4,   3",
        " 4.5,4",
        "5.6,  5",
        " 6.7,6 ",
    ]
    buffer = "\n".join(lines)
    cu_df = read_csv(StringIO(buffer))
    pd_df = pd.read_csv(StringIO(buffer))

    assert cu_df.shape == pd_df.shape
    assert list(cu_df.columns.values) == list(pd_df.columns.values)


def test_csv_reader_empty_dataframe():

    dtypes = ["float64", "int64"]
    buffer = "float_point, integer"

    # should work fine with dtypes
    df = read_csv(StringIO(buffer), dtype=dtypes)
    assert df.shape == (0, 2)
    assert all(df.dtypes == ["float64", "int64"])

    # should default to string columns without dtypes
    df = read_csv(StringIO(buffer))
    assert df.shape == (0, 2)
    assert all(df.dtypes == ["object", "object"])


def test_csv_reader_filenotfound(tmpdir):
    fname = "non-existing-filename.csv"

    # should raise an error
    with pytest.raises(FileNotFoundError):
        read_csv(str(fname))

    # should raise an error
    dname = tmpdir.mkdir("gdf_csv")
    with pytest.raises(FileNotFoundError):
        read_csv(str(dname))


@pytest.mark.parametrize(
    "src", ["filepath", "pathobj", "bytes_io", "string_io", "url"]
)
def test_csv_reader_filepath_or_buffer(tmpdir, path_or_buf, src):
    expect = pd.read_csv(path_or_buf("filepath"))
    got = cudf.read_csv(path_or_buf(src))

    assert_eq(expect, got)


def test_csv_reader_carriage_return(tmpdir):
    rows = 1000
    names = ["int_row", "int_double_row"]

    buffer = ",".join(names) + "\r\n"
    for row in range(rows):
        buffer += str(row) + ", " + str(2 * row) + "\r\n"

    df = read_csv(StringIO(buffer))

    assert len(df) == rows
    for row in range(0, rows):
        assert df[names[0]][row] == row
        assert df[names[1]][row] == 2 * row


def test_csv_reader_tabs():
    names = ["float_point", "integer", "date"]
    lines = [
        ",".join(names),
        "1.2,\t12,     \t11/22/1995",
        "3.4\t,\t34\t,\t 01/01/2001",
        "\t 5.6,56 \t, 12/12/1970",
        "\t7.8 , 78\t,06/15/2018 \t",
    ]
    buffer = "\n".join(lines)

    df = read_csv(StringIO(buffer), parse_dates=["date"])

    assert df.shape == (4, 3)

    floats = [1.2, 3.4, 5.6, 7.8]
    ints = [12, 34, 56, 78]
    dates = [
        "1995-11-22T00:00:00.000000000",
        "2001-01-01T00:00:00.000000000",
        "1970-12-12T00:00:00.000000000",
        "2018-06-15T00:00:00.000000000",
    ]
    np.testing.assert_allclose(floats, df["float_point"].to_array())
    np.testing.assert_allclose(ints, df["integer"].to_array())
    for row in range(4):
        assert str(df["date"][row]) == dates[row]


@pytest.mark.parametrize("segment_bytes", [10000, 19999, 30001, 36000])
def test_csv_reader_byte_range(tmpdir, segment_bytes):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file16.csv")

    names = ["int1", "int2"]

    rows = 10000
    with open(str(fname), "w") as fp:
        for i in range(rows):
            fp.write(str(i) + ", " + str(2 * i) + " \n")
    file_size = os.stat(str(fname)).st_size

    ref_df = read_csv(str(fname), names=names).to_pandas()

    dfs = []
    for segment in range((file_size + segment_bytes - 1) // segment_bytes):
        dfs.append(
            read_csv(
                str(fname),
                names=names,
                byte_range=(segment * segment_bytes, segment_bytes),
            )
        )
    df = cudf.concat(dfs).to_pandas()

    assert list(df["int1"]) == list(ref_df["int1"])
    assert list(df["int2"]) == list(ref_df["int2"])


def test_csv_reader_byte_range_type_corner_case(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file17.csv")

    cudf.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-02",
        dtypes={"name": str, "id": int, "x": float, "y": float},
    ).to_csv(fname, chunksize=100000)

    byte_range = (2_147_483_648, 0)
    with pytest.raises(RuntimeError, match="Offset is past end of file"):
        cudf.read_csv(fname, byte_range=byte_range, header=None)


@pytest.mark.parametrize("segment_bytes", [10, 19, 31, 36])
def test_csv_reader_byte_range_strings(segment_bytes):
    names = ["strings"]
    buffer = "\n".join('"' + str(x) + '"' for x in range(1, 100))
    file_size = len(buffer)

    ref_df = read_csv(StringIO(buffer), names=names).to_pandas()

    dfs = []
    for segment in range((file_size + segment_bytes - 1) // segment_bytes):
        dfs.append(
            read_csv(
                StringIO(buffer),
                names=names,
                byte_range=(segment * segment_bytes, segment_bytes),
            )
        )
    df = cudf.concat(dfs).to_pandas()

    assert list(df["strings"]) == list(ref_df["strings"])


@pytest.mark.parametrize(
    "header_row, skip_rows, skip_blanks",
    [
        (1, 0, True),
        ("infer", 2, True),
        (1, 4, True),
        (3, 0, False),
        ("infer", 5, False),
    ],
)
@pytest.mark.parametrize("line_terminator", ["\n", "\r\n"])
def test_csv_reader_blanks_and_comments(
    skip_rows, header_row, skip_blanks, line_terminator
):

    lines = [
        "# first comment line",
        line_terminator,
        "# third comment line",
        "1,2,3",
        "4,5,6",
        "7,8,9",
        line_terminator,
        "# last comment line",
        line_terminator,
        "1,1,1",
    ]
    buffer = line_terminator.join(lines)

    cu_df = read_csv(
        StringIO(buffer),
        comment="#",
        header=header_row,
        skiprows=skip_rows,
        skip_blank_lines=skip_blanks,
    )
    pd_df = pd.read_csv(
        StringIO(buffer),
        comment="#",
        header=header_row,
        skiprows=skip_rows,
        skip_blank_lines=skip_blanks,
    )

    assert cu_df.shape == pd_df.shape
    assert list(cu_df.columns.values) == list(pd_df.columns.values)


def test_csv_reader_prefix():

    lines = ["1, 1, 1, 1"]
    buffer = "\n".join(lines)

    prefix_str = "a_prefix"
    df = read_csv(StringIO(buffer), header=None, prefix=prefix_str)

    column_names = list(df.columns.values)
    for col in range(len(column_names)):
        assert column_names[col] == prefix_str + str(col)


def test_csv_reader_delim_whitespace():
    buffer = "1    2  3\n4  5 6"

    # with header row
    cu_df = read_csv(StringIO(buffer), delim_whitespace=True)
    pd_df = pd.read_csv(StringIO(buffer), delim_whitespace=True)
    assert_eq(pd_df, cu_df)

    # without header row
    cu_df = read_csv(StringIO(buffer), delim_whitespace=True, header=None)
    pd_df = pd.read_csv(StringIO(buffer), delim_whitespace=True, header=None)
    assert pd_df.shape == cu_df.shape

    # should raise an error if used with delimiter or sep
    with pytest.raises(ValueError):
        read_csv(StringIO(buffer), delim_whitespace=True, delimiter=" ")
    with pytest.raises(ValueError):
        read_csv(StringIO(buffer), delim_whitespace=True, sep=" ")


def test_csv_reader_unnamed_cols():
    # first and last columns are unnamed
    buffer = ",1,2,3,\n4,5,6,7,8"

    cu_df = read_csv(StringIO(buffer))
    pd_df = pd.read_csv(StringIO(buffer))

    assert all(pd_df.columns == cu_df.columns)
    assert pd_df.shape == cu_df.shape


def test_csv_reader_header_quotation():
    buffer = '"1,,1","2,\n,2",3\n+4,+5,+6'

    cu_df = read_csv(StringIO(buffer))
    pd_df = pd.read_csv(StringIO(buffer))
    assert cu_df.shape == (1, 3)
    assert_eq(pd_df, cu_df)

    # test cases that fail with pandas
    buffer_pd_fail = '"1,one," , ",2,two" ,3\n4,5,6'
    cu_df = read_csv(StringIO(buffer_pd_fail))
    assert cu_df.shape == (1, 3)


def test_csv_reader_oversized_byte_range():
    buffer = "a,b,c,d,e\n4,5,6,7,8"

    cu_df = read_csv(StringIO(buffer), byte_range=(0, 1024))
    pd_df = pd.read_csv(StringIO(buffer))

    assert all(pd_df.columns == cu_df.columns)
    assert pd_df.shape == cu_df.shape


def test_csv_reader_index_col():
    buffer = "0,1,2\n3,4,5\n6,7,8"
    names = ["int1", "int2", "int3"]

    # using a column name
    cu_df = read_csv(StringIO(buffer), names=names, index_col="int1")
    pd_df = pd.read_csv(StringIO(buffer), names=names, index_col="int1")
    assert_eq(pd_df, cu_df)

    # using a column index
    cu_df = read_csv(StringIO(buffer), header=None, index_col=0)
    pd_df = pd.read_csv(StringIO(buffer), header=None, index_col=0)
    assert_eq(cu_df.index, pd_df.index)

    # using a column index with names
    cu_df = read_csv(StringIO(buffer), header=None, index_col=0, names=names)
    pd_df = pd.read_csv(
        StringIO(buffer), header=None, index_col=0, names=names
    )
    assert_eq(cu_df.index, pd_df.index)

    # passing False to avoid using a column as index (no-op in cuDF)
    cu_df = read_csv(StringIO(buffer), header=None, index_col=False)
    pd_df = pd.read_csv(StringIO(buffer), header=None, index_col=False)
    assert_eq(cu_df.index, pd_df.index)


@pytest.mark.parametrize(
    "names", [["a", "b", "c"], [416, 905, 647], range(3), None]
)
def test_csv_reader_column_names(names):
    buffer = "0,1,2\n3,4,5\n6,7,8"

    df = read_csv(StringIO(buffer), names=names)
    if names is None:
        assert list(df) == ["0", "1", "2"]
    else:
        assert list(df) == list(names)


def test_csv_reader_bools_false_positives(tmpdir):
    # values that are equal to ["True", "TRUE", "False", "FALSE"]
    # when using ints to detect bool values
    items = [3977, 4329, 24015, 27567]

    buffer = "\n".join(str(i) for i in items)

    df = read_csv(StringIO(buffer), header=None, dtype=["int32"])

    np.testing.assert_array_equal(items, df["0"].to_array())


def test_csv_reader_aligned_byte_range(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file19.csv")
    nelem = 1000

    input_df = pd.DataFrame(
        {"key": np.arange(0, nelem), "zeros": np.zeros(nelem)}
    )
    input_df.to_csv(fname)

    df = cudf.read_csv(str(fname), byte_range=(0, 4096))
    # read_csv call above used to crash; the assert below is not crucial
    assert np.count_nonzero(df["zeros"].to_pandas().values) == 0


@pytest.mark.parametrize(
    "pdf_dtype, gdf_dtype",
    [(None, None), ("int", "hex"), ("int32", "hex32"), ("int64", "hex64")],
)
def test_csv_reader_hexadecimals(pdf_dtype, gdf_dtype):
    lines = ["0x0", "-0x1000", "0xfedcba", "0xABCDEF", "0xaBcDeF", "9512c20b"]
    values = [int(hex_int, 16) for hex_int in lines]

    buffer = "\n".join(lines)

    if gdf_dtype is not None:
        # require explicit `hex` dtype to parse hexadecimals
        pdf = pd.DataFrame(data=values, dtype=pdf_dtype, columns=["hex_int"])
        gdf = read_csv(StringIO(buffer), dtype=[gdf_dtype], names=["hex_int"])
        np.testing.assert_array_equal(
            pdf["hex_int"], gdf["hex_int"].to_array()
        )
    else:
        # otherwise, dtype inference returns as object (string)
        pdf = pd.read_csv(StringIO(buffer), names=["hex_int"])
        gdf = read_csv(StringIO(buffer), names=["hex_int"])
        assert_eq(pdf, gdf)


@pytest.mark.parametrize("quoting", [0, 1, 2, 3])
def test_csv_reader_pd_consistent_quotes(quoting):
    names = ["text"]
    dtypes = ["str"]
    lines = ['"a"', '"b ""c"" d"', '"f!\n."']

    buffer = "\n".join(lines)

    gd_df = read_csv(
        StringIO(buffer), names=names, dtype=dtypes, quoting=quoting
    )
    pd_df = pd.read_csv(StringIO(buffer), names=names, quoting=quoting)

    assert_eq(pd_df, gd_df)


def test_read_csv_names_header_combination():
    pdf = pd.DataFrame(
        {
            "firstname": ["Emma", "Ava", "Sophia"],
            "lastname": ["Olivia", "Isabella", "Charlotte"],
            "gender": ["F", "F", "F"],
        }
    )
    buffer = pdf.to_csv(header=True, index=False)
    names = pdf.columns

    gdf = read_csv(StringIO(buffer), names=names, header=0)
    assert_eq(pdf, gdf)

    gdf = read_csv(StringIO(buffer), header=0)
    assert_eq(pdf, gdf)

    gdf = read_csv(StringIO(buffer))
    assert_eq(pdf, gdf)


def test_csv_reader_scientific_type_detection():
    buffer = """1.,1.1,-1.1,1E1,1e1,-1e1,-1e-1,1e-1,1.1e1,1.1e-1,-1.1e-1,-1.1e1
                +1.1,1E+1,1e+1,+1e1,+1e-1,1e-1,+1.1e1,1.1e+1,+1.1e+1,+1.1e1"""
    expected = [
        1.0,
        1.1,
        -1.1,
        10.0,
        10.0,
        -10,
        -0.1,
        0.1,
        11,
        0.11,
        -0.11,
        -11,
        1.1,
        10.0,
        10.0,
        10,
        0.1,
        0.1,
        11,
        11,
        11,
        11,
    ]

    df = read_csv(StringIO(buffer), header=None)

    for dt in df.dtypes:
        assert dt == "float64"
    for col in df:
        assert np.isclose(df[col][0], expected[int(col)])


@pytest.mark.parametrize("line_terminator", ["\n", "\r\n"])
def test_csv_blank_first_row(line_terminator):

    lines = ["colA,colB", "", "1, 1.1", "2, 2.2"]
    buffer = line_terminator.join(lines)

    cu_df = read_csv(StringIO(buffer))

    assert cu_df.shape == (2, 2)
    assert all(cu_df.columns == ["colA", "colB"])


@pytest.mark.parametrize("contents", ["", "\n"])
def test_csv_empty_file(tmpdir, contents):
    fname = tmpdir.mkdir("gdf_csv").join("test_csv_empty_file.csv")
    with open(fname, "w") as f:
        f.write(contents)

    col_names = ["col1", "col2", "col3", "col4"]
    in_dtypes = ["int", "str", "float", "short"]
    out_dtypes = ["int32", "object", "float32", "int16"]

    # Empty dataframe if no columns names specified or inferred
    df = read_csv(str(fname))
    assert len(df.columns) == 0

    # No row dataframe if columns names are specified or inferred
    df = read_csv(str(fname), dtype=in_dtypes, names=col_names)
    assert all(df.columns == col_names)
    assert list(df.dtypes) == out_dtypes


@pytest.mark.parametrize("contents", ["", "\n"])
def test_csv_empty_buffer(tmpdir, contents):
    col_names = ["col1", "col2", "col3", "col4"]
    in_dtypes = ["int", "str", "float", "short"]
    out_dtypes = ["int32", "object", "float32", "int16"]

    # Empty dataframe if no columns names specified or inferred
    df = read_csv(StringIO(contents))
    assert len(df.columns) == 0

    # No row dataframe if columns names are specified or inferred
    df = read_csv(StringIO(contents), dtype=in_dtypes, names=col_names)
    assert all(df.columns == col_names)
    assert list(df.dtypes) == out_dtypes


@pytest.mark.parametrize(
    "dtype", [["short", "float", "int"], {"A": "short", "C": "int"}]
)
def test_csv_reader_partial_dtype(dtype):
    names_df = read_csv(
        StringIO("0,1,2"),
        names=["A", "B", "C"],
        dtype=dtype,
        usecols=["A", "C"],
    )
    header_df = read_csv(
        StringIO('"A","B","C"\n0,1,2'), dtype=dtype, usecols=["A", "C"]
    )

    assert names_df == header_df
    assert all(names_df.dtypes == ["int16", "int32"])


def test_csv_writer_file_handle(tmpdir):

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["xxx", "yyyy", "zzzzz"]})
    gdf = cudf.from_pandas(df)

    gdf_df_fname = tmpdir.join("gdf_df_1.csv")
    with open(gdf_df_fname, "w") as f:
        gdf.to_csv(path_or_buf=f, index=False)
    assert os.path.exists(gdf_df_fname)

    gdf2 = pd.read_csv(gdf_df_fname)
    assert_eq(gdf, gdf2)


def test_csv_writer_file_append(tmpdir):

    gdf1 = cudf.DataFrame({"a": [1, 2, 3], "b": ["xxx", "yyyy", "zzzzz"]})
    gdf2 = cudf.DataFrame({"a": [4, 5, 6], "b": ["foo", "bar", "baz"]})

    gdf_df_fname = tmpdir.join("gdf_df_append.csv")
    with open(gdf_df_fname, "w") as f:
        gdf1.to_csv(f, index=False)
    with open(gdf_df_fname, "a") as f:
        gdf2.to_csv(f, header=False, index=False)

    result = cudf.read_csv(gdf_df_fname)
    expected = cudf.concat([gdf1, gdf2], ignore_index=True)
    assert_eq(result, expected)


def test_csv_writer_buffer(tmpdir):

    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": ["xxx", "yyyy", "zzzzz"]})

    buffer = BytesIO()
    gdf.to_csv(buffer, index=False)

    result = cudf.read_csv(buffer)
    assert_eq(result, gdf)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("nelem", nelem)
def test_csv_writer_numeric_data(dtype, nelem, tmpdir):

    pdf_df_fname = tmpdir.join("pdf_df_1.csv")
    gdf_df_fname = tmpdir.join("gdf_df_1.csv")

    df = make_numeric_dataframe(nelem, dtype)
    gdf = cudf.from_pandas(df)
    df.to_csv(path_or_buf=pdf_df_fname, index=False, line_terminator="\n")
    gdf.to_csv(path_or_buf=gdf_df_fname, index=False)

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect = pd.read_csv(pdf_df_fname)
    got = pd.read_csv(gdf_df_fname)
    assert_eq(expect, got)


def test_csv_writer_datetime_data(tmpdir):
    pdf_df_fname = tmpdir.join("pdf_df_2.csv")
    gdf_df_fname = tmpdir.join("gdf_df_2.csv")

    df = make_datetime_dataframe()
    gdf = cudf.from_pandas(df)
    df.to_csv(path_or_buf=pdf_df_fname, index=False, line_terminator="\n")
    gdf.to_csv(path_or_buf=gdf_df_fname, index=False)

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect = pd.read_csv(pdf_df_fname)
    got = pd.read_csv(gdf_df_fname)
    assert_eq(expect, got)


@pytest.mark.parametrize("sep", [",", "|", " ", ";"])
@pytest.mark.parametrize(
    "columns",
    [
        # Category is not yet supported from libcudf
        # ["Integer", "Date", "Float", "Integer2", "Category"],
        ["Integer", "Date", "Float", "Integer2"],
        # ["Category", "Date", "Float"],
        ["Date", "Float"],
        ["Integer2"],
        # ["Category", "Integer2", "Float", "Date", "Integer"],
        ["Integer2", "Float", "Date", "Integer"],
        [
            # "Category",
            "Integer2",
            "Float",
            "Date",
            "Integer",
            "String",
            "Boolean",
        ],
        None,
    ],
)
@pytest.mark.parametrize(
    "header", [True, False, np.bool_(True), np.bool_(False)]
)
@pytest.mark.parametrize(
    "index", [True, False, np.bool_(True), np.bool_(False)]
)
@pytest.mark.parametrize("line_terminator", ["\r", "\n", "NEWLINE", "<<<<<"])
def test_csv_writer_mixed_data(
    sep, columns, header, index, line_terminator, tmpdir
):
    pdf_df_fname = tmpdir.join("pdf_df_3.csv")
    gdf_df_fname = tmpdir.join("gdf_df_3.csv")

    df = make_numpy_mixed_dataframe()
    df["Date"] = df["Date"].astype("datetime64")
    gdf = cudf.from_pandas(df)
    df.to_csv(
        path_or_buf=pdf_df_fname,
        index=index,
        sep=sep,
        columns=columns,
        header=header,
        line_terminator=line_terminator,
        date_format="%Y-%m-%dT%H:%M:%SZ",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    gdf.to_csv(
        path_or_buf=gdf_df_fname,
        index=index,
        sep=sep,
        columns=columns,
        header=header,
        line_terminator=line_terminator,
    )

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect = pd.read_csv(pdf_df_fname, quoting=csv.QUOTE_NONE, escapechar="\\")
    got = pd.read_csv(gdf_df_fname)
    assert_eq(expect, got)


def test_csv_writer_multiindex(tmpdir):
    pdf_df_fname = tmpdir.join("pdf_df_3.csv")
    gdf_df_fname = tmpdir.join("gdf_df_3.csv")

    gdf = cudf.DataFrame(
        {
            "a": np.random.randint(0, 5, 20),
            "b": np.random.randint(0, 5, 20),
            "c": range(20),
            "d": np.random.random(20),
        }
    )
    gdg = gdf.groupby(["a", "b"]).mean()
    pdg = gdg.to_pandas()
    pdg.to_csv(pdf_df_fname)
    gdg.to_csv(gdf_df_fname)

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect = pd.read_csv(pdf_df_fname)
    got = pd.read_csv(gdf_df_fname)
    assert_eq(expect, got)


@pytest.mark.parametrize("chunksize", [None, 9, 1000])
def test_csv_writer_chunksize(chunksize, tmpdir):
    pdf_df_fname = tmpdir.join("pdf_df_4.csv")
    gdf_df_fname = tmpdir.join("gdf_df_4.csv")

    pdf = make_numpy_mixed_dataframe()
    pdf["Date"] = pdf["Date"].astype("datetime64")
    # Increase the df len as chunked logic only gets applied from chunksize >=8
    pdf = pd.concat([pdf] * 5)
    gdf = cudf.from_pandas(pdf)

    pdf.to_csv(
        pdf_df_fname, date_format="%Y-%m-%dT%H:%M:%SZ", chunksize=chunksize,
    )
    gdf.to_csv(gdf_df_fname, chunksize=chunksize)

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect = pd.read_csv(pdf_df_fname)
    got = pd.read_csv(gdf_df_fname)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "df",
    [
        cudf.DataFrame({"vals": [1, 2, 3]}),
        cudf.DataFrame(
            {"vals1": [1, 2, 3], "vals2": ["hello", "rapids", "cudf"]}
        ),
        cudf.DataFrame(
            {"vals1": [None, 2.0, 3.0], "vals2": ["hello", "rapids", None]}
        ),
    ],
)
def test_to_csv_empty_filename(df):
    pdf = df.to_pandas()

    actual = df.to_csv()
    expected = pdf.to_csv()

    assert actual == expected


@pytest.mark.parametrize(
    "df",
    [
        cudf.DataFrame({"vals": [1, 2, 3]}),
        cudf.DataFrame(
            {"vals1": [1, 2, 3], "vals2": ["hello", "rapids", "cudf"]}
        ),
        cudf.DataFrame(
            {"vals1": [None, 2.0, 3.0], "vals2": ["hello", "rapids", None]}
        ),
    ],
)
def test_to_csv_StringIO(df):
    cudf_io = StringIO()
    pandas_io = StringIO()

    pdf = df.to_pandas()

    df.to_csv(cudf_io)
    pdf.to_csv(pandas_io)

    cudf_io.seek(0)
    pandas_io.seek(0)

    assert cudf_io.read() == pandas_io.read()


def test_csv_writer_empty_dataframe(tmpdir):

    df_fname = tmpdir.join("gdf_df_5.csv")
    gdf = cudf.DataFrame({"float_point": [], "integer": []})
    gdf["float_point"] = gdf["float_point"].astype("float")
    gdf["integer"] = gdf["integer"].astype("int")

    gdf.to_csv(df_fname, index=False)

    df = cudf.read_csv(df_fname)

    assert df.shape == (0, 2)
    assert all(df.dtypes == ["object", "object"])


def test_csv_write_chunksize_corner_case(tmpdir):
    # With this num of rows and chunksize
    # libcudf splits table such a way that it
    # will end up creating an empty table slice
    # which caused the issue 5588.
    df_fname = tmpdir.join("gdf_df_17.csv")
    df = cudf.DataFrame({"a": np.arange(10_000)})
    df.to_csv(df_fname, chunksize=1000, index=False)
    got = cudf.read_csv(df_fname)

    assert_eq(df, got)
