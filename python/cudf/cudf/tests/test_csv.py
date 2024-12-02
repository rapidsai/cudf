# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import codecs
import gzip
import os
import re
import shutil
from collections import OrderedDict
from io import BytesIO, StringIO
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import read_csv
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


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
    df["Category"] = np.array(["M", "F", "F", "F"])
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


@pytest.fixture
def pd_mixed_dataframe():
    return make_numpy_mixed_dataframe()


@pytest.fixture
def cudf_mixed_dataframe():
    return cudf.from_pandas(make_numpy_mixed_dataframe())


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
            df[gdf_dtype] = np.array(extremes * 4).astype(np_type)[:20]
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
def pandas_extreme_numeric_dataframe():
    return make_all_numeric_extremes_dataframe()[0]


@pytest.fixture
def cudf_extreme_numeric_dataframe(pandas_extreme_numeric_dataframe):
    return cudf.from_pandas(pandas_extreme_numeric_dataframe)


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
    # Need to used `date_format='mixed'`,
    # https://github.com/pandas-dev/pandas/issues/53355
    pdf = pd.read_csv(
        StringIO(buffer),
        names=["date1", "date2", "bad"],
        parse_dates=parse_dates,
        dayfirst=True,
        date_format="mixed",
    )

    assert_eq(gdf, pdf)


@pytest.mark.parametrize("pandas_arg", [{"delimiter": "|"}, {"sep": "|"}])
@pytest.mark.parametrize("cudf_arg", [{"sep": "|"}, {"delimiter": "|"}])
def test_csv_reader_mixed_data_delimiter_sep(
    tmpdir, pandas_arg, cudf_arg, pd_mixed_dataframe
):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file3.csv")

    pd_mixed_dataframe.to_csv(fname, sep="|", index=False, header=False)

    gdf1 = read_csv(
        str(fname),
        names=["1", "2", "3", "4", "5", "6", "7"],
        dtype=[
            "int64",
            "datetime64[ns]",
            "float64",
            "int64",
            "category",
            "str",
            "bool",
        ],
        dayfirst=True,
        **cudf_arg,
    )
    gdf2 = read_csv(
        str(fname),
        names=["1", "2", "3", "4", "5", "6", "7"],
        dtype=[
            "int64",
            "datetime64[ns]",
            "float64",
            "int64",
            "category",
            "str",
            "bool",
        ],
        dayfirst=True,
        **pandas_arg,
    )

    pdf = pd.read_csv(
        fname,
        names=["1", "2", "3", "4", "5", "6", "7"],
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
    dtypes = df.dtypes.to_dict()
    gdf_names = list(gdf_dtypes.keys()) if use_names else None
    pdf_names = list(pdf_dtypes.keys()) if use_names else None
    gdf = read_csv(StringIO(buffer), dtype=dtypes, names=gdf_names)
    pdf = pd.read_csv(StringIO(buffer), dtype=dtypes, names=pdf_names)

    assert_eq(gdf, pdf)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
@pytest.mark.parametrize("use_names", [True, False])
def test_csv_reader_dtype_extremes(use_names):
    # Save with the column header if not explicitly specifying a list of names
    df, gdf_dtypes, pdf_dtypes = make_all_numeric_extremes_dataframe()
    buffer = df.to_csv(index=False, header=(not use_names))
    dtypes = df.dtypes.to_dict()
    gdf_names = list(gdf_dtypes.keys()) if use_names else None
    pdf_names = list(pdf_dtypes.keys()) if use_names else None

    gdf = read_csv(StringIO(buffer), dtype=dtypes, names=gdf_names)
    pdf = pd.read_csv(StringIO(buffer), dtype=dtypes, names=pdf_names)

    assert_eq(gdf, pdf)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="https://github.com/pandas-dev/pandas/issues/52449",
)
def test_csv_reader_skiprows_skipfooter(tmpdir, pd_mixed_dataframe):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file5.csv")

    pd_mixed_dataframe.to_csv(
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
        dtype=["int64", "datetime64[ns]", "float64"],
        skiprows=1,
        skipfooter=1,
        dayfirst=True,
    )

    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)

    assert_eq(df_out, out, check_dtype=False)


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

    np.testing.assert_allclose(zero, df["0"].to_numpy())
    np.testing.assert_allclose(one, df["1"].to_numpy())
    np.testing.assert_allclose(two, df["2"].to_numpy())


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
    assert df["int"].dtype == np.dtype("int64")
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
    assert df["int"].dtype == np.dtype("int64")
    assert df["text"][0] == "a,\n"
    assert df["text"][1] == 'b "c" d'
    assert df["text"][2] == "e"
    assert df["text"][3] == "f,,!.,"


def test_csv_reader_usecols_int_char(tmpdir, pd_mixed_dataframe):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file10.csv")
    pd_mixed_dataframe.to_csv(
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


@pytest.mark.parametrize(
    "buffer",
    [
        "abc,ABC,abc,abcd,abc\n1,2,3,4,5\n",
        "A,A,A.1,A,A.2,A,A.4,A,A\n1,2,3.1,4,a.2,a,a.4,a,a",
        "A,A,A.1,,Unnamed: 4,A,A.4,A,A\n1,2,3.1,4,a.2,a,a.4,a,a",
    ],
)
@pytest.mark.parametrize("mangle_dupe_cols", [True, False])
def test_csv_reader_mangle_dupe_cols(tmpdir, buffer, mangle_dupe_cols):
    # Default: mangle_dupe_cols=True
    cu_df = read_csv(StringIO(buffer), mangle_dupe_cols=mangle_dupe_cols)
    if mangle_dupe_cols:
        pd_df = pd.read_csv(StringIO(buffer))
    else:
        # Pandas does not support mangle_dupe_cols=False
        head = buffer.split("\n")[0].split(",")
        first_cols = np.unique(head, return_index=True)[1]
        pd_df = pd.read_csv(StringIO(buffer), usecols=first_cols)
    assert_eq(cu_df, pd_df)


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

    np.testing.assert_allclose(basic_32_ref, df["basic_32"].to_numpy())
    np.testing.assert_allclose(basic_64_ref, df["basic_64"].to_numpy())
    np.testing.assert_allclose(round_ref, df["round"].to_numpy())
    np.testing.assert_allclose(decimal_only_ref, df["decimal_only"].to_numpy())
    np.testing.assert_allclose(precision_ref, df["precision"].to_numpy())


def test_csv_reader_NaN_values():
    names = dtypes = ["float32"]
    empty_cells = '\n""\n'
    default_na_cells = (
        "#N/A\n#N/A N/A\n#NA\n-1.#IND\n"
        "-1.#QNAN\n-NaN\n-nan\n1.#IND\n"
        "1.#QNAN\nN/A\n<NA>\nNA\nNULL\n"
        "NaN\nn/a\nnan\nnull\n"
    )
    custom_na_cells = "NV_NAN\nNotANumber\n"
    all_cells = empty_cells + default_na_cells + custom_na_cells
    custom_na_values = ["NV_NAN", "NotANumber"]

    # test default NA values. empty cells should also yield NaNs
    gdf = read_csv(
        StringIO(default_na_cells + empty_cells), names=names, dtype=dtypes
    )
    pdf = pd.read_csv(
        StringIO(default_na_cells + empty_cells), names=names, dtype=np.float32
    )
    assert_eq(pdf, gdf)

    # custom NA values
    gdf = read_csv(
        StringIO(all_cells),
        names=names,
        dtype=dtypes,
        na_values=custom_na_values,
    )
    pdf = pd.read_csv(
        StringIO(all_cells),
        names=names,
        dtype=np.float32,
        na_values=custom_na_values,
    )
    assert_eq(pdf, gdf)

    # custom NA values
    gdf = read_csv(
        StringIO(empty_cells + default_na_cells + "_NAA_\n"),
        names=names,
        dtype=dtypes,
        na_values="_NAA_",
    )
    pdf = pd.read_csv(
        StringIO(empty_cells + default_na_cells + "_NAA_\n"),
        names=names,
        dtype=np.float32,
        na_values="_NAA_",
    )
    assert_eq(pdf, gdf)

    # data type detection should evaluate the column to int8 (all nulls)
    gdf = read_csv(
        StringIO(all_cells),
        header=None,
        na_values=custom_na_values,
    )
    assert gdf.dtypes.iloc[0] == "int8"
    assert all(gdf["0"][idx] is cudf.NA for idx in range(len(gdf["0"])))

    # data type detection should evaluate the column to object if some nulls
    gdf = read_csv(StringIO(all_cells), header=None)
    assert gdf.dtypes.iloc[0] == np.dtype("object")


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

    np.testing.assert_allclose(f32_ref, df["float32"].to_numpy())
    np.testing.assert_allclose(f64_ref, df["float64"].to_numpy())
    np.testing.assert_allclose(int32_ref, df["int32"].to_numpy())
    np.testing.assert_allclose(int64_ref, df["int64"].to_numpy())
    np.testing.assert_allclose(uint32_ref, df["uint32"].to_numpy())
    np.testing.assert_allclose(uint64_ref, df["uint64"].to_numpy())


def test_csv_reader_buffer_strings():
    names = ["text", "int"]
    dtypes = ["str", "int"]
    lines = [",".join(names), "a,0", "b,0", "c,0", "d,0"]

    buffer = "\n".join(lines)

    df = read_csv(StringIO(buffer), names=names, dtype=dtypes, skiprows=1)
    assert len(df.columns) == 2
    assert df["text"].dtype == np.dtype("object")
    assert df["int"].dtype == np.dtype("int64")
    assert df["text"][0] == "a"
    assert df["text"][1] == "b"
    assert df["text"][2] == "c"
    assert df["text"][3] == "d"

    df2 = read_csv(
        BytesIO(str.encode(buffer)), names=names, dtype=dtypes, skiprows=1
    )
    assert len(df2.columns) == 2
    assert df2["text"].dtype == np.dtype("object")
    assert df2["int"].dtype == np.dtype("int64")
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
        (".beez", "bz2", np.str_("bz2")),
        (".data", None, "infer"),
        (".txt", None, None),
        ("", None, None),
    ],
)
def test_csv_reader_compression(
    tmpdir, ext, out_comp, in_comp, pd_mixed_dataframe
):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_compression" + ext)

    df = pd_mixed_dataframe
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


def test_csv_reader_bools_custom():
    names = ["text", "bool"]
    dtypes = {"text": "str", "bool": "bool"}
    trues = ["foo", "1"]
    falses = ["bar", "0"]
    lines = [
        ",".join(names),
        "true,true",
        "false,false",
        "foo,foo",
        "bar,bar",
        "0,0",
        "1,1",
    ]
    buffer = "\n".join(lines)

    df = read_csv(
        StringIO(buffer),
        names=names,
        dtype=dtypes,
        skiprows=1,
        true_values=trues,
        false_values=falses,
    )

    # Note: bool literals give parsing errors as int
    # "0" and "1" give parsing errors as bool in pandas
    expected = pd.read_csv(
        StringIO(buffer),
        names=names,
        dtype=dtypes,
        skiprows=1,
        true_values=trues,
        false_values=falses,
    )
    assert_eq(df, expected, check_dtype=True)


def test_csv_reader_bools_NA():
    names = ["text", "int"]
    dtypes = ["str", "int"]
    trues = ["foo"]
    falses = ["bar"]
    lines = [
        ",".join(names),
        "true,true",
        "false,false",
        "foo,foo",
        "bar,bar",
        "qux,qux",
    ]

    buffer = "\n".join(lines)

    df = read_csv(
        StringIO(buffer),
        names=names,
        dtype=dtypes,
        skiprows=1,
        true_values=trues,
        false_values=falses,
    )
    assert len(df.columns) == 2
    assert df["text"].dtype == np.dtype("object")
    assert df["int"].dtype == np.dtype("int64")
    expected = pd.DataFrame(
        {
            "text": ["true", "false", "foo", "bar", "qux"],
            "int": [1.0, 0.0, 1.0, 0.0, np.nan],
        }
    )
    assert_eq(df, expected)


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
    np.testing.assert_allclose(integer_ref, df1["integer"].to_numpy())
    np.testing.assert_allclose(decimal_ref, df1["decimal"].to_numpy())
    np.testing.assert_allclose(integer_ref, df2["integer"].to_numpy())
    np.testing.assert_allclose(decimal_ref, df2["decimal"].to_numpy())


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
    assert str(skip_rows) in next(iter(df))
    assert str(2 * skip_rows) in list(df)[1]
    for row in range(0, read_rows // sample_skip, sample_skip):
        assert df[next(iter(df))][row] == row + skip_rows + 1
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
    assert df["int"].dtype == np.dtype("int64")
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


def test_small_zip(tmpdir):
    df = pd.DataFrame(
        {
            "a": [1997] * 2,
            "b": ["Ford"] * 2,
            "c": ["Super, luxurious truck"] * 2,
        }
    )

    fname = tmpdir.join("small_zip_file.zip")
    df.to_csv(fname, index=False)

    got = cudf.read_csv(fname)
    assert_eq(df, got)


def test_csv_reader_carriage_return(tmpdir):
    rows = 1000
    names = ["int_row", "int_double_row"]
    buffer = ",".join(names) + "\r\n"
    for row in range(rows):
        buffer += str(row) + ", " + str(2 * row) + "\r\n"

    df = read_csv(StringIO(buffer))
    expect = cudf.DataFrame(
        {"int_row": cp.arange(rows), "int_double_row": cp.arange(rows) * 2}
    )

    assert len(df) == rows
    assert_eq(expect, df)


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
    np.testing.assert_allclose(floats, df["float_point"].to_numpy())
    np.testing.assert_allclose(ints, df["integer"].to_numpy())
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
    with pytest.raises(OverflowError, match="Offset is past end of file"):
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
@pytest.mark.parametrize("lineterminator", ["\n", "\r\n"])
def test_csv_reader_blanks_and_comments(
    skip_rows, header_row, skip_blanks, lineterminator
):
    lines = [
        "# first comment line",
        lineterminator,
        "# third comment line",
        "1,2,3",
        "4,5,6",
        "7,8,9",
        lineterminator,
        "# last comment line",
        lineterminator,
        "1,1,1",
    ]
    buffer = lineterminator.join(lines)

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
    with pytest.warns(FutureWarning):
        cu_df = read_csv(StringIO(buffer), delim_whitespace=True)
    with expect_warning_if(PANDAS_GE_220):
        pd_df = pd.read_csv(StringIO(buffer), delim_whitespace=True)
    assert_eq(pd_df, cu_df)

    # without header row
    with pytest.warns(FutureWarning):
        cu_df = read_csv(StringIO(buffer), delim_whitespace=True, header=None)
    with expect_warning_if(PANDAS_GE_220):
        pd_df = pd.read_csv(
            StringIO(buffer), delim_whitespace=True, header=None
        )
    assert pd_df.shape == cu_df.shape

    # should raise an error if used with delimiter or sep
    with pytest.raises(ValueError):
        with pytest.warns(FutureWarning):
            read_csv(StringIO(buffer), delim_whitespace=True, delimiter=" ")
    with pytest.raises(ValueError):
        with pytest.warns(FutureWarning):
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


@pytest.mark.parametrize("index_name", [None, "custom name", 124])
@pytest.mark.parametrize("index_col", [None, 0, "a"])
def test_csv_reader_index_names(index_name, index_col):
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [10, 11, 12]}, index=["AB", "CD", "EF"]
    )
    pdf.index.name = index_name

    buffer = pdf.to_csv()
    actual = cudf.read_csv(StringIO(buffer), index_col=index_col)
    expected = pd.read_csv(StringIO(buffer), index_col=index_col)
    assert_eq(actual, expected)


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


def test_csv_reader_repeated_column_name():
    buffer = """A,A,A.1,A,A.2,A,A.4,A,A
                1,2,3.1,4,a.2,a,a.4,a,a
                2,4,6.1,8,b.2,b,b.4,b,b"""

    # pandas and cudf to have same repeated column names
    pdf = pd.read_csv(StringIO(buffer))
    gdf = cudf.read_csv(StringIO(buffer))
    assert_eq(pdf.columns, gdf.columns)


def test_csv_reader_bools_false_positives(tmpdir):
    # values that are equal to ["True", "TRUE", "False", "FALSE"]
    # when using ints to detect bool values
    items = [3977, 4329, 24015, 27567]

    buffer = "\n".join(str(i) for i in items)

    df = read_csv(StringIO(buffer), header=None, dtype=["int32"])

    np.testing.assert_array_equal(items, df["0"].to_numpy())


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
    lines = ["0x0", "-0x1000", "0xfedcba", "0xABCDEF", "0xaBcDeF"]
    values = [int(hex_int, 16) for hex_int in lines]

    buffer = "\n".join(lines)

    if gdf_dtype is not None:
        # require explicit `hex` dtype to parse hexadecimals
        pdf = pd.DataFrame(data=values, dtype=pdf_dtype, columns=["hex_int"])
        gdf = read_csv(StringIO(buffer), dtype=[gdf_dtype], names=["hex_int"])
        np.testing.assert_array_equal(
            pdf["hex_int"], gdf["hex_int"].to_numpy()
        )
    else:
        # otherwise, dtype inference returns as object (string)
        pdf = pd.read_csv(StringIO(buffer), names=["hex_int"])
        gdf = read_csv(StringIO(buffer), names=["hex_int"])
        assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "np_dtype, gdf_dtype",
    [("int", "hex"), ("int32", "hex32"), ("int64", "hex64")],
)
def test_csv_reader_hexadecimal_overflow(np_dtype, gdf_dtype):
    # This tests values which cause an overflow warning that will become an
    # error in pandas. NumPy wraps the overflow silently up to the bounds of a
    # signed int64.
    lines = [
        "0x0",
        "-0x1000",
        "0xfedcba",
        "0xABCDEF",
        "0xaBcDeF",
        "0x9512c20b",
        "0x7fffffff",
        "0x7fffffffffffffff",
        "-0x8000000000000000",
    ]
    values = [int(hex_int, 16) for hex_int in lines]
    buffer = "\n".join(lines)

    gdf = read_csv(StringIO(buffer), dtype=[gdf_dtype], names=["hex_int"])

    expected = np.array(values).astype(np_dtype)
    actual = gdf["hex_int"].to_numpy()
    np.testing.assert_array_equal(expected, actual)


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


@pytest.mark.parametrize("lineterminator", ["\n", "\r\n"])
def test_csv_blank_first_row(lineterminator):
    lines = ["colA,colB", "", "1, 1.1", "2, 2.2"]
    buffer = lineterminator.join(lines)

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
    out_dtypes = ["int64", "object", "float64", "int16"]

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
    out_dtypes = ["int64", "object", "float64", "int16"]

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

    assert_eq(names_df, header_df)
    assert all(names_df.dtypes == ["int16", "int64"])


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
    assert_eq(result, expected, check_index_type=True)


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
    df.to_csv(path_or_buf=pdf_df_fname, index=False, lineterminator="\n")
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
    df.to_csv(path_or_buf=pdf_df_fname, index=False, lineterminator="\n")
    gdf.to_csv(path_or_buf=gdf_df_fname, index=False)

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect = pd.read_csv(pdf_df_fname)
    got = pd.read_csv(gdf_df_fname)
    assert_eq(expect, got)


@pytest.mark.parametrize("lineterminator", ["\r", "\n", "\t", np.str_("\n")])
@pytest.mark.parametrize("sep", [",", "/", np.str_(",")])
def test_csv_writer_terminator_sep(lineterminator, sep, cudf_mixed_dataframe):
    df = cudf_mixed_dataframe

    buffer = BytesIO()
    df.to_csv(buffer, lineterminator=lineterminator, sep=sep, index=False)

    got = read_csv(buffer, lineterminator=lineterminator, sep=sep)
    assert_eq(df, got)


@pytest.mark.parametrize(
    "lineterminator", ["\r\n", "ABC", "\t\t", np.str_("\r\n")]
)
def test_csv_writer_multichar_terminator(lineterminator, cudf_mixed_dataframe):
    df = cudf_mixed_dataframe

    default_terminator_csv = StringIO()
    df.to_csv(default_terminator_csv)

    # Need to check manually since readers don't support
    # multicharacter line terminators
    expected = default_terminator_csv.getvalue().replace("\n", lineterminator)

    buffer = StringIO()
    df.to_csv(buffer, lineterminator=lineterminator)
    got = buffer.getvalue()

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "columns",
    [
        ["Date", "Float"],
        ["Integer2", "Float", "Date", "Integer", "String", "Boolean"],
        None,
    ],
)
@pytest.mark.parametrize(
    "header", [True, False, np.bool_(True), np.bool_(False)]
)
@pytest.mark.parametrize(
    "index", [True, False, np.bool_(True), np.bool_(False)]
)
def test_csv_writer_column_and_header_options(
    columns, header, index, pd_mixed_dataframe
):
    pdf = pd_mixed_dataframe
    df = cudf.from_pandas(pdf)

    cudf_buffer = BytesIO()
    df.to_csv(cudf_buffer, columns=columns, header=header, index=index)
    pd_buffer = BytesIO()
    pdf.to_csv(pd_buffer, columns=columns, header=header, index=index)

    expected = cudf.read_csv(pd_buffer, header=0 if header else None)
    got = cudf.read_csv(cudf_buffer, header=0 if header else None)

    expected_column_cnt = (1 if index else 0) + (
        len(columns) if columns else pdf.shape[1]
    )
    assert_eq(expected_column_cnt, got.shape[1])
    assert_eq(expected, got)


def test_csv_writer_empty_columns_parameter(cudf_mixed_dataframe):
    df = cudf_mixed_dataframe
    write_str = df.to_csv(columns=[], index=False)
    assert_eq(write_str, "\n")


def test_csv_writer_multiindex(tmpdir):
    pdf_df_fname = tmpdir.join("pdf_df_3.csv")
    gdf_df_fname = tmpdir.join("gdf_df_3.csv")

    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {
            "a": rng.integers(0, 5, 20),
            "b": rng.integers(0, 5, 20),
            "c": range(20),
            "d": rng.random(20),
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
@pytest.mark.parametrize("dtype", dtypes)
def test_csv_writer_chunksize(chunksize, dtype):
    cu_df = cudf.from_pandas(make_numeric_dataframe(100, dtype))

    buffer = BytesIO()
    cu_df.to_csv(buffer, chunksize=chunksize, index=False)

    got = cudf.read_csv(buffer, dtype=[dtype])
    assert_eq(cu_df, got)


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


def test_csv_write_no_caller_manipulation():
    df = cudf.DataFrame({"a": [1, 2, 3]})
    df_copy = df.copy(deep=True)
    _ = df.to_csv(index=True)
    assert_eq(df, df_copy)


@pytest.mark.parametrize(
    "df",
    [
        cudf.DataFrame({"a": [1, 2, 3], "": [10, 20, 40]}),
        cudf.DataFrame({"": [10, 20, 40], "a": [1, 2, 3]}),
        cudf.DataFrame(
            {"a": [1, 2, 3], "": [10, 20, 40]},
            index=cudf.Index(["a", "z", "v"], name="custom name"),
        ),
    ],
)
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("columns", [["a"], [""], None])
def test_csv_write_empty_column_name(df, index, columns):
    pdf = df.to_pandas()
    expected = pdf.to_csv(index=index, columns=columns)
    actual = df.to_csv(index=index, columns=columns)

    assert expected == actual


@pytest.mark.parametrize(
    "df",
    [
        cudf.DataFrame(),
        cudf.DataFrame(index=cudf.Index([], name="index name")),
    ],
)
@pytest.mark.parametrize("index", [True, False])
def test_csv_write_empty_dataframe(df, index):
    pdf = df.to_pandas()

    expected = pdf.to_csv(index=index)
    actual = df.to_csv(index=index)

    assert expected == actual


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {
                "a": [1, 2, 3, None],
                "": ["a", "v", None, None],
                None: [12, 12, 32, 44],
            }
        ),
        pd.DataFrame(
            {
                np.nan: [1, 2, 3, None],
                "": ["a", "v", None, None],
                None: [12, 12, 32, 44],
            }
        ),
        pd.DataFrame({"": [1, None, 3, 4]}),
        pd.DataFrame({None: [1, None, 3, 4]}),
        pd.DataFrame(columns=[None, "", "a", "b"]),
        pd.DataFrame(columns=[None]),
        pd.DataFrame(columns=[""]),
    ],
)
@pytest.mark.parametrize(
    "na_rep", ["", "_NA_", "---", "_____CUSTOM_NA_REP______"]
)
def test_csv_write_dataframe_na_rep(df, na_rep):
    gdf = cudf.from_pandas(df)

    expected = df.to_csv(na_rep=na_rep)
    actual = gdf.to_csv(na_rep=na_rep)

    assert expected == actual


@pytest.mark.parametrize(
    "dtype",
    [
        "int",
        "str",
        "float",
        np.int32,
        np.dtype("float32"),
        {"a": "int32", "b": "float64", "c": "uint8"},
        int,
        str,
        object,
    ],
)
def test_csv_reader_dtypes(dtype):
    buf = "a,b,c\n1,10,111\n2,11,112\n3,12,113\n4,13,114\n"

    expected = pd.read_csv(StringIO(buf), dtype=dtype)
    actual = cudf.read_csv(StringIO(buf), dtype=dtype)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "dtype", ["Int64", "UInt32", {"a": "UInt64", "b": "Float64", "c": "Int32"}]
)
def test_csv_reader_nullable_dtypes(dtype):
    buf = "a,b,c\n1,10,111\n2,11,112\n3,12,113\n4,13,114\n"

    expected = pd.read_csv(StringIO(buf), dtype=dtype)
    actual = cudf.read_csv(StringIO(buf), dtype=dtype)

    assert_eq(expected, actual.to_pandas(nullable=True))


@pytest.mark.parametrize(
    "dtype", sorted(list(cudf.utils.dtypes.TIMEDELTA_TYPES))
)
def test_csv_reader_timedetla_dtypes(dtype):
    buf = "a,b,c\n1,10,111\n2,11,112\n3,12,113\n43432423,13342,13243214\n"

    expected = pd.read_csv(StringIO(buf)).astype(dtype)
    actual = cudf.read_csv(StringIO(buf), dtype=dtype)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "dtype", sorted(list(cudf.utils.dtypes.DATETIME_TYPES))
)
def test_csv_reader_datetime_dtypes(dtype):
    buf = "a,b,c\n1,10,111\n2,11,112\n3,12,113\n43432423,13342,13243214\n"

    expected = pd.read_csv(StringIO(buf)).astype(dtype)
    actual = cudf.read_csv(StringIO(buf), dtype=dtype)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "df",
    [
        cudf.DataFrame(
            {
                "a": cudf.Series([1, 2, 3, 1, 2], dtype="category"),
                "b": cudf.Series(["a", "c", "a", "b", "a"], dtype="category"),
            }
        ),
        cudf.DataFrame(
            {
                "a": cudf.Series([1.1, 2, 3, 1.1, 2], dtype="category"),
                "b": cudf.Series(
                    [None, "c", None, "b", "a"], dtype="category"
                ),
            }
        ),
        cudf.DataFrame(
            {
                "b": cudf.Series(
                    [1.1, 2, 3, 1.1, 2],
                    dtype="category",
                    index=cudf.CategoricalIndex(
                        ["abc", "def", "ghi", "jkl", "xyz"]
                    ),
                )
            }
        ),
    ],
)
def test_csv_writer_category(df):
    pdf = df.to_pandas()

    expected = pdf.to_csv()
    actual = df.to_csv()

    assert expected == actual


@pytest.mark.parametrize(
    "dtype",
    [
        "category",
        {"a": "category", "b": "str"},
        {"b": "category"},
        {"a": "category"},
        {"a": pd.CategoricalDtype([1, 2])},
        {"b": pd.CategoricalDtype([1, 2, 3])},
        {"b": pd.CategoricalDtype(["b", "a"]), "a": "str"},
        pd.CategoricalDtype(["a", "b"]),
    ],
)
def test_csv_reader_category(dtype):
    df = cudf.DataFrame({"a": [1, 2, 3, None], "b": ["a", "b", None, "c"]})
    csv_buf = df.to_csv()

    actual = cudf.read_csv(StringIO(csv_buf), dtype=dtype)
    expected = pd.read_csv(StringIO(csv_buf), dtype=dtype)

    assert_eq(expected, actual, check_dtype=True)


def test_csv_writer_datetime_sep():
    df = cudf.DataFrame(
        {"a": cudf.Series([22343, 2323423, 234324234], dtype="datetime64[ns]")}
    )
    df["a"] = df["a"].astype("datetime64[s]")
    expected = df.to_pandas().to_csv(date_format="%Y-%m-%dT%H:%M:%SZ", sep="-")
    actual = df.to_csv(sep="-")
    assert expected == actual


def test_na_filter_empty_fields():
    test_na = "TEST_NAN"
    df = pd.DataFrame({"col0": ["valid", None, "also_valid", "", test_na]})
    buffer = df.to_csv(index=False)

    pdf = pd.read_csv(StringIO(buffer), na_filter=False)
    gdf = cudf.read_csv(StringIO(buffer), na_filter=False)
    assert_eq(pdf, gdf)

    pdf = pd.read_csv(StringIO(buffer), keep_default_na=False)
    gdf = cudf.read_csv(StringIO(buffer), keep_default_na=False)
    assert_eq(pdf, gdf)

    pdf = pd.read_csv(
        StringIO(buffer), keep_default_na=False, na_values=test_na
    )
    gdf = cudf.read_csv(
        StringIO(buffer), keep_default_na=False, na_values=test_na
    )
    assert_eq(pdf, gdf)


def test_csv_sep_error():
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    gdf = cudf.DataFrame({"a": [1, 2, 3]})
    assert_exceptions_equal(
        lfunc=pdf.to_csv,
        rfunc=gdf.to_csv,
        lfunc_args_and_kwargs=([], {"sep": "abc"}),
        rfunc_args_and_kwargs=([], {"sep": "abc"}),
    )

    assert_exceptions_equal(
        lfunc=pdf.to_csv,
        rfunc=gdf.to_csv,
        lfunc_args_and_kwargs=([], {"sep": 1}),
        rfunc_args_and_kwargs=([], {"sep": 1}),
    )


def test_to_csv_encoding_error():
    # TODO: Remove this test once following
    # issue is fixed: https://github.com/rapidsai/cudf/issues/2957
    df = cudf.DataFrame({"a": ["你好", "test"]})
    encoding = "utf-8-sig"
    error_message = (
        f"Encoding {encoding} is not supported. "
        + "Currently, only utf-8 encoding is supported."
    )
    with pytest.raises(NotImplementedError, match=re.escape(error_message)):
        df.to_csv("test.csv", encoding=encoding)


def test_to_csv_compression_error():
    df = cudf.DataFrame({"a": ["test"]})
    compression = "snappy"
    error_message = "Writing compressed csv is not currently supported in cudf"
    with pytest.raises(NotImplementedError, match=re.escape(error_message)):
        df.to_csv("test.csv", compression=compression)


def test_empty_df_no_index():
    actual = cudf.DataFrame({})
    buffer = BytesIO()
    actual.to_csv(buffer, index=False)

    result = cudf.read_csv(buffer)

    assert_eq(actual, result)


def test_default_integer_bitwidth(
    cudf_mixed_dataframe, default_integer_bitwidth
):
    # Test that integer columns in csv are _inferred_ as user specified
    # bitwidth
    buf = BytesIO()
    cudf_mixed_dataframe.to_csv(buf)
    buf.seek(0)
    read = cudf.read_csv(buf)
    assert read["Integer"].dtype == np.dtype(f"i{default_integer_bitwidth//8}")
    assert read["Integer2"].dtype == np.dtype(
        f"i{default_integer_bitwidth//8}"
    )


def test_default_integer_bitwidth_partial(
    cudf_mixed_dataframe, default_integer_bitwidth
):
    # Test that integer columns in csv are _inferred_ as user specified
    # bitwidth
    buf = BytesIO()
    cudf_mixed_dataframe.to_csv(buf)
    buf.seek(0)
    read = cudf.read_csv(buf, dtype={"Integer": "int64"})
    assert read["Integer"].dtype == np.dtype("i8")
    assert read["Integer2"].dtype == np.dtype(
        f"i{default_integer_bitwidth//8}"
    )


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_default_integer_bitwidth_extremes(
    cudf_extreme_numeric_dataframe, default_integer_bitwidth
):
    # Test that integer columns in csv are _inferred_ as user specified
    # bitwidth
    buf = BytesIO()
    cudf_extreme_numeric_dataframe.to_csv(buf)
    buf.seek(0)
    read = cudf.read_csv(buf)

    assert read["int64"].dtype == np.dtype(f"i{default_integer_bitwidth//8}")
    assert read["long"].dtype == np.dtype(f"i{default_integer_bitwidth//8}")
    assert read["uint64"].dtype == np.dtype(f"u{default_integer_bitwidth//8}")


def test_default_float_bitwidth(cudf_mixed_dataframe, default_float_bitwidth):
    # Test that float columns in csv are _inferred_ as user specified
    # bitwidth
    buf = BytesIO()
    cudf_mixed_dataframe.to_csv(buf)
    buf.seek(0)
    read = cudf.read_csv(buf)
    assert read["Float"].dtype == np.dtype(f"f{default_float_bitwidth//8}")


def test_default_float_bitwidth_partial(default_float_bitwidth):
    # Test that float columns in csv are _inferred_ as user specified
    # bitwidth
    read = cudf.read_csv(
        StringIO("float1,float2\n1.0,2.0\n3.0,4.0"),
        dtype={"float2": "float64"},
    )
    assert read["float1"].dtype == np.dtype(f"f{default_float_bitwidth//8}")
    assert read["float2"].dtype == np.dtype("f8")


@pytest.mark.parametrize(
    "usecols,names",
    [
        # selection using indices; only names of selected columns are specified
        ([1, 2], ["b", "c"]),
        # selection using indices; names of all columns are specified
        ([1, 2], ["a", "b", "c"]),
        # selection using indices; duplicates
        ([2, 2], ["a", "b", "c"]),
        # selection using indices; out of order
        ([2, 1], ["a", "b", "c"]),
        # selection using names
        (["b"], ["a", "b", "c"]),
        # selection using names; multiple columns
        (["b", "c"], ["a", "b", "c"]),
        # selection using names; duplicates
        (["c", "c"], ["a", "b", "c"]),
        # selection using names; out of order
        (["c", "b"], ["a", "b", "c"]),
    ],
)
def test_column_selection_plus_column_names(usecols, names):
    lines = [
        "num,datetime,text",
        "123,2018-11-13T12:00:00,abc",
        "456,2018-11-14T12:35:01,def",
        "789,2018-11-15T18:02:59,ghi",
    ]

    buffer = "\n".join(lines) + "\n"

    assert_eq(
        pd.read_csv(StringIO(buffer), usecols=usecols, names=names),
        cudf.read_csv(StringIO(buffer), usecols=usecols, names=names),
    )


def test_read_compressed_BOM(tmpdir):
    buffer = 'int, string\n1, "a"\n2, "b"\n3, "c"\n'

    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file20.gz")
    with gzip.open(fname, "wt", encoding="utf-8") as f:
        f.write(codecs.BOM_UTF8.decode("utf-8"))
        f.write(buffer)

    assert_eq(pd.read_csv(fname), cudf.read_csv(fname))


def test_read_header_none_pandas_compat_column_type():
    data = "1\n2\n"
    with cudf.option_context("mode.pandas_compatible", True):
        result = cudf.read_csv(StringIO(data), header=None).columns
    expected = pd.read_csv(StringIO(data), header=None).columns
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize("buffer", ["1", '"one"'])
def test_read_single_unterminated_row(buffer):
    gdf = cudf.read_csv(StringIO(buffer), header=None)
    assert_eq(gdf.shape, (1, 1))


@pytest.mark.parametrize("buffer", ["\n", "\r\n"])
def test_read_empty_only_row(buffer):
    gdf = cudf.read_csv(StringIO(buffer), header=None)
    assert_eq(gdf.shape, (0, 0))


def test_read_empty_only_row_custom_terminator():
    gdf = cudf.read_csv(StringIO("*"), header=None, lineterminator="*")
    assert_eq(gdf.shape, (0, 0))
