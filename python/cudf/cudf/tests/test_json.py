# Copyright (c) 2018-2022, NVIDIA CORPORATION.

import copy
import gzip
import itertools
import os
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_110
from cudf.testing._utils import DATETIME_TYPES, NUMERIC_TYPES, assert_eq


def make_numeric_dataframe(nrows, dtype):
    df = pd.DataFrame()
    df["col1"] = np.arange(nrows, dtype=dtype)
    df["col2"] = np.arange(1, 1 + nrows, dtype=dtype)
    return df


@pytest.fixture(params=[0, 1, 10, 100])
def pdf(request):
    types = NUMERIC_TYPES + DATETIME_TYPES + ["bool"]
    renamer = {
        "C_l0_g" + str(idx): "col_" + val for (idx, val) in enumerate(types)
    }
    typer = {"col_" + val: val for val in types}
    ncols = len(types)
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd._testing.makeCustomDataframe(
        nrows=nrows, ncols=ncols, data_gen_f=lambda r, c: r, r_idx_type="i"
    )
    # Delete the name of the column index, and rename the row index
    test_pdf.columns.name = None
    test_pdf.index.name = "test_index"

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.astype("object").rename(renamer, axis=1).astype(typer)

    return test_pdf


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


index_params = [True, False]
compression_params = ["gzip", "bz2", "zip", "xz", None]
orient_params = ["columns", "records", "table", "split"]
params = itertools.product(index_params, compression_params, orient_params)


@pytest.fixture(params=params)
def json_files(request, tmp_path_factory, pdf):
    index, compression, orient = request.param
    if index is False and orient not in ("split", "table"):
        pytest.skip(
            "'index=False' is only valid when 'orient' is 'split' or "
            "'table'"
        )
    if index is False and orient == "table":
        pytest.skip("'index=False' isn't valid when 'orient' is 'table'")
    fname_df = tmp_path_factory.mktemp("json") / "test_df.json"
    fname_series = tmp_path_factory.mktemp("json") / "test_series.json"
    pdf.to_json(fname_df, index=index, compression=compression, orient=orient)
    pdf["col_int32"].to_json(
        fname_series, index=index, compression=compression, orient=orient
    )
    return (fname_df, fname_series, orient, compression)


@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_reader(json_files):
    path_df, path_series, orient, compression = json_files
    expect_df = pd.read_json(path_df, orient=orient, compression=compression)
    got_df = cudf.read_json(path_df, orient=orient, compression=compression)
    if len(expect_df) == 0:
        expect_df = expect_df.reset_index(drop=True)
        expect_df.columns = expect_df.columns.astype("object")
    if len(got_df) == 0:
        got_df = got_df.reset_index(drop=True)

    assert_eq(expect_df, got_df, check_categorical=False)

    # Only these orients are allowed for Series, but isn't enforced by Pandas
    if orient in ("split", "records", "index"):
        expect_series = pd.read_json(
            path_series, orient=orient, compression=compression, typ="series"
        )
        got_series = cudf.read_json(
            path_series, orient=orient, compression=compression, typ="series"
        )
        if len(expect_series) == 0:
            expect_series = expect_series.reset_index(drop=True)
        if len(got_df) == 0:
            got_series = got_series.reset_index(drop=True)

        assert_eq(expect_series, got_series)


@pytest.mark.filterwarnings("ignore:Can't infer compression")
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_writer(tmpdir, pdf, gdf):
    pdf_df_fname = tmpdir.join("pdf_df.json")
    gdf_df_fname = tmpdir.join("gdf_df.json")

    pdf.to_json(pdf_df_fname)
    gdf.to_json(gdf_df_fname)

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect_df = pd.read_json(pdf_df_fname)
    got_df = pd.read_json(gdf_df_fname)

    assert_eq(expect_df, got_df)

    for column in pdf.columns:
        pdf_series_fname = tmpdir.join(column + "_" + "pdf_series.json")
        gdf_series_fname = tmpdir.join(column + "_" + "gdf_series.json")

        pdf[column].to_json(pdf_series_fname)
        gdf[column].to_json(gdf_series_fname)

        assert os.path.exists(pdf_series_fname)
        assert os.path.exists(gdf_series_fname)

        try:
            # xref 'https://github.com/pandas-dev/pandas/pull/33373'
            expect_series = pd.read_json(pdf_series_fname, typ="series")
        except TypeError as e:
            if (
                not PANDAS_GE_110
                and str(e) == "<class 'bool'> is not convertible to datetime"
            ):
                continue
            else:
                raise e

        got_series = pd.read_json(gdf_series_fname, typ="series")

        assert_eq(expect_series, got_series)

        # Make sure results align for regular strings, not just files
        pdf_string = pdf[column].to_json()
        gdf_string = pdf[column].to_json()
        assert_eq(pdf_string, gdf_string)


@pytest.fixture(
    params=["string", "filepath", "pathobj", "bytes_io", "string_io", "url"]
)
def json_input(request, tmp_path_factory):
    input_type = request.param
    buffer = "[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]\n"
    fname = tmp_path_factory.mktemp("json") / "test_df.json"
    if not os.path.isfile(fname):
        with open(str(fname), "w") as fp:
            fp.write(buffer)

    if input_type == "string":
        return buffer
    if input_type == "filepath":
        return str(fname)
    if input_type == "pathobj":
        return Path(fname)
    if input_type == "bytes_io":
        return BytesIO(buffer.encode())
    if input_type == "string_io":
        return StringIO(buffer)
    if input_type == "url":
        return Path(fname).as_uri()


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("engine", ["auto", "cudf", "pandas"])
def test_json_lines_basic(json_input, engine):
    cu_df = cudf.read_json(json_input, engine=engine, lines=True)
    pd_df = pd.read_json(json_input, lines=True)

    assert all(cu_df.dtypes == ["int64", "int64", "int64"])
    for cu_col, pd_col in zip(cu_df.columns, pd_df.columns):
        assert str(cu_col) == str(pd_col)
        np.testing.assert_array_equal(pd_df[pd_col], cu_df[cu_col].to_numpy())


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("engine", ["auto", "cudf"])
def test_json_lines_multiple(tmpdir, json_input, engine):
    tmp_file1 = tmpdir.join("MultiInputs1.json")
    tmp_file2 = tmpdir.join("MultiInputs2.json")

    pdf = pd.read_json(json_input, lines=True)
    pdf.to_json(tmp_file1, compression="infer", lines=True, orient="records")
    pdf.to_json(tmp_file2, compression="infer", lines=True, orient="records")

    cu_df = cudf.read_json([tmp_file1, tmp_file2], engine=engine, lines=True)
    pd_df = pd.concat([pdf, pdf])

    assert all(cu_df.dtypes == ["int64", "int64", "int64"])
    for cu_col, pd_col in zip(cu_df.columns, pd_df.columns):
        assert str(cu_col) == str(pd_col)
        np.testing.assert_array_equal(pd_df[pd_col], cu_df[cu_col].to_numpy())


@pytest.mark.parametrize("engine", ["auto", "cudf"])
def test_json_read_directory(tmpdir, json_input, engine):
    pdf = pd.read_json(json_input, lines=True)
    pdf.to_json(
        tmpdir.join("MultiInputs1.json"),
        compression="infer",
        lines=True,
        orient="records",
    )
    pdf.to_json(
        tmpdir.join("MultiInputs2.json"),
        compression="infer",
        lines=True,
        orient="records",
    )
    pdf.to_json(
        tmpdir.join("MultiInputs3.json"),
        compression="infer",
        lines=True,
        orient="records",
    )

    cu_df = cudf.read_json(tmpdir, engine=engine, lines=True)
    pd_df = pd.concat([pdf, pdf, pdf])

    assert all(cu_df.dtypes == ["int64", "int64", "int64"])
    for cu_col, pd_col in zip(cu_df.columns, pd_df.columns):
        assert str(cu_col) == str(pd_col)
        np.testing.assert_array_equal(pd_df[pd_col], cu_df[cu_col].to_numpy())


def test_json_lines_byte_range(json_input):
    # include the first row and half of the second row
    # should parse the first two rows
    df = cudf.read_json(
        copy.deepcopy(json_input), lines=True, byte_range=(0, 15)
    )
    assert df.shape == (2, 3)

    # include half of the second row and half of the third row
    # should parse only the third row
    df = cudf.read_json(
        copy.deepcopy(json_input), lines=True, byte_range=(15, 10)
    )
    assert df.shape == (1, 3)

    # include half of the second row and entire third row
    # should parse only the third row
    df = cudf.read_json(
        copy.deepcopy(json_input), lines=True, byte_range=(15, 0)
    )
    assert df.shape == (1, 3)

    # include half of the second row till past the end of the file
    # should parse only the third row
    df = cudf.read_json(
        copy.deepcopy(json_input), lines=True, byte_range=(10, 50)
    )
    assert df.shape == (1, 3)


def test_json_lines_dtypes(json_input):
    df = cudf.read_json(
        json_input, lines=True, dtype={1: "int", 2: "short", 0: "float"}
    )
    assert all(df.dtypes == ["float64", "int64", "int16"])


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
def test_json_lines_compression(tmpdir, ext, out_comp, in_comp):
    fname = tmpdir.mkdir("gdf_json").join("tmp_json_compression" + ext)

    nrows = 20
    pd_df = make_numeric_dataframe(nrows, np.int32)
    pd_df.to_json(fname, compression=out_comp, lines=True, orient="records")

    cu_df = cudf.read_json(
        str(fname),
        compression=in_comp,
        lines=True,
        dtype={"col1": "int32", "col2": "int32"},
    )
    assert_eq(pd_df, cu_df)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_engine_selection():
    json = "[1, 2, 3]"

    # should use the cudf engine
    df = cudf.read_json(json, lines=True)
    # column names are strings when parsing with cudf
    for col_name in df.columns:
        assert isinstance(col_name, str)

    # should use the pandas engine
    df = cudf.read_json(json, lines=False)
    # column names are ints when parsing with pandas
    for col_name in df.columns:
        assert isinstance(col_name, int)

    # should use the pandas engine
    df = cudf.read_json(json, lines=True, engine="pandas")
    # column names are ints when parsing with pandas
    for col_name in df.columns:
        assert isinstance(col_name, int)

    # should raise an exception
    with pytest.raises(ValueError):
        cudf.read_json(json, lines=False, engine="cudf")


def test_json_bool_values():
    buffer = "[true,1]\n[false,false]\n[true,true]"
    cu_df = cudf.read_json(buffer, lines=True)
    pd_df = pd.read_json(buffer, lines=True)

    # types should be ['bool', 'int64']
    np.testing.assert_array_equal(pd_df.dtypes, cu_df.dtypes)
    np.testing.assert_array_equal(pd_df[0], cu_df["0"].to_numpy())
    # boolean values should be converted to 0/1
    np.testing.assert_array_equal(pd_df[1], cu_df["1"].to_numpy())

    cu_df = cudf.read_json(
        buffer, lines=True, dtype={"0": "bool", "1": "long"}
    )
    np.testing.assert_array_equal(pd_df.dtypes, cu_df.dtypes)


@pytest.mark.parametrize(
    "buffer",
    [
        "[1.0,]\n[null, ]",
        '{"0":1.0,"1":}\n{"0":null,"1": }',
        '{ "0" : 1.0 , "1" : }\n{ "0" : null , "1" : }',
        '{"0":1.0}\n{"1":}',
    ],
)
def test_json_null_literal(buffer):
    df = cudf.read_json(buffer, lines=True)

    # first column contains a null field, type should be set to float
    # second column contains only empty fields, type should be set to int8
    np.testing.assert_array_equal(df.dtypes, ["float64", "int8"])
    np.testing.assert_array_equal(
        df["0"].to_numpy(na_value=np.nan), [1.0, np.nan]
    )
    np.testing.assert_array_equal(df["1"].to_numpy(na_value=0), [0, 0])


def test_json_bad_protocol_string():
    test_string = '{"field": "s3://path"}'

    expect = pd.DataFrame([{"field": "s3://path"}])
    got = cudf.read_json(test_string, lines=True)

    assert_eq(expect, got)


def test_json_corner_case_with_escape_and_double_quote_char_with_pandas(
    tmpdir,
):
    fname = tmpdir.mkdir("gdf_json").join("tmp_json_escape_double_quote")

    pdf = pd.DataFrame(
        {
            "a": ['ab"cd', "\\\b", "\r\\", "'"],
            "b": ["a\tb\t", "\\", '\\"', "\t"],
            "c": ["aeiou", "try", "json", "cudf"],
        }
    )
    pdf.to_json(fname, compression="infer", lines=True, orient="records")

    df = cudf.read_json(
        fname, compression="infer", lines=True, orient="records"
    )
    pdf = pd.read_json(
        fname, compression="infer", lines=True, orient="records"
    )

    assert_eq(cudf.DataFrame(pdf), df)


def test_json_corner_case_with_escape_and_double_quote_char_with_strings():
    str_buffer = StringIO(
        """{"a":"ab\\"cd","b":"a\\tb\\t","c":"aeiou"}
           {"a":"\\\\\\b","b":"\\\\","c":"try"}
           {"a":"\\r\\\\","b":"\\\\\\"","c":"json"}
           {"a":"\'","b":"\\t","c":"cudf"}"""
    )

    df = cudf.read_json(
        str_buffer, compression="infer", lines=True, orient="records"
    )

    expected = {
        "a": ['ab"cd', "\\\b", "\r\\", "'"],
        "b": ["a\tb\t", "\\", '\\"', "\t"],
        "c": ["aeiou", "try", "json", "cudf"],
    }

    num_rows = df.shape[0]
    for col_name in df._data:
        for i in range(num_rows):
            assert expected[col_name][i] == df[col_name][i]


@pytest.mark.parametrize(
    "gdf,pdf",
    [
        (
            cudf.DataFrame(
                {
                    "int col": cudf.Series(
                        [1, 2, None, 2, 2323, 234, None], dtype="int64"
                    )
                }
            ),
            pd.DataFrame(
                {
                    "int col": pd.Series(
                        [1, 2, None, 2, 2323, 234, None], dtype=pd.Int64Dtype()
                    )
                }
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "int64 col": cudf.Series(
                        [1, 2, None, 2323, None], dtype="int64"
                    ),
                    "string col": cudf.Series(
                        ["abc", "a", None, "", None], dtype="str"
                    ),
                    "float col": cudf.Series(
                        [0.234, None, 234234.2343, None, 0.0], dtype="float64"
                    ),
                    "bool col": cudf.Series(
                        [None, True, False, None, True], dtype="bool"
                    ),
                    "categorical col": cudf.Series(
                        [1, 2, 1, None, 2], dtype="category"
                    ),
                    "datetime col": cudf.Series(
                        [1231233, None, 2323234, None, 1],
                        dtype="datetime64[ns]",
                    ),
                    "timedelta col": cudf.Series(
                        [None, 34687236, 2323234, 1, None],
                        dtype="timedelta64[ns]",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "int64 col": pd.Series(
                        [1, 2, None, 2323, None], dtype=pd.Int64Dtype()
                    ),
                    "string col": pd.Series(
                        ["abc", "a", None, "", None], dtype=pd.StringDtype()
                    ),
                    "float col": pd.Series(
                        [0.234, None, 234234.2343, None, 0.0], dtype="float64"
                    ),
                    "bool col": pd.Series(
                        [None, True, False, None, True],
                        dtype=pd.BooleanDtype(),
                    ),
                    "categorical col": pd.Series(
                        [1, 2, 1, None, 2], dtype="category"
                    ),
                    "datetime col": pd.Series(
                        [1231233, None, 2323234, None, 1],
                        dtype="datetime64[ns]",
                    ),
                    "timedelta col": pd.Series(
                        [None, 34687236, 2323234, 1, None],
                        dtype="timedelta64[ns]",
                    ),
                }
            ),
        ),
    ],
)
def test_json_to_json_compare_contents(gdf, pdf):
    expected_json = pdf.to_json(lines=True, orient="records")
    actual_json = gdf.to_json(lines=True, orient="records")

    assert expected_json == actual_json


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("engine", ["cudf", "pandas"])
def test_default_integer_bitwidth(default_integer_bitwidth, engine):
    buf = BytesIO()
    pd.DataFrame({"a": range(10)}).to_json(buf, lines=True, orient="records")
    buf.seek(0)
    df = cudf.read_json(buf, engine=engine, lines=True, orient="records")

    assert df["a"].dtype == np.dtype(f"i{default_integer_bitwidth//8}")


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(
            "cudf",
            marks=pytest.mark.skip(
                reason="cannot partially set dtypes for cudf json engine"
            ),
        ),
        "pandas",
    ],
)
def test_default_integer_bitwidth_partial(default_integer_bitwidth, engine):
    buf = BytesIO()
    pd.DataFrame({"a": range(10), "b": range(10, 20)}).to_json(
        buf, lines=True, orient="records"
    )
    buf.seek(0)
    df = cudf.read_json(
        buf, engine=engine, lines=True, orient="records", dtype={"b": "i8"}
    )

    assert df["a"].dtype == np.dtype(f"i{default_integer_bitwidth//8}")
    assert df["b"].dtype == np.dtype("i8")


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("engine", ["cudf", "pandas"])
def test_default_integer_bitwidth_extremes(default_integer_bitwidth, engine):
    # Test that integer columns in json are _inferred_ as 32 bit columns.
    buf = StringIO(
        '{"u8":18446744073709551615, "i8":9223372036854775807}\n'
        '{"u8": 0, "i8": -9223372036854775808}'
    )
    df = cudf.read_json(buf, engine=engine, lines=True, orient="records")

    assert df["u8"].dtype == np.dtype(f"u{default_integer_bitwidth//8}")
    assert df["i8"].dtype == np.dtype(f"i{default_integer_bitwidth//8}")


def test_default_float_bitwidth(default_float_bitwidth):
    # Test that float columns in json are _inferred_ as 32 bit columns.
    df = cudf.read_json(
        '{"a": 1.0, "b": 2.5}\n{"a": 3.5, "b": 4.0}',
        engine="cudf",
        lines=True,
        orient="records",
    )
    assert df["a"].dtype == np.dtype(f"f{default_float_bitwidth//8}")
    assert df["b"].dtype == np.dtype(f"f{default_float_bitwidth//8}")


def test_json_nested_basic(tmpdir):
    fname = tmpdir.mkdir("gdf_json").join("tmp_json_nested_basic")
    data = {
        "c1": [{"f1": "sf11", "f2": "sf21"}, {"f1": "sf12", "f2": "sf22"}],
        "c2": [["l11", "l21"], ["l12", "l22"]],
    }
    pdf = pd.DataFrame(data)
    pdf.to_json(fname, orient="records")

    df = cudf.read_json(fname, engine="cudf_experimental", orient="records")
    pdf = pd.read_json(fname, orient="records")

    assert_eq(pdf, df)


@pytest.mark.parametrize(
    "data",
    [
        {
            "c1": [{"f1": "sf11", "f2": "sf21"}, {"f1": "sf12", "f2": "sf22"}],
            "c2": [["l11", "l21"], ["l12", "l22"]],
        },
        # Essential test case to handle omissions
        {
            "c1": [{"f2": "sf21"}, {"f1": "sf12"}],
            "c2": [["l11", "l21"], []],
        },
        # empty input
        {},
    ],
)
@pytest.mark.parametrize("lines", [True, False])
def test_json_nested_lines(data, lines):
    bytes = BytesIO()
    pdf = pd.DataFrame(data)
    pdf.to_json(bytes, orient="records", lines=lines)
    bytes.seek(0)
    df = cudf.read_json(
        bytes, engine="cudf_experimental", orient="records", lines=lines
    )
    bytes.seek(0)
    pdf = pd.read_json(bytes, orient="records", lines=lines)
    # In the second test-case we need to take a detour via pyarrow
    # Pandas omits "f1" in first row, so we have to enforce a common schema,
    # such that pandas would have the f1 member with null
    # Also, pyarrow chooses to select different ordering of a nested column
    # children though key-value pairs are correct.
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


def test_json_nested_data():
    json_str = (
        '[{"0":{},"2":{}},{"1":[[""],[]],"2":{"2":""}},'
        '{"0":{"a":"1"},"2":{"0":"W&RR=+I","1":""}}]'
    )
    df = cudf.read_json(
        StringIO(json_str), engine="cudf_experimental", orient="records"
    )
    pdf = pd.read_json(StringIO(json_str), orient="records")
    pdf.columns = pdf.columns.astype("str")
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


def test_json_empty_types():
    json_str = """ {}
    {"a": [], "b": {}}
    {"a": []}
    {"b": {}}
    {"c": {"d": []}}
    {"e": [{}]}
    """
    df = cudf.read_json(
        StringIO(json_str),
        engine="cudf_experimental",
        orient="records",
        lines=True,
    )
    pdf = pd.read_json(StringIO(json_str), orient="records", lines=True)
    assert_eq(df, pdf)


def test_json_types_data():
    # 0:<0:string,1:float>
    # 1:list<int>
    # 2:<0:bool>
    json_str = (
        '[{"0":null,"2":{}},'
        '{"1":[123],"0":{"0":"foo","1":123.4},"2":{"0":false}},'
        '{"0":{},"1":[],"2":{"0":null}}]'
    )
    df = cudf.read_json(
        StringIO(json_str), engine="cudf_experimental", orient="records"
    )
    pdf = pd.read_json(StringIO(json_str), orient="records")
    pdf.columns = pdf.columns.astype("str")
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


@pytest.mark.parametrize(
    "col_type,json_str",
    [
        # without quotes
        ("int", '[{"k": 1}, {"k": 2}, {"k": 3}, {"k": 4}]'),
        # with quotes
        ("int", '[{"k": "1"}, {"k": "2"}]'),
        # with quotes, mixed
        ("int", '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]'),
        # with quotes, null, mixed
        ("int", '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]'),
        # without quotes, null
        ("int", '[{"k": 1}, {"k": 2}, {"k": null}, {"k": 4}]'),
        # without quotes
        ("float", '[{"k": 1}, {"k": 2}, {"k": 3}, {"k": 4}]'),
        # with quotes
        ("float", '[{"k": "1"}, {"k": "2"}]'),
        # with quotes, mixed
        ("float", '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]'),
        # with quotes, null, mixed
        ("float", '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]'),
        # with quotes, NAN
        ("float", '[{"k": "1"}, {"k": "2"}, {"k": NaN}, {"k": "4"}]'),
        # without quotes
        ("str", '[{"k": 1}, {"k": 2}, {"k": 3}, {"k": 4}]'),
        # with quotes
        ("str", '[{"k": "1"}, {"k": "2"}]'),
        # with quotes, mixed
        ("str", '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]'),
        # with quotes, null, mixed
        ("str", '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]'),
        # without quotes, null
        ("str", '[{"k": 1}, {"k": 2}, {"k": null}, {"k": 4}]'),
    ],
)
def test_json_quoted_values_with_schema(col_type, json_str):
    experimental_df = cudf.read_json(
        StringIO(json_str),
        engine="cudf_experimental",
        orient="records",
        dtype={"k": col_type},
    )
    cudf_df = cudf.read_json(
        StringIO(json_str.replace(",", "\n")[1:-1]),
        engine="cudf",
        orient="records",
        lines=True,
        dtype={"k": col_type},
    )
    assert_eq(cudf_df, experimental_df)


@pytest.mark.parametrize(
    "col_type,json_str,expected",
    [
        # with quotes, mixed
        ("int", '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]', [1, 2, 3, 4]),
        # with quotes, null, mixed
        (
            "int",
            '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]',
            [1, 2, None, 4],
        ),
        # with quotes, mixed
        (
            "str",
            '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]',
            ["1", "2", "3", "4"],
        ),
        # with quotes, null, mixed
        (
            "str",
            '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]',
            ["1", "2", None, "4"],
        ),
    ],
)
def test_json_quoted_values(col_type, json_str, expected):
    experimental_df = cudf.read_json(
        StringIO(json_str),
        engine="cudf_experimental",
        orient="records",
        dtype={"k": col_type},
    )
    cudf_df = cudf.read_json(
        StringIO(json_str.replace(",", "\n")[1:-1]),
        engine="cudf",
        orient="records",
        lines=True,
        dtype={"k": col_type},
    )
    assert_eq(expected, experimental_df.k.to_arrow().to_pylist())
    assert_eq(expected, cudf_df.k.to_arrow().to_pylist())


@pytest.mark.parametrize(
    "keep_quotes,result",
    [
        (
            True,
            {
                "c1": [
                    {"f1": '"sf11"', "f2": '"sf21"'},
                    {"f1": '"sf12"', "f2": '"sf22"'},
                ],
                "c2": [['"l11"', '"l21"'], ['"l12"', '"l22"']],
            },
        ),
        (
            False,
            {
                "c1": [
                    {"f1": "sf11", "f2": "sf21"},
                    {"f1": "sf12", "f2": "sf22"},
                ],
                "c2": [["l11", "l21"], ["l12", "l22"]],
            },
        ),
    ],
)
def test_json_keep_quotes(keep_quotes, result):
    bytes_file = BytesIO()
    data = {
        "c1": [{"f1": "sf11", "f2": "sf21"}, {"f1": "sf12", "f2": "sf22"}],
        "c2": [["l11", "l21"], ["l12", "l22"]],
    }
    pdf = pd.DataFrame(data)
    pdf.to_json(bytes_file, orient="records", lines=True)

    actual = cudf.read_json(
        bytes_file,
        engine="cudf_experimental",
        orient="records",
        lines=True,
        keep_quotes=keep_quotes,
    )
    expected = pd.DataFrame(result)

    assert_eq(actual, expected)


def test_json_dtypes_nested_data():
    # a: StructDtype({'a': StructDtype({'b': dtype('float64')}),
    #                 'b': dtype('int64')})
    # b: ListDtype(ListDtype(float64))
    actual_json_str = (
        '{"a":{"a":{"b":10.0},"b":11},"b":[[10.0,1.1],[12.0,23.0]]}\n'
        '{"a":{"a":{"b":107.0},"b":5},"b":[[10.0,11.2],[12.0,0.23]]}\n'
        '{"a":{"a":{"b":50.7},"b":2},"b":[[10.0,11.3],[12.0,2.3]]}\n'
        '{"a":{"a":{"b":1.2},"b":67},"b":[[6.0,7.0]]}\n'
        '{"a":{"a":{"b":40.1},"b":1090},"b":null}\n'
    )

    """
    In [3]: df
    Out[3]:
                                   a                             b
    0    {'a': {'b': 10.0}, 'b': 11}   [[10.0, 1.1], [12.0, 23.0]]
    1    {'a': {'b': 107.0}, 'b': 5}  [[10.0, 11.2], [12.0, 0.23]]
    2     {'a': {'b': 50.7}, 'b': 2}   [[10.0, 11.3], [12.0, 2.3]]
    3     {'a': {'b': 1.2}, 'b': 67}                  [[6.0, 7.0]]
    4  {'a': {'b': 40.1}, 'b': 1090}                          None
    """

    # a: StructDtype({'a': StructDtype({'b': dtype('int64')}),
    #                 'b': dtype('float64')})
    # b: ListDtype(ListDtype(int64))
    expected_json_str = (
        '{"a":{"a":{"b":10},"b":11.0},"b":[[10,1],[12,23]]}\n'
        '{"a":{"a":{"b":107},"b":5.0},"b":[[10,11],[12,0]]}\n'
        '{"a":{"a":{"b":50},"b":2.0},"b":[[10,11],[12,2]]}\n'
        '{"a":{"a":{"b":1},"b":67.0},"b":[[6,7]]}\n'
        '{"a":{"a":{"b":40},"b":1090.0},"b":null}\n'
    )

    """
    In [7]: df
    Out[7]:
                                  a                    b
    0    {'a': {'b': 10}, 'b': 11.0}  [[10, 1], [12, 23]]
    1    {'a': {'b': 107}, 'b': 5.0}  [[10, 11], [12, 0]]
    2     {'a': {'b': 50}, 'b': 2.0}  [[10, 11], [12, 2]]
    3     {'a': {'b': 1}, 'b': 67.0}             [[6, 7]]
    4  {'a': {'b': 40}, 'b': 1090.0}                 None
    """

    df = cudf.read_json(
        StringIO(actual_json_str),
        engine="cudf_experimental",
        orient="records",
        lines=True,
        dtype={
            "a": cudf.StructDtype(
                {
                    "a": cudf.StructDtype({"b": cudf.dtype("int64")}),
                    "b": cudf.dtype("float64"),
                }
            ),
            "b": cudf.ListDtype(cudf.ListDtype("int64")),
        },
    )

    pdf = pd.read_json(
        StringIO(expected_json_str), orient="records", lines=True
    )
    pdf.columns = pdf.columns.astype("str")
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


@pytest.mark.parametrize(
    "tag, data",
    [
        (
            "normal",
            """\
{"a": 1, "b": 2}
{"a": 3, "b": 4}""",
        ),
        (
            "multiple",
            """\
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 }""",
        ),
        (
            "reordered",
            """\
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "c": 12 , "b" : [4, 5   ]}
    { "b" : [6      ],  "a": { "y" : 6}, "c": 13}
    { "c" : 14, "a": { "y" : 6}, "b" : [7      ]}
""",
        ),
        (
            "missing",
            """
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ]          }
    { "a": { "y" : 6}, "c": 13                  }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 }
""",
        ),
        pytest.param(
            "dtype_mismatch",
            """\
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14.0 }""",
        ),
    ],
)
class TestNestedJsonReaderCommon:
    @pytest.mark.parametrize("chunk_size", [10, 100, 1024, 1024 * 1024])
    def test_chunked_nested_json_reader(self, tag, data, chunk_size):
        expected = cudf.read_json(
            StringIO(data), engine="cudf_experimental", lines=True
        )

        source_size = len(data)
        chunks = []
        for chunk_start in range(0, source_size, chunk_size):
            chunks.append(
                cudf.read_json(
                    StringIO(data),
                    engine="cudf_experimental",
                    byte_range=[chunk_start, chunk_size],
                    lines=True,
                )
            )
        df = cudf.concat(chunks, ignore_index=True)
        assert expected.to_arrow().equals(df.to_arrow())

    def test_order_nested_json_reader(self, tag, data):
        expected = pd.read_json(StringIO(data), lines=True)
        target = cudf.read_json(
            StringIO(data), engine="cudf_experimental", lines=True
        )
        if tag == "dtype_mismatch":
            with pytest.raises(AssertionError):
                # pandas parses integer values in float representation
                # as integer
                assert pa.Table.from_pandas(expected).equals(target.to_arrow())
        elif tag == "missing":
            with pytest.raises(AssertionError):
                # pandas inferences integer with nulls as float64
                assert pa.Table.from_pandas(expected).equals(target.to_arrow())
        else:
            assert pa.Table.from_pandas(expected).equals(target.to_arrow())


def test_json_round_trip_gzip():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": ["abc", "def", "ghi"]})
    bytes = BytesIO()
    with gzip.open(bytes, mode="wb") as fo:
        df.to_json(fo, orient="records", lines=True)
    bytes.seek(0)
    with gzip.open(bytes, mode="rb") as fo:
        written_df = cudf.read_json(fo, orient="records", lines=True)
    assert_eq(written_df, df)

    # Testing writing from middle of the file.
    loc = bytes.tell()

    with gzip.open(bytes, mode="wb") as fo:
        fo.seek(loc)
        df.to_json(fo, orient="records", lines=True)
    bytes.seek(loc)
    with gzip.open(bytes, mode="rb") as fo:
        fo.seek(loc)
        written_df = cudf.read_json(fo, orient="records", lines=True)
    assert_eq(written_df, df)
