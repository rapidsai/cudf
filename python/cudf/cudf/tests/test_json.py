# Copyright (c) 2018, NVIDIA CORPORATION.

import copy
import itertools
import os
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import DATETIME_TYPES, NUMERIC_TYPES, assert_eq


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
            # xref 'https://github.com/pandas-dev/pandas/pull/33373')
            expect_series = pd.read_json(pdf_series_fname, typ="series")
        except TypeError as e:
            if str(e) == "<class 'bool'> is not convertible to datetime":
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
        np.testing.assert_array_equal(pd_df[pd_col], cu_df[cu_col].to_array())


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


@pytest.mark.parametrize(
    "dtype", [["float", "int", "short"], {1: "int", 2: "short", 0: "float"}]
)
def test_json_lines_dtypes(json_input, dtype):
    df = cudf.read_json(json_input, lines=True, dtype=dtype)
    assert all(df.dtypes == ["float32", "int32", "int16"])


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
        str(fname), compression=in_comp, lines=True, dtype=["int", "int"]
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
        df = cudf.read_json(json, lines=False, engine="cudf")


def test_json_bool_values():
    buffer = "[true,1]\n[false,false]\n[true,true]"
    cu_df = cudf.read_json(buffer, lines=True)
    pd_df = pd.read_json(buffer, lines=True)

    # types should be ['bool', 'int64']
    np.testing.assert_array_equal(pd_df.dtypes, cu_df.dtypes)
    np.testing.assert_array_equal(pd_df[0], cu_df["0"].to_array())
    # boolean values should be converted to 0/1
    np.testing.assert_array_equal(pd_df[1], cu_df["1"].to_array())

    cu_df = cudf.read_json(buffer, lines=True, dtype=["bool", "long"])
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
        df["0"].to_array(fillna=np.nan), [1.0, np.nan]
    )
    np.testing.assert_array_equal(
        df["1"].to_array(fillna=np.nan),
        [
            df["1"]._column.default_na_value(),
            df["1"]._column.default_na_value(),
        ],
    )


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
