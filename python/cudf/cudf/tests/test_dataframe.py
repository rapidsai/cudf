# Copyright (c) 2018-2020, NVIDIA CORPORATION.
import array as arr
import io
import operator
import random
import re
import textwrap

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numba import cuda

import cudf as gd
from cudf.core._compat import PANDAS_GE_110
from cudf.core.column import column
from cudf.tests import utils
from cudf.tests.utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_eq,
    does_not_raise,
    gen_rand,
)


def test_init_via_list_of_tuples():
    data = [
        (5, "cats", "jump", np.nan),
        (2, "dogs", "dig", 7.5),
        (3, "cows", "moo", -2.1, "occasionally"),
    ]

    pdf = pd.DataFrame(data)
    gdf = gd.DataFrame(data)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("rows", [0, 1, 2, 100])
def test_init_via_list_of_empty_tuples(rows):
    data = [()] * rows

    pdf = pd.DataFrame(data)
    gdf = gd.DataFrame(data)

    assert_eq(pdf, gdf, check_like=True)


@pytest.mark.parametrize(
    "dict_of_series",
    [
        {"a": pd.Series([1.0, 2.0, 3.0])},
        {"a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6])},
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": pd.Series([1.0, 2.0, 4.0], index=[1, 2, 3]),
        },
        {"a": [1, 2, 3], "b": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6])},
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
            "b": pd.Series([1.0, 2.0, 4.0], index=["c", "d", "e"]),
        },
        {
            "a": pd.Series(
                ["a", "b", "c"],
                index=pd.MultiIndex.from_tuples([(1, 2), (1, 3), (2, 3)]),
            ),
            "b": pd.Series(
                ["a", " b", "d"],
                index=pd.MultiIndex.from_tuples([(1, 2), (1, 3), (2, 3)]),
            ),
        },
    ],
)
def test_init_from_series_align(dict_of_series):
    pdf = pd.DataFrame(dict_of_series)
    gdf = gd.DataFrame(dict_of_series)

    assert_eq(pdf, gdf)

    for key in dict_of_series:
        if isinstance(dict_of_series[key], pd.Series):
            dict_of_series[key] = gd.Series(dict_of_series[key])

    gdf = gd.DataFrame(dict_of_series)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    ("dict_of_series", "expectation"),
    [
        (
            {
                "a": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
                "b": pd.Series(["a", "b", "c"], index=[4, 5, 6]),
            },
            pytest.raises(
                ValueError, match="Cannot align indices with non-unique values"
            ),
        ),
        (
            {
                "a": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
                "b": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
            },
            does_not_raise(),
        ),
    ],
)
def test_init_from_series_align_nonunique(dict_of_series, expectation):
    with expectation:
        gdf = gd.DataFrame(dict_of_series)

    if expectation == does_not_raise():
        pdf = pd.DataFrame(dict_of_series)
        assert_eq(pdf, gdf)


def test_init_unaligned_with_index():
    pdf = pd.DataFrame(
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3]),
        },
        index=[7, 8, 9],
    )
    gdf = gd.DataFrame(
        {
            "a": gd.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": gd.Series([1.0, 2.0, 3.0], index=[1, 2, 3]),
        },
        index=[7, 8, 9],
    )

    assert_eq(pdf, gdf, check_dtype=False)


def test_series_basic():
    # Make series from buffer
    a1 = np.arange(10, dtype=np.float64)
    series = gd.Series(a1)
    assert len(series) == 10
    np.testing.assert_equal(series.to_array(), np.hstack([a1]))


def test_series_from_cupy_scalars():
    data = [0.1, 0.2, 0.3]
    data_np = np.array(data)
    data_cp = cupy.array(data)
    s_np = gd.Series([data_np[0], data_np[2]])
    s_cp = gd.Series([data_cp[0], data_cp[2]])
    assert_eq(s_np, s_cp)


@pytest.mark.parametrize("a", [[1, 2, 3], [1, 10, 30]])
@pytest.mark.parametrize("b", [[4, 5, 6], [-11, -100, 30]])
def test_append_index(a, b):

    df = pd.DataFrame()
    df["a"] = a
    df["b"] = b

    gdf = gd.DataFrame()
    gdf["a"] = a
    gdf["b"] = b

    # Check the default index after appending two columns(Series)
    expected = df.a.append(df.b)
    actual = gdf.a.append(gdf.b)

    assert len(expected) == len(actual)
    assert_eq(expected.index, actual.index)

    expected = df.a.append(df.b, ignore_index=True)
    actual = gdf.a.append(gdf.b, ignore_index=True)

    assert len(expected) == len(actual)
    assert_eq(expected.index, actual.index)


def test_series_init_none():

    # test for creating empty series
    # 1: without initializing
    sr1 = gd.Series()
    got = sr1.to_string()
    print(got)
    expect = "Series([], dtype: float64)"
    # values should match despite whitespace difference
    assert got.split() == expect.split()

    # 2: Using `None` as an initializer
    sr2 = gd.Series(None)
    got = sr2.to_string()
    print(got)
    expect = "Series([], dtype: float64)"
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_basic():
    np.random.seed(0)
    df = gd.DataFrame()

    # Populate with cuda memory
    df["keys"] = np.arange(10, dtype=np.float64)
    np.testing.assert_equal(df["keys"].to_array(), np.arange(10))
    assert len(df) == 10

    # Populate with numpy array
    rnd_vals = np.random.random(10)
    df["vals"] = rnd_vals
    np.testing.assert_equal(df["vals"].to_array(), rnd_vals)
    assert len(df) == 10
    assert tuple(df.columns) == ("keys", "vals")

    # Make another dataframe
    df2 = gd.DataFrame()
    df2["keys"] = np.array([123], dtype=np.float64)
    df2["vals"] = np.array([321], dtype=np.float64)

    # Concat
    df = gd.concat([df, df2])
    assert len(df) == 11

    hkeys = np.asarray(np.arange(10, dtype=np.float64).tolist() + [123])
    hvals = np.asarray(rnd_vals.tolist() + [321])

    np.testing.assert_equal(df["keys"].to_array(), hkeys)
    np.testing.assert_equal(df["vals"].to_array(), hvals)

    # As matrix
    mat = df.as_matrix()

    expect = np.vstack([hkeys, hvals]).T

    print(expect)
    print(mat)
    np.testing.assert_equal(mat, expect)

    # test dataframe with tuple name
    df_tup = gd.DataFrame()
    data = np.arange(10)
    df_tup[(1, "foobar")] = data
    np.testing.assert_equal(data, df_tup[(1, "foobar")].to_array())

    df = gd.DataFrame(pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]}))
    pdf = pd.DataFrame(pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]}))
    assert_eq(df, pdf)

    gdf = gd.DataFrame({"id": [0, 1], "val": [None, None]})
    gdf["val"] = gdf["val"].astype("int")

    assert gdf["val"].isnull().all()


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame({"a": range(10), "b": range(10, 20), "c": range(1, 11)}),
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5}
        ),
    ],
)
@pytest.mark.parametrize(
    "columns", [["a"], ["b"], "a", "b", ["a", "b"]],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_columns(pdf, columns, inplace):
    pdf = pdf.copy()
    gdf = gd.from_pandas(pdf)

    expected = pdf.drop(columns=columns, inplace=inplace)
    actual = gdf.drop(columns=columns, inplace=inplace)

    if inplace:
        expected = pdf
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame({"a": range(10), "b": range(10, 20), "c": range(1, 11)}),
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5}
        ),
    ],
)
@pytest.mark.parametrize(
    "labels",
    [[1], [0], 1, 5, [5, 9], pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_labels_axis_0(pdf, labels, inplace):
    pdf = pdf.copy()
    gdf = gd.from_pandas(pdf)

    expected = pdf.drop(labels=labels, axis=0, inplace=inplace)
    actual = gdf.drop(labels=labels, axis=0, inplace=inplace)

    if inplace:
        expected = pdf
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame({"a": range(10), "b": range(10, 20), "c": range(1, 11)}),
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5}
        ),
    ],
)
@pytest.mark.parametrize(
    "index",
    [[1], [0], 1, 5, [5, 9], pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_index(pdf, index, inplace):
    pdf = pdf.copy()
    gdf = gd.from_pandas(pdf)

    expected = pdf.drop(index=index, inplace=inplace)
    actual = gdf.drop(index=index, inplace=inplace)

    if inplace:
        expected = pdf
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5},
            index=pd.MultiIndex(
                levels=[
                    ["lama", "cow", "falcon"],
                    ["speed", "weight", "length"],
                ],
                codes=[
                    [0, 0, 0, 1, 1, 1, 2, 2, 2, 1],
                    [0, 1, 2, 0, 1, 2, 0, 1, 2, 1],
                ],
            ),
        )
    ],
)
@pytest.mark.parametrize(
    "index,level",
    [
        ("cow", 0),
        ("lama", 0),
        ("falcon", 0),
        ("speed", 1),
        ("weight", 1),
        ("length", 1),
        pytest.param(
            "cow",
            None,
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/36293"
            ),
        ),
        pytest.param(
            "lama",
            None,
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/36293"
            ),
        ),
        pytest.param(
            "falcon",
            None,
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/36293"
            ),
        ),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_multiindex(pdf, index, level, inplace):
    pdf = pdf.copy()
    gdf = gd.from_pandas(pdf)

    expected = pdf.drop(index=index, inplace=inplace, level=level)
    actual = gdf.drop(index=index, inplace=inplace, level=level)

    if inplace:
        expected = pdf
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame({"a": range(10), "b": range(10, 20), "c": range(1, 11)}),
        pd.DataFrame(
            {"a": range(10), "b": range(10, 20), "d": ["a", "v"] * 5}
        ),
    ],
)
@pytest.mark.parametrize(
    "labels", [["a"], ["b"], "a", "b", ["a", "b"]],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_labels_axis_1(pdf, labels, inplace):
    pdf = pdf.copy()
    gdf = gd.from_pandas(pdf)

    expected = pdf.drop(labels=labels, axis=1, inplace=inplace)
    actual = gdf.drop(labels=labels, axis=1, inplace=inplace)

    if inplace:
        expected = pdf
        actual = gdf

    assert_eq(expected, actual)


def test_dataframe_drop_error():
    df = gd.DataFrame({"a": [1], "b": [2], "c": [3]})
    with pytest.raises(KeyError, match="column 'd' does not exist"):
        df.drop(columns="d")

    with pytest.raises(KeyError, match="column 'd' does not exist"):
        df.drop(columns=["a", "d", "b"])

    with pytest.raises(ValueError, match="Cannot specify both"):
        df.drop("a", axis=1, columns="a")

    with pytest.raises(ValueError, match="Need to specify at least"):
        df.drop(axis=1)

    with pytest.raises(KeyError, match="One or more values not found in axis"):
        df.drop([2, 0])


def test_dataframe_drop_raises():
    df = gd.DataFrame(
        {"a": [1, 2, 3], "c": [10, 20, 30]}, index=["x", "y", "z"]
    )
    pdf = df.to_pandas()
    try:
        pdf.drop("p")
    except Exception as e:
        with pytest.raises(
            type(e), match="One or more values not found in axis"
        ):
            df.drop("p")
    else:
        raise AssertionError("Expected pdf.drop to fail")

    expect = pdf.drop("p", errors="ignore")
    actual = df.drop("p", errors="ignore")

    assert_eq(actual, expect)

    try:
        pdf.drop(columns="p")
    except Exception as e:
        with pytest.raises(type(e), match="column 'p' does not exist"):
            df.drop(columns="p")
    else:
        raise AssertionError("Expected pdf.drop to fail")

    expect = pdf.drop(columns="p", errors="ignore")
    actual = df.drop(columns="p", errors="ignore")

    assert_eq(actual, expect)

    try:
        pdf.drop(labels="p", axis=1)
    except Exception as e:
        with pytest.raises(type(e), match="column 'p' does not exist"):
            df.drop(labels="p", axis=1)
    else:
        raise AssertionError("Expected pdf.drop to fail")

    expect = pdf.drop(labels="p", axis=1, errors="ignore")
    actual = df.drop(labels="p", axis=1, errors="ignore")

    assert_eq(actual, expect)


def test_dataframe_column_add_drop_via_setitem():
    df = gd.DataFrame()
    data = np.asarray(range(10))
    df["a"] = data
    df["b"] = data
    assert tuple(df.columns) == ("a", "b")
    del df["a"]
    assert tuple(df.columns) == ("b",)
    df["c"] = data
    assert tuple(df.columns) == ("b", "c")
    df["a"] = data
    assert tuple(df.columns) == ("b", "c", "a")


def test_dataframe_column_set_via_attr():
    data_0 = np.asarray([0, 2, 4, 5])
    data_1 = np.asarray([1, 4, 2, 3])
    data_2 = np.asarray([2, 0, 3, 0])
    df = gd.DataFrame({"a": data_0, "b": data_1, "c": data_2})

    for i in range(10):
        df.c = df.a
        assert assert_eq(df.c, df.a, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")

        df.c = df.b
        assert assert_eq(df.c, df.b, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")


def test_dataframe_column_drop_via_attr():
    df = gd.DataFrame({"a": []})

    with pytest.raises(AttributeError):
        del df.a

    assert tuple(df.columns) == tuple("a")


@pytest.mark.parametrize("axis", [0, "index"])
def test_dataframe_index_rename(axis):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf = gd.DataFrame.from_pandas(pdf)

    expect = pdf.rename(mapper={1: 5, 2: 6}, axis=axis)
    got = gdf.rename(mapper={1: 5, 2: 6}, axis=axis)

    assert_eq(expect, got)

    expect = pdf.rename(index={1: 5, 2: 6})
    got = gdf.rename(index={1: 5, 2: 6})

    assert_eq(expect, got)

    expect = pdf.rename({1: 5, 2: 6})
    got = gdf.rename({1: 5, 2: 6})

    assert_eq(expect, got)

    # `pandas` can support indexes with mixed values. We throw a
    # `NotImplementedError`.
    with pytest.raises(NotImplementedError):
        got = gdf.rename(mapper={1: "x", 2: "y"}, axis=axis)


def test_dataframe_MI_rename():
    gdf = gd.DataFrame(
        {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)}
    )
    gdg = gdf.groupby(["a", "b"]).count()
    pdg = gdg.to_pandas()

    expect = pdg.rename(mapper={1: 5, 2: 6}, axis=0)
    got = gdg.rename(mapper={1: 5, 2: 6}, axis=0)

    assert_eq(expect, got)


@pytest.mark.parametrize("axis", [1, "columns"])
def test_dataframe_column_rename(axis):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf = gd.DataFrame.from_pandas(pdf)

    expect = pdf.rename(mapper=lambda name: 2 * name, axis=axis)
    got = gdf.rename(mapper=lambda name: 2 * name, axis=axis)

    assert_eq(expect, got)

    expect = pdf.rename(columns=lambda name: 2 * name)
    got = gdf.rename(columns=lambda name: 2 * name)

    assert_eq(expect, got)

    rename_mapper = {"a": "z", "b": "y", "c": "x"}
    expect = pdf.rename(columns=rename_mapper)
    got = gdf.rename(columns=rename_mapper)

    assert_eq(expect, got)

    gdf = gd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    rename_mapper = {"a": "z", "b": "z", "c": "z"}
    expect = gd.DataFrame({"z": [1, 2, 3], "z_1": [4, 5, 6], "z_2": [7, 8, 9]})
    got = gdf.rename(columns=rename_mapper)

    assert_eq(expect, got)


def test_dataframe_pop():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [7.0, 8.0, 9.0]}
    )
    gdf = gd.DataFrame.from_pandas(pdf)

    # Test non-existing column error
    with pytest.raises(KeyError) as raises:
        gdf.pop("fake_colname")
    raises.match("fake_colname")

    # check pop numeric column
    pdf_pop = pdf.pop("a")
    gdf_pop = gdf.pop("a")
    assert_eq(pdf_pop, gdf_pop)
    assert_eq(pdf, gdf)

    # check string column
    pdf_pop = pdf.pop("b")
    gdf_pop = gdf.pop("b")
    assert_eq(pdf_pop, gdf_pop)
    assert_eq(pdf, gdf)

    # check float column and empty dataframe
    pdf_pop = pdf.pop("c")
    gdf_pop = gdf.pop("c")
    assert_eq(pdf_pop, gdf_pop)
    assert_eq(pdf, gdf)

    # check empty dataframe edge case
    empty_pdf = pd.DataFrame(columns=["a", "b"])
    empty_gdf = gd.DataFrame(columns=["a", "b"])
    pb = empty_pdf.pop("b")
    gb = empty_gdf.pop("b")
    assert len(pb) == len(gb)
    assert empty_pdf.empty and empty_gdf.empty


@pytest.mark.parametrize("nelem", [0, 3, 100, 1000])
def test_dataframe_astype(nelem):
    df = gd.DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df["a"].dtype is np.dtype(np.int32)
    df["b"] = df["a"].astype(np.float32)
    assert df["b"].dtype is np.dtype(np.float32)
    np.testing.assert_equal(df["a"].to_array(), df["b"].to_array())


@pytest.mark.parametrize("nelem", [0, 100])
def test_index_astype(nelem):
    df = gd.DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df.index.dtype is np.dtype(np.int64)
    df.index = df.index.astype(np.float32)
    assert df.index.dtype is np.dtype(np.float32)
    df["a"] = df["a"].astype(np.float32)
    np.testing.assert_equal(df.index.to_array(), df["a"].to_array())
    df["b"] = df["a"]
    df = df.set_index("b")
    df["a"] = df["a"].astype(np.int16)
    df.index = df.index.astype(np.int16)
    np.testing.assert_equal(df.index.to_array(), df["a"].to_array())


def test_dataframe_to_string():
    pd.options.display.max_rows = 5
    pd.options.display.max_columns = 8
    # Test basic
    df = gd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]})
    string = str(df)
    print(string)
    assert string.splitlines()[-1] == "[6 rows x 2 columns]"

    # Test skipped columns
    df = gd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [11, 12, 13, 14, 15, 16],
            "c": [11, 12, 13, 14, 15, 16],
            "d": [11, 12, 13, 14, 15, 16],
        }
    )
    string = df.to_string()
    print(string)
    assert string.splitlines()[-1] == "[6 rows x 4 columns]"

    # Test masked
    df = gd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]})

    data = np.arange(6)
    mask = np.zeros(1, dtype=gd.utils.utils.mask_dtype)
    mask[0] = 0b00101101

    masked = gd.Series.from_masked_array(data, mask)
    assert masked.null_count == 2
    df["c"] = masked

    # check data
    values = masked.copy()
    validids = [0, 2, 3, 5]
    densearray = masked.to_array()
    np.testing.assert_equal(data[validids], densearray)
    # valid position is corret
    for i in validids:
        assert data[i] == values[i]
    # null position is correct
    for i in range(len(values)):
        if i not in validids:
            assert values[i] is None

    pd.options.display.max_rows = 10
    got = df.to_string()
    print(got)
    expect = """
a b  c
0 1 11 0
1 2 12 <NA>
2 3 13 2
3 4 14 3
4 5 15 <NA>
5 6 16 5
"""
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_to_string_wide():
    # Test basic
    df = gd.DataFrame()
    for i in range(100):
        df["a{}".format(i)] = list(range(3))
    pd.options.display.max_columns = 0
    got = df.to_string()
    print(got)
    expect = """
    a0  a1  a2  a3  a4  a5  a6  a7 ...  a92 a93 a94 a95 a96 a97 a98 a99
0    0   0   0   0   0   0   0   0 ...    0   0   0   0   0   0   0   0
1    1   1   1   1   1   1   1   1 ...    1   1   1   1   1   1   1   1
2    2   2   2   2   2   2   2   2 ...    2   2   2   2   2   2   2   2
[3 rows x 100 columns]
"""
    # values should match despite whitespace difference

    assert got.split() == expect.split()


def test_dataframe_empty_to_string():
    # Test for printing empty dataframe
    df = gd.DataFrame()
    got = df.to_string()
    print(got)
    expect = "Empty DataFrame\nColumns: []\nIndex: []\n"
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_emptycolumns_to_string():
    # Test for printing dataframe having empty columns
    df = gd.DataFrame()
    df["a"] = []
    df["b"] = []
    got = df.to_string()
    print(got)
    expect = "Empty DataFrame\nColumns: [a, b]\nIndex: []\n"
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_copy():
    # Test for copying the dataframe using python copy pkg
    from copy import copy

    df = gd.DataFrame()
    df["a"] = [1, 2, 3]
    df2 = copy(df)
    df2["b"] = [4, 5, 6]
    got = df.to_string()
    print(got)
    expect = """
     a
0    1
1    2
2    3
"""
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_copy_shallow():
    # Test for copy dataframe using class method
    df = gd.DataFrame()
    df["a"] = [1, 2, 3]
    df2 = df.copy()
    df2["b"] = [4, 2, 3]
    got = df.to_string()
    print(got)
    expect = """
     a
0    1
1    2
2    3
"""
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_dtypes():
    dtypes = pd.Series(
        [np.int32, np.float32, np.float64], index=["c", "a", "b"]
    )
    df = gd.DataFrame({k: np.ones(10, dtype=v) for k, v in dtypes.iteritems()})
    assert df.dtypes.equals(dtypes)


def test_dataframe_add_col_to_object_dataframe():
    # Test for adding column to an empty object dataframe
    cols = ["a", "b", "c"]
    df = pd.DataFrame(columns=cols, dtype="str")

    data = {k: v for (k, v) in zip(cols, [["a"] for _ in cols])}

    gdf = gd.DataFrame(data)
    gdf = gdf[:0]

    assert gdf.dtypes.equals(df.dtypes)
    gdf["a"] = [1]
    df["a"] = [10]
    assert gdf.dtypes.equals(df.dtypes)
    gdf["b"] = [1.0]
    df["b"] = [10.0]
    assert gdf.dtypes.equals(df.dtypes)


def test_dataframe_dir_and_getattr():
    df = gd.DataFrame(
        {
            "a": np.ones(10),
            "b": np.ones(10),
            "not an id": np.ones(10),
            "oop$": np.ones(10),
        }
    )
    o = dir(df)
    assert {"a", "b"}.issubset(o)
    assert "not an id" not in o
    assert "oop$" not in o

    # Getattr works
    assert df.a.equals(df["a"])
    assert df.b.equals(df["b"])
    with pytest.raises(AttributeError):
        df.not_a_column


@pytest.mark.parametrize("order", ["C", "F"])
def test_empty_dataframe_as_gpu_matrix(order):
    df = gd.DataFrame()

    # Check fully empty dataframe.
    mat = df.as_gpu_matrix(order=order).copy_to_host()
    assert mat.shape == (0, 0)

    df = gd.DataFrame()
    nelem = 123
    for k in "abc":
        df[k] = np.random.random(nelem)

    # Check all columns in empty dataframe.
    mat = df.head(0).as_gpu_matrix(order=order).copy_to_host()
    assert mat.shape == (0, 3)


@pytest.mark.parametrize("order", ["C", "F"])
def test_dataframe_as_gpu_matrix(order):
    df = gd.DataFrame()

    nelem = 123
    for k in "abcd":
        df[k] = np.random.random(nelem)

    # Check all columns
    mat = df.as_gpu_matrix(order=order).copy_to_host()
    assert mat.shape == (nelem, 4)
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(df[k].to_array(), mat[:, i])

    # Check column subset
    mat = df.as_gpu_matrix(order=order, columns=["a", "c"]).copy_to_host()
    assert mat.shape == (nelem, 2)

    for i, k in enumerate("ac"):
        np.testing.assert_array_equal(df[k].to_array(), mat[:, i])


def test_dataframe_as_gpu_matrix_null_values():
    df = gd.DataFrame()

    nelem = 123
    na = -10000

    refvalues = {}
    for k in "abcd":
        df[k] = data = np.random.random(nelem)
        bitmask = utils.random_bitmask(nelem)
        df[k] = df[k].set_mask(bitmask)
        boolmask = np.asarray(
            utils.expand_bits_to_bytes(bitmask)[:nelem], dtype=np.bool_
        )
        data[~boolmask] = na
        refvalues[k] = data

    # Check null value causes error
    with pytest.raises(ValueError) as raises:
        df.as_gpu_matrix()
    raises.match("column 'a' has null values")

    for k in df.columns:
        df[k] = df[k].fillna(na)

    mat = df.as_gpu_matrix().copy_to_host()
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(refvalues[k], mat[:, i])


def test_dataframe_append_empty():
    pdf = pd.DataFrame(
        {
            "key": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    gdf = gd.DataFrame.from_pandas(pdf)

    gdf["newcol"] = 100
    pdf["newcol"] = 100

    assert len(gdf["newcol"]) == len(pdf)
    assert len(pdf["newcol"]) == len(pdf)
    assert_eq(gdf, pdf)


def test_dataframe_setitem_from_masked_object():
    ary = np.random.randn(100)
    mask = np.zeros(100, dtype=bool)
    mask[:20] = True
    np.random.shuffle(mask)
    ary[mask] = np.nan

    test1_null = gd.Series(ary, nan_as_null=True)
    assert test1_null.nullable
    assert test1_null.null_count == 20
    test1_nan = gd.Series(ary, nan_as_null=False)
    assert test1_nan.null_count == 0

    test2_null = gd.DataFrame.from_pandas(
        pd.DataFrame({"a": ary}), nan_as_null=True
    )
    assert test2_null["a"].nullable
    assert test2_null["a"].null_count == 20
    test2_nan = gd.DataFrame.from_pandas(
        pd.DataFrame({"a": ary}), nan_as_null=False
    )
    assert test2_nan["a"].null_count == 0

    gpu_ary = cupy.asarray(ary)
    test3_null = gd.Series(gpu_ary, nan_as_null=True)
    assert test3_null.nullable
    assert test3_null.null_count == 20
    test3_nan = gd.Series(gpu_ary, nan_as_null=False)
    assert test3_nan.null_count == 0

    test4 = gd.DataFrame()
    lst = [1, 2, None, 4, 5, 6, None, 8, 9]
    test4["lst"] = lst
    assert test4["lst"].nullable
    assert test4["lst"].null_count == 2


def test_dataframe_append_to_empty():
    pdf = pd.DataFrame()
    pdf["a"] = []
    pdf["b"] = [1, 2, 3]

    gdf = gd.DataFrame()
    gdf["a"] = []
    gdf["b"] = [1, 2, 3]

    assert_eq(gdf, pdf)


def test_dataframe_setitem_index_len1():
    gdf = gd.DataFrame()
    gdf["a"] = [1]
    gdf["b"] = gdf.index._values

    np.testing.assert_equal(gdf.b.to_array(), [0])


def test_assign():
    gdf = gd.DataFrame({"x": [1, 2, 3]})
    gdf2 = gdf.assign(y=gdf.x + 1)
    assert list(gdf.columns) == ["x"]
    assert list(gdf2.columns) == ["x", "y"]

    np.testing.assert_equal(gdf2.y.to_array(), [2, 3, 4])


@pytest.mark.parametrize("nrows", [1, 8, 100, 1000])
def test_dataframe_hash_columns(nrows):
    gdf = gd.DataFrame()
    data = np.asarray(range(nrows))
    data[0] = data[-1]  # make first and last the same
    gdf["a"] = data
    gdf["b"] = gdf.a + 100
    out = gdf.hash_columns(["a", "b"])
    assert isinstance(out, cupy.ndarray)
    assert len(out) == nrows
    assert out.dtype == np.int32

    # Check default
    out_all = gdf.hash_columns()
    np.testing.assert_array_equal(cupy.asnumpy(out), cupy.asnumpy(out_all))

    # Check single column
    out_one = cupy.asnumpy(gdf.hash_columns(["a"]))
    # First matches last
    assert out_one[0] == out_one[-1]
    # Equivalent to the gd.Series.hash_values()
    np.testing.assert_array_equal(cupy.asnumpy(gdf.a.hash_values()), out_one)


@pytest.mark.parametrize("nrows", [3, 10, 100, 1000])
@pytest.mark.parametrize("nparts", [1, 2, 8, 13])
@pytest.mark.parametrize("nkeys", [1, 2])
def test_dataframe_hash_partition(nrows, nparts, nkeys):
    np.random.seed(123)
    gdf = gd.DataFrame()
    keycols = []
    for i in range(nkeys):
        keyname = "key{}".format(i)
        gdf[keyname] = np.random.randint(0, 7 - i, nrows)
        keycols.append(keyname)
    gdf["val1"] = np.random.randint(0, nrows * 2, nrows)

    got = gdf.partition_by_hash(keycols, nparts=nparts)
    # Must return a list
    assert isinstance(got, list)
    # Must have correct number of partitions
    assert len(got) == nparts
    # All partitions must be DataFrame type
    assert all(isinstance(p, gd.DataFrame) for p in got)
    # Check that all partitions have unique keys
    part_unique_keys = set()
    for p in got:
        if len(p):
            # Take rows of the keycolumns and build a set of the key-values
            unique_keys = set(map(tuple, p.as_matrix(columns=keycols)))
            # Ensure that none of the key-values have occurred in other groups
            assert not (unique_keys & part_unique_keys)
            part_unique_keys |= unique_keys
    assert len(part_unique_keys)


@pytest.mark.parametrize("nrows", [3, 10, 50])
def test_dataframe_hash_partition_masked_value(nrows):
    gdf = gd.DataFrame()
    gdf["key"] = np.arange(nrows)
    gdf["val"] = np.arange(nrows) + 100
    bitmask = utils.random_bitmask(nrows)
    bytemask = utils.expand_bits_to_bytes(bitmask)
    gdf["val"] = gdf["val"].set_mask(bitmask)
    parted = gdf.partition_by_hash(["key"], nparts=3)
    # Verify that the valid mask is correct
    for p in parted:
        df = p.to_pandas()
        for row in df.itertuples():
            valid = bool(bytemask[row.key])
            expected_value = row.key + 100 if valid else np.nan
            got_value = row.val
            assert (expected_value == got_value) or (
                np.isnan(expected_value) and np.isnan(got_value)
            )


@pytest.mark.parametrize("nrows", [3, 10, 50])
def test_dataframe_hash_partition_masked_keys(nrows):
    gdf = gd.DataFrame()
    gdf["key"] = np.arange(nrows)
    gdf["val"] = np.arange(nrows) + 100
    bitmask = utils.random_bitmask(nrows)
    bytemask = utils.expand_bits_to_bytes(bitmask)
    gdf["key"] = gdf["key"].set_mask(bitmask)
    parted = gdf.partition_by_hash(["key"], nparts=3, keep_index=False)
    # Verify that the valid mask is correct
    for p in parted:
        df = p.to_pandas()
        for row in df.itertuples():
            valid = bool(bytemask[row.val - 100])
            # val is key + 100
            expected_value = row.val - 100 if valid else np.nan
            got_value = row.key
            assert (expected_value == got_value) or (
                np.isnan(expected_value) and np.isnan(got_value)
            )


@pytest.mark.parametrize("keep_index", [True, False])
def test_dataframe_hash_partition_keep_index(keep_index):

    gdf = gd.DataFrame(
        {"val": [1, 2, 3, 4], "key": [3, 2, 1, 4]}, index=[4, 3, 2, 1]
    )

    expected_df1 = gd.DataFrame(
        {"val": [1], "key": [3]}, index=[4] if keep_index else None
    )
    expected_df2 = gd.DataFrame(
        {"val": [2, 3, 4], "key": [2, 1, 4]},
        index=[3, 2, 1] if keep_index else range(1, 4),
    )
    expected = [expected_df1, expected_df2]

    parts = gdf.partition_by_hash(["key"], nparts=2, keep_index=keep_index)

    for exp, got in zip(expected, parts):
        assert_eq(exp, got)


def test_dataframe_hash_partition_empty():
    gdf = gd.DataFrame({"val": [1, 2], "key": [3, 2]}, index=["a", "b"])
    parts = gdf.iloc[:0].partition_by_hash(["key"], nparts=3)
    assert len(parts) == 3
    for part in parts:
        assert_eq(gdf.iloc[:0], part)


@pytest.mark.parametrize("dtype1", utils.supported_numpy_dtypes)
@pytest.mark.parametrize("dtype2", utils.supported_numpy_dtypes)
def test_dataframe_concat_different_numerical_columns(dtype1, dtype2):
    df1 = pd.DataFrame(dict(x=pd.Series(np.arange(5)).astype(dtype1)))
    df2 = pd.DataFrame(dict(x=pd.Series(np.arange(5)).astype(dtype2)))
    if dtype1 != dtype2 and "datetime" in dtype1 or "datetime" in dtype2:
        with pytest.raises(ValueError):
            gd.concat([df1, df2])
    else:
        pres = pd.concat([df1, df2])
        gres = gd.concat([gd.from_pandas(df1), gd.from_pandas(df2)])
        assert_eq(gd.from_pandas(pres), gres)


def test_dataframe_concat_different_column_types():
    df1 = gd.Series([42], dtype=np.float)
    df2 = gd.Series(["a"], dtype="category")
    with pytest.raises(ValueError):
        gd.concat([df1, df2])

    df2 = gd.Series(["a string"])
    with pytest.raises(TypeError):
        gd.concat([df1, df2])


@pytest.mark.parametrize(
    "df_1", [gd.DataFrame({"a": [1, 2], "b": [1, 3]}), gd.DataFrame({})]
)
@pytest.mark.parametrize(
    "df_2", [gd.DataFrame({"a": [], "b": []}), gd.DataFrame({})]
)
def test_concat_empty_dataframe(df_1, df_2):

    got = gd.concat([df_1, df_2])
    expect = pd.concat([df_1.to_pandas(), df_2.to_pandas()], sort=False)

    # ignoring dtypes as pandas upcasts int to float
    # on concatenation with empty dataframes

    assert_eq(got, expect, check_dtype=False)


@pytest.mark.parametrize(
    "df1_d",
    [
        {"a": [1, 2], "b": [1, 2], "c": ["s1", "s2"], "d": [1.0, 2.0]},
        {"b": [1.9, 10.9], "c": ["s1", "s2"]},
        {"c": ["s1"], "b": [None], "a": [False]},
    ],
)
@pytest.mark.parametrize(
    "df2_d",
    [
        {"a": [1, 2, 3]},
        {"a": [1, None, 3], "b": [True, True, False], "c": ["s3", None, "s4"]},
        {"a": [], "b": []},
        {},
    ],
)
def test_concat_different_column_dataframe(df1_d, df2_d):
    got = gd.concat(
        [gd.DataFrame(df1_d), gd.DataFrame(df2_d), gd.DataFrame(df1_d)],
        sort=False,
    )

    expect = pd.concat(
        [pd.DataFrame(df1_d), pd.DataFrame(df2_d), pd.DataFrame(df1_d)],
        sort=False,
    )

    # numerical columns are upcasted to float in cudf.DataFrame.to_pandas()
    # casts nan to 0 in non-float numerical columns

    numeric_cols = got.dtypes[got.dtypes != "object"].index
    for col in numeric_cols:
        got[col] = got[col].astype(np.float64).fillna(np.nan)

    assert_eq(got, expect, check_dtype=False)


@pytest.mark.parametrize("ser_1", [pd.Series([1, 2, 3]), pd.Series([])])
@pytest.mark.parametrize("ser_2", [pd.Series([])])
def test_concat_empty_series(ser_1, ser_2):
    got = gd.concat([gd.Series(ser_1), gd.Series(ser_2)])
    expect = pd.concat([ser_1, ser_2])

    assert_eq(got, expect)


def test_concat_with_axis():
    df1 = pd.DataFrame(dict(x=np.arange(5), y=np.arange(5)))
    df2 = pd.DataFrame(dict(a=np.arange(5), b=np.arange(5)))

    concat_df = pd.concat([df1, df2], axis=1)
    cdf1 = gd.from_pandas(df1)
    cdf2 = gd.from_pandas(df2)

    # concat only dataframes
    concat_cdf = gd.concat([cdf1, cdf2], axis=1)
    assert_eq(concat_cdf, concat_df)

    # concat only series
    concat_s = pd.concat([df1.x, df1.y], axis=1)
    cs1 = gd.Series.from_pandas(df1.x)
    cs2 = gd.Series.from_pandas(df1.y)
    concat_cdf_s = gd.concat([cs1, cs2], axis=1)

    assert_eq(concat_cdf_s, concat_s)

    # concat series and dataframes
    s3 = pd.Series(np.random.random(5))
    cs3 = gd.Series.from_pandas(s3)

    concat_cdf_all = gd.concat([cdf1, cs3, cdf2], axis=1)
    concat_df_all = pd.concat([df1, s3, df2], axis=1)
    assert_eq(concat_cdf_all, concat_df_all)

    # concat manual multi index
    midf1 = gd.from_pandas(df1)
    midf1.index = gd.MultiIndex(
        levels=[[0, 1, 2, 3], [0, 1]], codes=[[0, 1, 2, 3, 2], [0, 1, 0, 1, 0]]
    )
    midf2 = midf1[2:]
    midf2.index = gd.MultiIndex(
        levels=[[3, 4, 5], [2, 0]], codes=[[0, 1, 2], [1, 0, 1]]
    )
    mipdf1 = midf1.to_pandas()
    mipdf2 = midf2.to_pandas()

    assert_eq(gd.concat([midf1, midf2]), pd.concat([mipdf1, mipdf2]))
    assert_eq(gd.concat([midf2, midf1]), pd.concat([mipdf2, mipdf1]))
    assert_eq(
        gd.concat([midf1, midf2, midf1]), pd.concat([mipdf1, mipdf2, mipdf1])
    )

    # concat groupby multi index
    gdf1 = gd.DataFrame(
        {
            "x": np.random.randint(0, 10, 10),
            "y": np.random.randint(0, 10, 10),
            "z": np.random.randint(0, 10, 10),
            "v": np.random.randint(0, 10, 10),
        }
    )
    gdf2 = gdf1[5:]
    gdg1 = gdf1.groupby(["x", "y"]).min()
    gdg2 = gdf2.groupby(["x", "y"]).min()
    pdg1 = gdg1.to_pandas()
    pdg2 = gdg2.to_pandas()

    assert_eq(gd.concat([gdg1, gdg2]), pd.concat([pdg1, pdg2]))
    assert_eq(gd.concat([gdg2, gdg1]), pd.concat([pdg2, pdg1]))

    # series multi index concat
    gdgz1 = gdg1.z
    gdgz2 = gdg2.z
    pdgz1 = gdgz1.to_pandas()
    pdgz2 = gdgz2.to_pandas()

    assert_eq(gd.concat([gdgz1, gdgz2]), pd.concat([pdgz1, pdgz2]))
    assert_eq(gd.concat([gdgz2, gdgz1]), pd.concat([pdgz2, pdgz1]))


@pytest.mark.parametrize("nrows", [0, 3, 10, 100, 1000])
def test_nonmatching_index_setitem(nrows):
    np.random.seed(0)

    gdf = gd.DataFrame()
    gdf["a"] = np.random.randint(2147483647, size=nrows)
    gdf["b"] = np.random.randint(2147483647, size=nrows)
    gdf = gdf.set_index("b")

    test_values = np.random.randint(2147483647, size=nrows)
    gdf["c"] = test_values
    assert len(test_values) == len(gdf["c"])
    assert (
        gdf["c"]
        .to_pandas()
        .equals(gd.Series(test_values).set_index(gdf._index).to_pandas())
    )


def test_from_pandas():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])
    gdf = gd.DataFrame.from_pandas(df)
    assert isinstance(gdf, gd.DataFrame)

    assert_eq(df, gdf)

    s = df.x
    gs = gd.Series.from_pandas(s)
    assert isinstance(gs, gd.Series)

    assert_eq(s, gs)


@pytest.mark.parametrize("dtypes", [int, float])
def test_from_records(dtypes):
    h_ary = np.ndarray(shape=(10, 4), dtype=dtypes)
    rec_ary = h_ary.view(np.recarray)

    gdf = gd.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    df = pd.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    assert isinstance(gdf, gd.DataFrame)
    assert_eq(df, gdf)

    gdf = gd.DataFrame.from_records(rec_ary)
    df = pd.DataFrame.from_records(rec_ary)
    assert isinstance(gdf, gd.DataFrame)
    assert_eq(df, gdf)


@pytest.mark.parametrize("columns", [None, ["first", "second", "third"]])
@pytest.mark.parametrize(
    "index",
    [
        None,
        ["first", "second"],
        "name",
        "age",
        "weight",
        [10, 11],
        ["abc", "xyz"],
    ],
)
def test_from_records_index(columns, index):
    rec_ary = np.array(
        [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
        dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
    )
    gdf = gd.DataFrame.from_records(rec_ary, columns=columns, index=index)
    df = pd.DataFrame.from_records(rec_ary, columns=columns, index=index)
    assert isinstance(gdf, gd.DataFrame)
    assert_eq(df, gdf)


def test_from_gpu_matrix():
    h_ary = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    d_ary = cupy.asarray(h_ary)

    gdf = gd.DataFrame.from_gpu_matrix(d_ary, columns=["a", "b", "c"])
    df = pd.DataFrame(h_ary, columns=["a", "b", "c"])
    assert isinstance(gdf, gd.DataFrame)

    assert_eq(df, gdf)

    gdf = gd.DataFrame.from_gpu_matrix(d_ary)
    df = pd.DataFrame(h_ary)
    assert isinstance(gdf, gd.DataFrame)

    assert_eq(df, gdf)

    gdf = gd.DataFrame.from_gpu_matrix(d_ary, index=["a", "b"])
    df = pd.DataFrame(h_ary, index=["a", "b"])
    assert isinstance(gdf, gd.DataFrame)

    assert_eq(df, gdf)

    gdf = gd.DataFrame.from_gpu_matrix(d_ary, index=0)
    df = pd.DataFrame(h_ary)
    df = df.set_index(keys=0, drop=False)
    assert isinstance(gdf, gd.DataFrame)

    assert_eq(df, gdf)

    gdf = gd.DataFrame.from_gpu_matrix(d_ary, index=1)
    df = pd.DataFrame(h_ary)
    df = df.set_index(keys=1, drop=False)
    assert isinstance(gdf, gd.DataFrame)

    assert_eq(df, gdf)


def test_from_gpu_matrix_wrong_dimensions():
    d_ary = cupy.empty((2, 3, 4), dtype=np.int32)
    with pytest.raises(
        ValueError, match="matrix dimension expected 2 but found 3"
    ):
        gd.DataFrame.from_gpu_matrix(d_ary)


def test_from_gpu_matrix_wrong_index():
    d_ary = cupy.empty((2, 3), dtype=np.int32)

    with pytest.raises(
        ValueError, match="index length expected 2 but found 1"
    ):
        gd.DataFrame.from_gpu_matrix(d_ary, index=["a"])

    with pytest.raises(KeyError):
        gd.DataFrame.from_gpu_matrix(d_ary, index="a")


@pytest.mark.xfail(reason="constructor does not coerce index inputs")
def test_index_in_dataframe_constructor():
    a = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])
    b = gd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])

    assert_eq(a, b)
    assert_eq(a.loc[4:], b.loc[4:])


dtypes = NUMERIC_TYPES + DATETIME_TYPES + ["bool"]


@pytest.mark.parametrize("nelem", [0, 2, 3, 100, 1000])
@pytest.mark.parametrize("data_type", dtypes)
def test_from_arrow(nelem, data_type):
    df = pd.DataFrame(
        {
            "a": np.random.randint(0, 1000, nelem).astype(data_type),
            "b": np.random.randint(0, 1000, nelem).astype(data_type),
        }
    )
    padf = pa.Table.from_pandas(
        df, preserve_index=False
    ).replace_schema_metadata(None)
    gdf = gd.DataFrame.from_arrow(padf)
    assert isinstance(gdf, gd.DataFrame)

    assert_eq(df, gdf)

    s = pa.Array.from_pandas(df.a)
    gs = gd.Series.from_arrow(s)
    assert isinstance(gs, gd.Series)

    # For some reason PyArrow to_pandas() converts to numpy array and has
    # better type compatibility
    np.testing.assert_array_equal(s.to_pandas(), gs.to_array())


@pytest.mark.parametrize("nelem", [0, 2, 3, 100, 1000])
@pytest.mark.parametrize("data_type", dtypes)
def test_to_arrow(nelem, data_type):
    df = pd.DataFrame(
        {
            "a": np.random.randint(0, 1000, nelem).astype(data_type),
            "b": np.random.randint(0, 1000, nelem).astype(data_type),
        }
    )
    gdf = gd.DataFrame.from_pandas(df)

    pa_df = pa.Table.from_pandas(
        df, preserve_index=False
    ).replace_schema_metadata(None)

    pa_gdf = gdf.to_arrow(preserve_index=False).replace_schema_metadata(None)

    assert isinstance(pa_gdf, pa.Table)
    assert pa.Table.equals(pa_df, pa_gdf)

    pa_s = pa.Array.from_pandas(df.a)
    pa_gs = gdf["a"].to_arrow()

    assert isinstance(pa_gs, pa.Array)
    assert pa.Array.equals(pa_s, pa_gs)

    pa_i = pa.Array.from_pandas(df.index)
    pa_gi = gdf.index.to_arrow()

    assert isinstance(pa_gi, pa.Array)
    assert pa.Array.equals(pa_i, pa_gi)


@pytest.mark.parametrize("data_type", dtypes)
def test_to_from_arrow_nulls(data_type):
    if data_type == "longlong":
        data_type = "int64"
    if data_type == "bool":
        s1 = pa.array([True, None, False, None, True], type=data_type)
    else:
        dtype = np.dtype(data_type)
        if dtype.type == np.datetime64:
            time_unit, _ = np.datetime_data(dtype)
            data_type = pa.timestamp(unit=time_unit)
        s1 = pa.array([1, None, 3, None, 5], type=data_type)
    gs1 = gd.Series.from_arrow(s1)
    assert isinstance(gs1, gd.Series)
    # We have 64B padded buffers for nulls whereas Arrow returns a minimal
    # number of bytes, so only check the first byte in this case
    np.testing.assert_array_equal(
        np.asarray(s1.buffers()[0]).view("u1")[0],
        gs1._column.mask_array_view.copy_to_host().view("u1")[0],
    )
    assert pa.Array.equals(s1, gs1.to_arrow())

    s2 = pa.array([None, None, None, None, None], type=data_type)
    gs2 = gd.Series.from_arrow(s2)
    assert isinstance(gs2, gd.Series)
    # We have 64B padded buffers for nulls whereas Arrow returns a minimal
    # number of bytes, so only check the first byte in this case
    np.testing.assert_array_equal(
        np.asarray(s2.buffers()[0]).view("u1")[0],
        gs2._column.mask_array_view.copy_to_host().view("u1")[0],
    )
    assert pa.Array.equals(s2, gs2.to_arrow())


def test_to_arrow_categorical():
    df = pd.DataFrame()
    df["a"] = pd.Series(["a", "b", "c"], dtype="category")
    gdf = gd.DataFrame.from_pandas(df)

    pa_df = pa.Table.from_pandas(
        df, preserve_index=False
    ).replace_schema_metadata(None)
    pa_gdf = gdf.to_arrow(preserve_index=False).replace_schema_metadata(None)

    assert isinstance(pa_gdf, pa.Table)
    assert pa.Table.equals(pa_df, pa_gdf)

    pa_s = pa.Array.from_pandas(df.a)
    pa_gs = gdf["a"].to_arrow()

    assert isinstance(pa_gs, pa.Array)
    assert pa.Array.equals(pa_s, pa_gs)


def test_from_arrow_missing_categorical():
    pd_cat = pd.Categorical(["a", "b", "c"], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = gd.Series(pa_cat)

    assert isinstance(gd_cat, gd.Series)
    assert_eq(
        pd.Series(pa_cat.to_pandas()),  # PyArrow returns a pd.Categorical
        gd_cat.to_pandas(),
    )


def test_to_arrow_missing_categorical():
    pd_cat = pd.Categorical(["a", "b", "c"], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = gd.Series(pa_cat)

    assert isinstance(gd_cat, gd.Series)
    assert pa.Array.equals(pa_cat, gd_cat.to_arrow())


@pytest.mark.parametrize("data_type", dtypes)
def test_from_scalar_typing(data_type):
    if data_type == "datetime64[ms]":
        scalar = (
            np.dtype("int64")
            .type(np.random.randint(0, 5))
            .astype("datetime64[ms]")
        )
    elif data_type.startswith("datetime64"):
        from datetime import date

        scalar = np.datetime64(date.today()).astype("datetime64[ms]")
        data_type = "datetime64[ms]"
    else:
        scalar = np.dtype(data_type).type(np.random.randint(0, 5))

    gdf = gd.DataFrame()
    gdf["a"] = [1, 2, 3, 4, 5]
    gdf["b"] = scalar
    assert gdf["b"].dtype == np.dtype(data_type)
    assert len(gdf["b"]) == len(gdf["a"])


@pytest.mark.parametrize("data_type", NUMERIC_TYPES)
def test_from_python_array(data_type):
    np_arr = np.random.randint(0, 100, 10).astype(data_type)
    data = memoryview(np_arr)
    data = arr.array(data.format, data)

    gs = gd.Series(data)

    np.testing.assert_equal(gs.to_array(), np_arr)


def test_series_shape():
    ps = pd.Series([1, 2, 3, 4])
    cs = gd.Series([1, 2, 3, 4])

    assert ps.shape == cs.shape


def test_series_shape_empty():
    ps = pd.Series()
    cs = gd.Series([])

    assert ps.shape == cs.shape


def test_dataframe_shape():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    gdf = gd.DataFrame.from_pandas(pdf)

    assert pdf.shape == gdf.shape


def test_dataframe_shape_empty():
    pdf = pd.DataFrame()
    gdf = gd.DataFrame()

    assert pdf.shape == gdf.shape


@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 20])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_dataframe_transpose(nulls, num_cols, num_rows, dtype):

    pdf = pd.DataFrame()
    from string import ascii_lowercase

    null_rep = np.nan if dtype in ["float32", "float64"] else None

    for i in range(num_cols):
        colname = ascii_lowercase[i]
        data = pd.Series(np.random.randint(0, 26, num_rows).astype(dtype))
        if nulls == "some":
            idx = np.random.choice(
                num_rows, size=int(num_rows / 2), replace=False
            )
            data[idx] = null_rep
        elif nulls == "all":
            data[:] = null_rep
        pdf[colname] = data

    gdf = gd.DataFrame.from_pandas(pdf)

    got_function = gdf.transpose()
    got_property = gdf.T

    expect = pdf.transpose()

    assert_eq(expect, got_function)
    assert_eq(expect, got_property)


@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 20])
def test_dataframe_transpose_category(num_cols, num_rows):
    pdf = pd.DataFrame()
    from string import ascii_lowercase

    for i in range(num_cols):
        colname = ascii_lowercase[i]
        data = pd.Series(list(ascii_lowercase), dtype="category")
        data = data.sample(num_rows, replace=True).reset_index(drop=True)
        pdf[colname] = data

    gdf = gd.DataFrame.from_pandas(pdf)

    got_function = gdf.transpose()
    got_property = gdf.T

    expect = pdf.transpose()

    assert_eq(expect, got_function.to_pandas())
    assert_eq(expect, got_property.to_pandas())


def test_generated_column():
    gdf = gd.DataFrame({"a": (i for i in range(5))})
    assert len(gdf) == 5


@pytest.fixture
def pdf():
    return pd.DataFrame({"x": range(10), "y": range(10)})


@pytest.fixture
def gdf(pdf):
    return gd.DataFrame.from_pandas(pdf)


@pytest.mark.parametrize(
    "data",
    [
        {"x": [np.nan, 2, 3, 4, 100, np.nan], "y": [4, 5, 6, 88, 99, np.nan]},
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan]},
        {"x": [], "y": []},
        {"x": []},
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        lambda df, **kwargs: df.min(**kwargs),
        lambda df, **kwargs: df.max(**kwargs),
        lambda df, **kwargs: df.sum(**kwargs),
        lambda df, **kwargs: df.product(**kwargs),
        lambda df, **kwargs: df.cummin(**kwargs),
        lambda df, **kwargs: df.cummax(**kwargs),
        lambda df, **kwargs: df.cumsum(**kwargs),
        lambda df, **kwargs: df.cumprod(**kwargs),
        lambda df, **kwargs: df.mean(**kwargs),
        lambda df, **kwargs: df.sum(**kwargs),
        lambda df, **kwargs: df.max(**kwargs),
        lambda df, **kwargs: df.std(ddof=1, **kwargs),
        lambda df, **kwargs: df.var(ddof=1, **kwargs),
        lambda df, **kwargs: df.std(ddof=2, **kwargs),
        lambda df, **kwargs: df.var(ddof=2, **kwargs),
        lambda df, **kwargs: df.kurt(**kwargs),
        lambda df, **kwargs: df.skew(**kwargs),
        lambda df, **kwargs: df.all(**kwargs),
        lambda df, **kwargs: df.any(**kwargs),
    ],
)
@pytest.mark.parametrize("skipna", [True, False, None])
def test_dataframe_reductions(data, func, skipna):
    pdf = pd.DataFrame(data=data)
    print(func(pdf, skipna=skipna))
    gdf = gd.DataFrame.from_pandas(pdf)
    print(func(gdf, skipna=skipna))
    assert_eq(func(pdf, skipna=skipna), func(gdf, skipna=skipna))


@pytest.mark.parametrize(
    "data",
    [
        {"x": [np.nan, 2, 3, 4, 100, np.nan], "y": [4, 5, 6, 88, 99, np.nan]},
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan]},
        {"x": [], "y": []},
        {"x": []},
    ],
)
@pytest.mark.parametrize("func", [lambda df: df.count()])
def test_dataframe_count_reduction(data, func):
    pdf = pd.DataFrame(data=data)
    gdf = gd.DataFrame.from_pandas(pdf)

    assert_eq(func(pdf), func(gdf))


@pytest.mark.parametrize(
    "data",
    [
        {"x": [np.nan, 2, 3, 4, 100, np.nan], "y": [4, 5, 6, 88, 99, np.nan]},
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan]},
        {"x": [], "y": []},
        {"x": []},
    ],
)
@pytest.mark.parametrize("ops", ["sum", "product", "prod"])
@pytest.mark.parametrize("skipna", [True, False, None])
@pytest.mark.parametrize("min_count", [-10, -1, 0, 1, 2, 3, 10])
def test_dataframe_min_count_ops(data, ops, skipna, min_count):
    psr = pd.DataFrame(data)
    gsr = gd.DataFrame(data)

    assert_eq(
        getattr(psr, ops)(skipna=skipna, min_count=min_count),
        getattr(gsr, ops)(skipna=skipna, min_count=min_count),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "binop",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.ne,
    ],
)
def test_binops_df(pdf, gdf, binop):
    pdf = pdf + 1.0
    gdf = gdf + 1.0
    d = binop(pdf, pdf)
    g = binop(gdf, gdf)
    assert_eq(d, g)


@pytest.mark.parametrize("binop", [operator.and_, operator.or_, operator.xor])
def test_bitwise_binops_df(pdf, gdf, binop):
    d = binop(pdf, pdf + 1)
    g = binop(gdf, gdf + 1)
    assert_eq(d, g)


@pytest.mark.parametrize(
    "binop",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.ne,
    ],
)
def test_binops_series(pdf, gdf, binop):
    pdf = pdf + 1.0
    gdf = gdf + 1.0
    d = binop(pdf.x, pdf.y)
    g = binop(gdf.x, gdf.y)
    assert_eq(d, g)


@pytest.mark.parametrize("binop", [operator.and_, operator.or_, operator.xor])
def test_bitwise_binops_series(pdf, gdf, binop):
    d = binop(pdf.x, pdf.y + 1)
    g = binop(gdf.x, gdf.y + 1)
    assert_eq(d, g)


@pytest.mark.parametrize("unaryop", [operator.neg, operator.inv, operator.abs])
def test_unaryops_df(pdf, gdf, unaryop):
    d = unaryop(pdf - 5)
    g = unaryop(gdf - 5)
    assert_eq(d, g)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.empty,
        lambda df: df.x.empty,
        lambda df: df.x.fillna(123, limit=None, method=None, axis=None),
        lambda df: df.drop("x", axis=1, errors="raise"),
    ],
)
def test_unary_operators(func, pdf, gdf):
    p = func(pdf)
    g = func(gdf)
    assert_eq(p, g)


def test_is_monotonic(gdf):
    pdf = pd.DataFrame({"x": [1, 2, 3]}, index=[3, 1, 2])
    gdf = gd.DataFrame.from_pandas(pdf)
    assert not gdf.index.is_monotonic
    assert not gdf.index.is_monotonic_increasing
    assert not gdf.index.is_monotonic_decreasing


def test_iter(pdf, gdf):
    assert list(pdf) == list(gdf)


def test_iteritems(gdf):
    for k, v in gdf.iteritems():
        assert k in gdf.columns
        assert isinstance(v, gd.Series)
        assert_eq(v, gdf[k])


@pytest.mark.parametrize("q", [0.5, 1, 0.001, [0.5], [], [0.005, 0.5, 1]])
def test_quantile(pdf, gdf, q):
    assert_eq(pdf["x"].quantile(q), gdf["x"].quantile(q))
    assert_eq(pdf.quantile(q), gdf.quantile(q))


def test_empty_quantile():
    pdf = pd.DataFrame({"x": []})
    df = gd.DataFrame({"x": []})

    actual = df.quantile()
    expected = pdf.quantile()

    assert_eq(actual, expected)


def test_from_pandas_function(pdf):
    gdf = gd.from_pandas(pdf)
    assert isinstance(gdf, gd.DataFrame)
    assert_eq(pdf, gdf)

    gdf = gd.from_pandas(pdf.x)
    assert isinstance(gdf, gd.Series)
    assert_eq(pdf.x, gdf)

    with pytest.raises(TypeError):
        gd.from_pandas(123)


@pytest.mark.parametrize("preserve_index", [True, False])
def test_arrow_pandas_compat(pdf, gdf, preserve_index):
    pdf["z"] = range(10)
    pdf = pdf.set_index("z")
    gdf["z"] = range(10)
    gdf = gdf.set_index("z")

    pdf_arrow_table = pa.Table.from_pandas(pdf, preserve_index=preserve_index)
    gdf_arrow_table = gdf.to_arrow(preserve_index=preserve_index)

    assert pa.Table.equals(pdf_arrow_table, gdf_arrow_table)

    gdf2 = gd.DataFrame.from_arrow(pdf_arrow_table)
    pdf2 = pdf_arrow_table.to_pandas()

    assert_eq(pdf2, gdf2)


@pytest.mark.parametrize("nrows", [1, 8, 100, 1000, 100000])
def test_series_hash_encode(nrows):
    data = np.asarray(range(nrows))
    # Python hash returns different value which sometimes
    # results in enc_with_name_arr and enc_arr to be same.
    # And there is no other better way to make hash return same value.
    # So using an integer name to get constant value back from hash.
    s = gd.Series(data, name=1)
    num_features = 1000

    encoded_series = s.hash_encode(num_features)
    assert isinstance(encoded_series, gd.Series)
    enc_arr = encoded_series.to_array()
    assert np.all(enc_arr >= 0)
    assert np.max(enc_arr) < num_features

    enc_with_name_arr = s.hash_encode(num_features, use_name=True).to_array()
    assert enc_with_name_arr[0] != enc_arr[0]


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["bool"])
def test_cuda_array_interface(dtype):

    np_data = np.arange(10).astype(dtype)
    cupy_data = cupy.array(np_data)
    pd_data = pd.Series(np_data)

    cudf_data = gd.Series(cupy_data)
    assert_eq(pd_data, cudf_data)

    gdf = gd.DataFrame()
    gdf["test"] = cupy_data
    pd_data.name = "test"
    assert_eq(pd_data, gdf["test"])


@pytest.mark.parametrize("nelem", [0, 2, 3, 100])
@pytest.mark.parametrize("nchunks", [1, 2, 5, 10])
@pytest.mark.parametrize("data_type", dtypes)
def test_from_arrow_chunked_arrays(nelem, nchunks, data_type):
    np_list_data = [
        np.random.randint(0, 100, nelem).astype(data_type)
        for i in range(nchunks)
    ]
    pa_chunk_array = pa.chunked_array(np_list_data)

    expect = pd.Series(pa_chunk_array.to_pandas())
    got = gd.Series(pa_chunk_array)

    assert_eq(expect, got)

    np_list_data2 = [
        np.random.randint(0, 100, nelem).astype(data_type)
        for i in range(nchunks)
    ]
    pa_chunk_array2 = pa.chunked_array(np_list_data2)
    pa_table = pa.Table.from_arrays(
        [pa_chunk_array, pa_chunk_array2], names=["a", "b"]
    )

    expect = pa_table.to_pandas()
    got = gd.DataFrame.from_arrow(pa_table)

    assert_eq(expect, got)


@pytest.mark.skip(reason="Test was designed to be run in isolation")
def test_gpu_memory_usage_with_boolmask():
    import cudf

    ctx = cuda.current_context()

    def query_GPU_memory(note=""):
        memInfo = ctx.get_memory_info()
        usedMemoryGB = (memInfo.total - memInfo.free) / 1e9
        return usedMemoryGB

    cuda.current_context().deallocations.clear()
    nRows = int(1e8)
    nCols = 2
    dataNumpy = np.asfortranarray(np.random.rand(nRows, nCols))
    colNames = ["col" + str(iCol) for iCol in range(nCols)]
    pandasDF = pd.DataFrame(data=dataNumpy, columns=colNames, dtype=np.float32)
    cudaDF = cudf.core.DataFrame.from_pandas(pandasDF)
    boolmask = gd.Series(np.random.randint(1, 2, len(cudaDF)).astype("bool"))

    memory_used = query_GPU_memory()
    cudaDF = cudaDF[boolmask]

    assert (
        cudaDF.index._values.data_array_view.device_ctypes_pointer
        == cudaDF["col0"].index._values.data_array_view.device_ctypes_pointer
    )
    assert (
        cudaDF.index._values.data_array_view.device_ctypes_pointer
        == cudaDF["col1"].index._values.data_array_view.device_ctypes_pointer
    )

    assert memory_used == query_GPU_memory()


def test_boolmask(pdf, gdf):
    boolmask = np.random.randint(0, 2, len(pdf)) > 0
    gdf = gdf[boolmask]
    pdf = pdf[boolmask]
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "mask_shape",
    [
        (2, "ab"),
        (2, "abc"),
        (3, "ab"),
        (3, "abc"),
        (3, "abcd"),
        (4, "abc"),
        (4, "abcd"),
    ],
)
def test_dataframe_boolmask(mask_shape):
    pdf = pd.DataFrame()
    for col in "abc":
        pdf[col] = np.random.randint(0, 10, 3)
    pdf_mask = pd.DataFrame()
    for col in mask_shape[1]:
        pdf_mask[col] = np.random.randint(0, 2, mask_shape[0]) > 0
    gdf = gd.DataFrame.from_pandas(pdf)
    gdf_mask = gd.DataFrame.from_pandas(pdf_mask)
    gdf = gdf[gdf_mask]
    pdf = pdf[pdf_mask]

    assert np.array_equal(gdf.columns, pdf.columns)
    for col in gdf.columns:
        assert np.array_equal(
            gdf[col].fillna(-1).to_pandas().values, pdf[col].fillna(-1).values
        )


@pytest.mark.parametrize(
    "mask",
    [
        [True, False, True],
        pytest.param(
            gd.Series([True, False, True]),
            marks=pytest.mark.xfail(
                reason="Pandas can't index a multiindex with a Series"
            ),
        ),
    ],
)
def test_dataframe_multiindex_boolmask(mask):
    gdf = gd.DataFrame(
        {"w": [3, 2, 1], "x": [1, 2, 3], "y": [0, 1, 0], "z": [1, 1, 1]}
    )
    gdg = gdf.groupby(["w", "x"]).count()
    pdg = gdg.to_pandas()
    assert_eq(gdg[mask], pdg[mask])


def test_dataframe_assignment():
    pdf = pd.DataFrame()
    for col in "abc":
        pdf[col] = np.array([0, 1, 1, -2, 10])
    gdf = gd.DataFrame.from_pandas(pdf)
    gdf[gdf < 0] = 999
    pdf[pdf < 0] = 999
    assert_eq(gdf, pdf)


def test_1row_arrow_table():
    data = [pa.array([0]), pa.array([1])]
    batch = pa.RecordBatch.from_arrays(data, ["f0", "f1"])
    table = pa.Table.from_batches([batch])

    expect = table.to_pandas()
    got = gd.DataFrame.from_arrow(table)
    assert_eq(expect, got)


def test_arrow_handle_no_index_name(pdf, gdf):
    gdf_arrow = gdf.to_arrow()
    pdf_arrow = pa.Table.from_pandas(pdf)
    assert pa.Table.equals(pdf_arrow, gdf_arrow)

    got = gd.DataFrame.from_arrow(gdf_arrow)
    expect = pdf_arrow.to_pandas()
    assert_eq(expect, got)


@pytest.mark.parametrize("num_rows", [1, 3, 10, 100])
@pytest.mark.parametrize("num_bins", [1, 2, 4, 20])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["bool"])
def test_series_digitize(num_rows, num_bins, right, dtype):
    data = np.random.randint(0, 100, num_rows).astype(dtype)
    bins = np.unique(np.sort(np.random.randint(2, 95, num_bins).astype(dtype)))
    s = gd.Series(data)
    indices = s.digitize(bins, right)
    np.testing.assert_array_equal(
        np.digitize(data, bins, right), indices.to_array()
    )


def test_pandas_non_contiguious():
    arr1 = np.random.sample([5000, 10])
    assert arr1.flags["C_CONTIGUOUS"] is True
    df = pd.DataFrame(arr1)
    for col in df.columns:
        assert df[col].values.flags["C_CONTIGUOUS"] is False

    gdf = gd.DataFrame.from_pandas(df)
    assert_eq(gdf.to_pandas(), df)


@pytest.mark.parametrize("num_elements", [0, 2, 10, 100])
@pytest.mark.parametrize("null_type", [np.nan, None, "mixed"])
def test_series_all_null(num_elements, null_type):
    if null_type == "mixed":
        data = []
        data1 = [np.nan] * int(num_elements / 2)
        data2 = [None] * int(num_elements / 2)
        for idx in range(len(data1)):
            data.append(data1[idx])
            data.append(data2[idx])
    else:
        data = [null_type] * num_elements

    # Typecast Pandas because None will return `object` dtype
    expect = pd.Series(data).astype("float64")
    got = gd.Series(data)

    assert_eq(expect, got)


@pytest.mark.parametrize("num_elements", [0, 2, 10, 100])
def test_series_all_valid_nan(num_elements):
    data = [np.nan] * num_elements
    sr = gd.Series(data, nan_as_null=False)
    np.testing.assert_equal(sr.null_count, 0)


def test_series_rename():
    pds = pd.Series([1, 2, 3], name="asdf")
    gds = gd.Series([1, 2, 3], name="asdf")

    expect = pds.rename("new_name")
    got = gds.rename("new_name")

    assert_eq(expect, got)

    pds = pd.Series(expect)
    gds = gd.Series(got)

    assert_eq(pds, gds)

    pds = pd.Series(expect, name="name name")
    gds = gd.Series(got, name="name name")

    assert_eq(pds, gds)


@pytest.mark.parametrize("data_type", dtypes)
@pytest.mark.parametrize("nelem", [0, 100])
def test_head_tail(nelem, data_type):
    def check_index_equality(left, right):
        assert left.index.equals(right.index)

    def check_values_equality(left, right):
        if len(left) == 0 and len(right) == 0:
            return None

        np.testing.assert_array_equal(left.to_pandas(), right.to_pandas())

    def check_frame_series_equality(left, right):
        check_index_equality(left, right)
        check_values_equality(left, right)

    gdf = gd.DataFrame(
        {
            "a": np.random.randint(0, 1000, nelem).astype(data_type),
            "b": np.random.randint(0, 1000, nelem).astype(data_type),
        }
    )

    check_frame_series_equality(gdf.head(), gdf[:5])
    check_frame_series_equality(gdf.head(3), gdf[:3])
    check_frame_series_equality(gdf.head(-2), gdf[:-2])
    check_frame_series_equality(gdf.head(0), gdf[0:0])

    check_frame_series_equality(gdf["a"].head(), gdf["a"][:5])
    check_frame_series_equality(gdf["a"].head(3), gdf["a"][:3])
    check_frame_series_equality(gdf["a"].head(-2), gdf["a"][:-2])

    check_frame_series_equality(gdf.tail(), gdf[-5:])
    check_frame_series_equality(gdf.tail(3), gdf[-3:])
    check_frame_series_equality(gdf.tail(-2), gdf[2:])
    check_frame_series_equality(gdf.tail(0), gdf[0:0])

    check_frame_series_equality(gdf["a"].tail(), gdf["a"][-5:])
    check_frame_series_equality(gdf["a"].tail(3), gdf["a"][-3:])
    check_frame_series_equality(gdf["a"].tail(-2), gdf["a"][2:])


def test_tail_for_string():
    gdf = gd.DataFrame()
    gdf["id"] = gd.Series(["a", "b"], dtype=np.object)
    gdf["v"] = gd.Series([1, 2])
    assert_eq(gdf.tail(3), gdf.to_pandas().tail(3))


@pytest.mark.parametrize("drop", [True, False])
def test_reset_index(pdf, gdf, drop):
    assert_eq(
        pdf.reset_index(drop=drop, inplace=False),
        gdf.reset_index(drop=drop, inplace=False),
    )
    assert_eq(
        pdf.x.reset_index(drop=drop, inplace=False),
        gdf.x.reset_index(drop=drop, inplace=False),
    )


@pytest.mark.parametrize("drop", [True, False])
def test_reset_named_index(pdf, gdf, drop):
    pdf.index.name = "cudf"
    gdf.index.name = "cudf"
    assert_eq(
        pdf.reset_index(drop=drop, inplace=False),
        gdf.reset_index(drop=drop, inplace=False),
    )
    assert_eq(
        pdf.x.reset_index(drop=drop, inplace=False),
        gdf.x.reset_index(drop=drop, inplace=False),
    )


@pytest.mark.parametrize("drop", [True, False])
def test_reset_index_inplace(pdf, gdf, drop):
    pdf.reset_index(drop=drop, inplace=True)
    gdf.reset_index(drop=drop, inplace=True)
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "b", "c", "d", "e"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        "a",
        ["a", "b"],
        pd.CategoricalIndex(["I", "II", "III", "IV", "V"]),
        pd.Series(["h", "i", "k", "l", "m"]),
        ["b", pd.Index(["I", "II", "III", "IV", "V"])],
        ["c", [11, 12, 13, 14, 15]],
        pd.MultiIndex(
            levels=[
                ["I", "II", "III", "IV", "V"],
                ["one", "two", "three", "four", "five"],
            ],
            codes=[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
            names=["col1", "col2"],
        ),
        pd.RangeIndex(0, 5),  # corner case
        [pd.Series(["h", "i", "k", "l", "m"]), pd.RangeIndex(0, 5)],
        [
            pd.MultiIndex(
                levels=[
                    ["I", "II", "III", "IV", "V"],
                    ["one", "two", "three", "four", "five"],
                ],
                codes=[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
                names=["col1", "col2"],
            ),
            pd.RangeIndex(0, 5),
        ],
    ],
)
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_set_index(data, index, drop, append, inplace):
    gdf = gd.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = pdf.set_index(index, inplace=inplace, drop=drop, append=append)
    actual = gdf.set_index(index, inplace=inplace, drop=drop, append=append)

    if inplace:
        expected = pdf
        actual = gdf
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        {
            "a": [1, 1, 2, 2, 5],
            "b": ["a", "b", "c", "d", "e"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ],
)
@pytest.mark.parametrize("index", ["a", pd.Index([1, 1, 2, 2, 3])])
@pytest.mark.parametrize("verify_integrity", [True])
@pytest.mark.xfail
def test_set_index_verify_integrity(data, index, verify_integrity):
    gdf = gd.DataFrame(data)
    gdf.set_index(index, verify_integrity=verify_integrity)


@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("nelem", [10, 200, 1333])
def test_set_index_multi(drop, nelem):
    np.random.seed(0)
    a = np.arange(nelem)
    np.random.shuffle(a)
    df = pd.DataFrame(
        {
            "a": a,
            "b": np.random.randint(0, 4, size=nelem),
            "c": np.random.uniform(low=0, high=4, size=nelem),
            "d": np.random.choice(["green", "black", "white"], nelem),
        }
    )
    df["e"] = df["d"].astype("category")
    gdf = gd.DataFrame.from_pandas(df)

    assert_eq(gdf.set_index("a", drop=drop), gdf.set_index(["a"], drop=drop))
    assert_eq(
        df.set_index(["b", "c"], drop=drop),
        gdf.set_index(["b", "c"], drop=drop),
    )
    assert_eq(
        df.set_index(["d", "b"], drop=drop),
        gdf.set_index(["d", "b"], drop=drop),
    )
    assert_eq(
        df.set_index(["b", "d", "e"], drop=drop),
        gdf.set_index(["b", "d", "e"], drop=drop),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_0(copy):
    # TODO (ptaylor): pandas changes `int` dtype to `float64`
    # when reindexing and filling new label indices with NaN
    gdf = gd.datasets.randomdata(
        nrows=6,
        dtypes={
            "a": "category",
            # 'b': int,
            "c": float,
            "d": str,
        },
    )
    pdf = gdf.to_pandas()
    # Validate reindex returns a copy unmodified
    assert_eq(pdf.reindex(copy=True), gdf.reindex(copy=copy))


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_1(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate labels are used as index when axis defaults to 0
    assert_eq(pdf.reindex(index, copy=True), gdf.reindex(index, copy=copy))


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_2(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate labels are used as index when axis=0
    assert_eq(
        pdf.reindex(index, axis=0, copy=True),
        gdf.reindex(index, axis=0, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_3(copy):
    columns = ["a", "b", "c", "d", "e"]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate labels are used as columns when axis=0
    assert_eq(
        pdf.reindex(columns, axis=1, copy=True),
        gdf.reindex(columns, axis=1, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_4(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate labels are used as index when axis=0
    assert_eq(
        pdf.reindex(labels=index, axis=0, copy=True),
        gdf.reindex(labels=index, axis=0, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_5(copy):
    columns = ["a", "b", "c", "d", "e"]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate labels are used as columns when axis=1
    assert_eq(
        pdf.reindex(labels=columns, axis=1, copy=True),
        gdf.reindex(labels=columns, axis=1, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_6(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate labels are used as index when axis='index'
    assert_eq(
        pdf.reindex(labels=index, axis="index", copy=True),
        gdf.reindex(labels=index, axis="index", copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_7(copy):
    columns = ["a", "b", "c", "d", "e"]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate labels are used as columns when axis='columns'
    assert_eq(
        pdf.reindex(labels=columns, axis="columns", copy=True),
        gdf.reindex(labels=columns, axis="columns", copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_8(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate reindexes labels when index=labels
    assert_eq(
        pdf.reindex(index=index, copy=True),
        gdf.reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_9(copy):
    columns = ["a", "b", "c", "d", "e"]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate reindexes column names when columns=labels
    assert_eq(
        pdf.reindex(columns=columns, copy=True),
        gdf.reindex(columns=columns, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_10(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    columns = ["a", "b", "c", "d", "e"]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate reindexes both labels and column names when
    # index=index_labels and columns=column_labels
    assert_eq(
        pdf.reindex(index=index, columns=columns, copy=True),
        gdf.reindex(index=index, columns=columns, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_change_dtype(copy):
    if PANDAS_GE_110:
        kwargs = {"check_freq": False}
    else:
        kwargs = {}
    index = pd.date_range("12/29/2009", periods=10, freq="D")
    columns = ["a", "b", "c", "d", "e"]
    gdf = gd.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate reindexes both labels and column names when
    # index=index_labels and columns=column_labels
    assert_eq(
        pdf.reindex(index=index, columns=columns, copy=True),
        gdf.reindex(index=index, columns=columns, copy=copy),
        **kwargs,
    )


@pytest.mark.parametrize("copy", [True, False])
def test_series_categorical_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = gd.datasets.randomdata(nrows=6, dtypes={"a": "category"})
    pdf = gdf.to_pandas()
    assert_eq(pdf["a"].reindex(copy=True), gdf["a"].reindex(copy=copy))
    assert_eq(
        pdf["a"].reindex(index, copy=True), gdf["a"].reindex(index, copy=copy)
    )
    assert_eq(
        pdf["a"].reindex(index=index, copy=True),
        gdf["a"].reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_series_float_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = gd.datasets.randomdata(nrows=6, dtypes={"c": float})
    pdf = gdf.to_pandas()
    assert_eq(pdf["c"].reindex(copy=True), gdf["c"].reindex(copy=copy))
    assert_eq(
        pdf["c"].reindex(index, copy=True), gdf["c"].reindex(index, copy=copy)
    )
    assert_eq(
        pdf["c"].reindex(index=index, copy=True),
        gdf["c"].reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_series_string_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = gd.datasets.randomdata(nrows=6, dtypes={"d": str})
    pdf = gdf.to_pandas()
    assert_eq(pdf["d"].reindex(copy=True), gdf["d"].reindex(copy=copy))
    assert_eq(
        pdf["d"].reindex(index, copy=True), gdf["d"].reindex(index, copy=copy)
    )
    assert_eq(
        pdf["d"].reindex(index=index, copy=True),
        gdf["d"].reindex(index=index, copy=copy),
    )


def test_to_frame(pdf, gdf):
    assert_eq(pdf.x.to_frame(), gdf.x.to_frame())

    name = "foo"
    gdf_new_name = gdf.x.to_frame(name=name)
    pdf_new_name = pdf.x.to_frame(name=name)
    assert_eq(pdf.x.to_frame(), gdf.x.to_frame())

    name = False
    gdf_new_name = gdf.x.to_frame(name=name)
    pdf_new_name = pdf.x.to_frame(name=name)
    assert_eq(gdf_new_name, pdf_new_name)
    assert gdf_new_name.columns[0] is name


def test_dataframe_empty_sort_index():
    pdf = pd.DataFrame({"x": []})
    gdf = gd.DataFrame.from_pandas(pdf)

    expect = pdf.sort_index()
    got = gdf.sort_index()

    assert_eq(expect, got)


@pytest.mark.parametrize("axis", [0, 1, "index", "columns"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_sort_index(
    axis, ascending, inplace, ignore_index, na_position
):
    pdf = pd.DataFrame(
        {"b": [1, 3, 2], "a": [1, 4, 3], "c": [4, 1, 5]},
        index=[3.0, 1.0, np.nan],
    )
    gdf = gd.DataFrame.from_pandas(pdf)

    expected = pdf.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )
    got = gdf.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )

    if inplace is True:
        assert_eq(pdf, gdf)
    else:
        assert_eq(expected, got)


@pytest.mark.parametrize("axis", [0, 1, "index", "columns"])
@pytest.mark.parametrize(
    "level",
    [
        0,
        "b",
        1,
        ["b"],
        "a",
        ["a", "b"],
        ["b", "a"],
        [0, 1],
        [1, 0],
        [0, 2],
        None,
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_mulitindex_sort_index(
    axis, level, ascending, inplace, ignore_index, na_position
):
    pdf = pd.DataFrame(
        {
            "b": [1.0, 3.0, np.nan],
            "a": [1, 4, 3],
            1: ["a", "b", "c"],
            "e": [3, 1, 4],
            "d": [1, 2, 8],
        }
    ).set_index(["b", "a", 1])
    gdf = gd.DataFrame.from_pandas(pdf)

    # ignore_index is supported in v.1.0
    expected = pdf.sort_index(
        axis=axis,
        level=level,
        ascending=ascending,
        inplace=inplace,
        na_position=na_position,
    )
    if ignore_index is True:
        expected = expected
    got = gdf.sort_index(
        axis=axis,
        level=level,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )

    if inplace is True:
        if ignore_index is True:
            pdf = pdf.reset_index(drop=True)
        assert_eq(pdf, gdf)
    else:
        if ignore_index is True:
            expected = expected.reset_index(drop=True)
        assert_eq(expected, got)


@pytest.mark.parametrize("dtype", dtypes + ["category"])
def test_dataframe_0_row_dtype(dtype):
    if dtype == "category":
        data = pd.Series(["a", "b", "c", "d", "e"], dtype="category")
    else:
        data = np.array([1, 2, 3, 4, 5], dtype=dtype)

    expect = gd.DataFrame()
    expect["x"] = data
    expect["y"] = data
    got = expect.head(0)

    for col_name in got.columns:
        assert expect[col_name].dtype == got[col_name].dtype

    expect = gd.Series(data)
    got = expect.head(0)

    assert expect.dtype == got.dtype


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_series_list_nanasnull(nan_as_null):
    data = [1.0, 2.0, 3.0, np.nan, None]

    expect = pa.array(data, from_pandas=nan_as_null)
    got = gd.Series(data, nan_as_null=nan_as_null).to_arrow()

    # Bug in Arrow 0.14.1 where NaNs aren't handled
    expect = expect.cast("int64", safe=False)
    got = got.cast("int64", safe=False)

    assert pa.Array.equals(expect, got)


def test_column_assignment():
    gdf = gd.datasets.randomdata(
        nrows=20, dtypes={"a": "category", "b": int, "c": float}
    )
    new_cols = ["q", "r", "s"]
    gdf.columns = new_cols
    assert list(gdf.columns) == new_cols


def test_select_dtype():
    gdf = gd.datasets.randomdata(
        nrows=20, dtypes={"a": "category", "b": int, "c": float, "d": str}
    )
    pdf = gdf.to_pandas()

    assert_eq(pdf.select_dtypes("float64"), gdf.select_dtypes("float64"))
    assert_eq(pdf.select_dtypes(np.float64), gdf.select_dtypes(np.float64))
    assert_eq(
        pdf.select_dtypes(include=["float64"]),
        gdf.select_dtypes(include=["float64"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["object", "int", "category"]),
        gdf.select_dtypes(include=["object", "int", "category"]),
    )

    assert_eq(
        pdf.select_dtypes(include=["int64", "float64"]),
        gdf.select_dtypes(include=["int64", "float64"]),
    )
    assert_eq(
        pdf.select_dtypes(include=np.number),
        gdf.select_dtypes(include=np.number),
    )
    assert_eq(
        pdf.select_dtypes(include=[np.int64, np.float64]),
        gdf.select_dtypes(include=[np.int64, np.float64]),
    )

    assert_eq(
        pdf.select_dtypes(include=["category"]),
        gdf.select_dtypes(include=["category"]),
    )
    assert_eq(
        pdf.select_dtypes(exclude=np.number),
        gdf.select_dtypes(exclude=np.number),
    )

    with pytest.raises(TypeError):
        assert_eq(
            pdf.select_dtypes(include=["Foo"]),
            gdf.select_dtypes(include=["Foo"]),
        )

    with pytest.raises(ValueError):
        gdf.select_dtypes(exclude=np.number, include=np.number)
    with pytest.raises(ValueError):
        pdf.select_dtypes(exclude=np.number, include=np.number)

    gdf = gd.DataFrame({"A": [3, 4, 5], "C": [1, 2, 3], "D": ["a", "b", "c"]})
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(include=["object", "int", "category"]),
        gdf.select_dtypes(include=["object", "int", "category"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["object"], exclude=["category"]),
        gdf.select_dtypes(include=["object"], exclude=["category"]),
    )

    gdf = gd.DataFrame({"a": range(10), "b": range(10, 20)})
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(include=["category"]),
        gdf.select_dtypes(include=["category"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["float"]),
        gdf.select_dtypes(include=["float"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["object"]),
        gdf.select_dtypes(include=["object"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"]), gdf.select_dtypes(include=["int"])
    )
    assert_eq(
        pdf.select_dtypes(exclude=["float"]),
        gdf.select_dtypes(exclude=["float"]),
    )
    assert_eq(
        pdf.select_dtypes(exclude=["object"]),
        gdf.select_dtypes(exclude=["object"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"], exclude=["object"]),
        gdf.select_dtypes(include=["int"], exclude=["object"]),
    )
    with pytest.raises(ValueError):
        pdf.select_dtypes()
    with pytest.raises(ValueError):
        gdf.select_dtypes()

    gdf = gd.DataFrame(
        {"a": gd.Series([], dtype="int"), "b": gd.Series([], dtype="str")}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(exclude=["object"]),
        gdf.select_dtypes(exclude=["object"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"], exclude=["object"]),
        gdf.select_dtypes(include=["int"], exclude=["object"]),
    )


def test_select_dtype_datetime():
    gdf = gd.datasets.timeseries(
        start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={"x": int}
    )
    gdf = gdf.reset_index()
    pdf = gdf.to_pandas()

    assert_eq(pdf.select_dtypes("datetime64"), gdf.select_dtypes("datetime64"))
    assert_eq(
        pdf.select_dtypes(np.dtype("datetime64")),
        gdf.select_dtypes(np.dtype("datetime64")),
    )
    assert_eq(
        pdf.select_dtypes(include="datetime64"),
        gdf.select_dtypes(include="datetime64"),
    )


def test_select_dtype_datetime_with_frequency():
    gdf = gd.datasets.timeseries(
        start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={"x": int}
    )
    gdf = gdf.reset_index()
    pdf = gdf.to_pandas()

    try:
        pdf.select_dtypes("datetime64[ms]")
    except Exception as e:
        with pytest.raises(type(e), match=re.escape(str(e))):
            gdf.select_dtypes("datetime64[ms]")
    else:
        raise AssertionError(
            "Expected pdf.select_dtypes('datetime64[ms]') to fail"
        )


def test_array_ufunc():
    gdf = gd.DataFrame({"x": [2, 3, 4.0], "y": [9.0, 2.5, 1.1]})
    pdf = gdf.to_pandas()

    assert_eq(np.sqrt(gdf), np.sqrt(pdf))
    assert_eq(np.sqrt(gdf.x), np.sqrt(pdf.x))


@pytest.mark.parametrize("nan_value", [-5, -5.0, 0, 5, 5.0, None, "pandas"])
def test_series_to_gpu_array(nan_value):

    s = gd.Series([0, 1, None, 3])
    np.testing.assert_array_equal(
        s.to_array(nan_value), s.to_gpu_array(nan_value).copy_to_host()
    )


def test_dataframe_describe_exclude():
    np.random.seed(12)
    data_length = 10000

    df = gd.DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["x"] = df.x.astype("int64")
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe(exclude=["float"])
    pdf_results = pdf.describe(exclude=["float"])

    assert_eq(gdf_results, pdf_results)


def test_dataframe_describe_include():
    np.random.seed(12)
    data_length = 10000

    df = gd.DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["x"] = df.x.astype("int64")
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe(include=["int"])
    pdf_results = pdf.describe(include=["int"])

    assert_eq(gdf_results, pdf_results)


def test_dataframe_describe_default():
    np.random.seed(12)
    data_length = 10000

    df = gd.DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe()
    pdf_results = pdf.describe()

    assert_eq(pdf_results, gdf_results)


def test_series_describe_include_all():
    np.random.seed(12)
    data_length = 10000

    df = gd.DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["x"] = df.x.astype("int64")
    df["y"] = np.random.normal(10, 1, data_length)
    df["animal"] = np.random.choice(["dog", "cat", "bird"], data_length)

    pdf = df.to_pandas()
    gdf_results = df.describe(include="all")
    pdf_results = pdf.describe(include="all")

    assert_eq(gdf_results[["x", "y"]], pdf_results[["x", "y"]])
    assert_eq(gdf_results.index, pdf_results.index)
    assert_eq(gdf_results.columns, pdf_results.columns)
    assert_eq(
        gdf_results[["animal"]].fillna(-1).astype("str"),
        pdf_results[["animal"]].fillna(-1).astype("str"),
    )


def test_dataframe_describe_percentiles():
    np.random.seed(12)
    data_length = 10000
    sample_percentiles = [0.0, 0.1, 0.33, 0.84, 0.4, 0.99]

    df = gd.DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe(percentiles=sample_percentiles)
    pdf_results = pdf.describe(percentiles=sample_percentiles)

    assert_eq(pdf_results, gdf_results)


def test_get_numeric_data():
    pdf = pd.DataFrame(
        {"x": [1, 2, 3], "y": [1.0, 2.0, 3.0], "z": ["a", "b", "c"]}
    )
    gdf = gd.from_pandas(pdf)

    assert_eq(pdf._get_numeric_data(), gdf._get_numeric_data())


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("period", [-1, -5, -10, -20, 0, 1, 5, 10, 20])
@pytest.mark.parametrize("data_empty", [False, True])
def test_shift(dtype, period, data_empty):

    if data_empty:
        data = None
    else:
        if dtype == np.int8:
            # to keep data in range
            data = gen_rand(dtype, 100000, low=-2, high=2)
        else:
            data = gen_rand(dtype, 100000)

    gdf = gd.DataFrame({"a": gd.Series(data, dtype=dtype)})
    pdf = pd.DataFrame({"a": pd.Series(data, dtype=dtype)})

    shifted_outcome = gdf.a.shift(period).fillna(0)
    expected_outcome = pdf.a.shift(period).fillna(0).astype(dtype)

    if data_empty:
        assert_eq(shifted_outcome, expected_outcome, check_index_type=False)
    else:
        assert_eq(shifted_outcome, expected_outcome)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("period", [-1, -5, -10, -20, 0, 1, 5, 10, 20])
@pytest.mark.parametrize("data_empty", [False, True])
def test_diff(dtype, period, data_empty):
    if data_empty:
        data = None
    else:
        if dtype == np.int8:
            # to keep data in range
            data = gen_rand(dtype, 100000, low=-2, high=2)
        else:
            data = gen_rand(dtype, 100000)

    gdf = gd.DataFrame({"a": gd.Series(data, dtype=dtype)})
    pdf = pd.DataFrame({"a": pd.Series(data, dtype=dtype)})

    expected_outcome = pdf.a.diff(period)
    diffed_outcome = gdf.a.diff(period).astype(expected_outcome.dtype)

    if data_empty:
        assert_eq(diffed_outcome, expected_outcome, check_index_type=False)
    else:
        assert_eq(diffed_outcome, expected_outcome)


def test_isnull_isna():
    # float & strings some missing
    ps = pd.DataFrame(
        {
            "a": [0, 1, 2, np.nan, 4, None, 6],
            "b": [np.nan, None, "u", "h", "d", "a", "m"],
        }
    )
    ps.index = ["q", "w", "e", "r", "t", "y", "u"]
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # integer & string none missing
    ps = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": ["a", "b", "u", "h", "d"]})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # all missing
    ps = pd.DataFrame(
        {"a": [None, None, np.nan, None], "b": [np.nan, None, np.nan, None]}
    )
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # empty
    ps = pd.DataFrame({"a": []})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # one missing
    ps = pd.DataFrame({"a": [np.nan], "b": [None]})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # strings missing
    ps = pd.DataFrame({"a": ["a", "b", "c", None, "e"]})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # strings none missing
    ps = pd.DataFrame({"a": ["a", "b", "c", "d", "e"]})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # unnamed series
    ps = pd.Series([0, 1, 2, np.nan, 4, None, 6])
    gs = gd.Series.from_pandas(ps)
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.isna(), gs.isna())


def test_notna_notnull():
    # float & strings some missing
    ps = pd.DataFrame(
        {
            "a": [0, 1, 2, np.nan, 4, None, 6],
            "b": [np.nan, None, "u", "h", "d", "a", "m"],
        }
    )
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # integer & string none missing
    ps = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": ["a", "b", "u", "h", "d"]})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # all missing
    ps = pd.DataFrame(
        {"a": [None, None, np.nan, None], "b": [np.nan, None, np.nan, None]}
    )
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # empty
    ps = pd.DataFrame({"a": []})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # one missing
    ps = pd.DataFrame({"a": [np.nan], "b": [None]})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # strings missing
    ps = pd.DataFrame({"a": ["a", "b", "c", None, "e"]})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # strings none missing
    ps = pd.DataFrame({"a": ["a", "b", "c", "d", "e"]})
    gs = gd.DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # unnamed series
    ps = pd.Series([0, 1, 2, np.nan, 4, None, 6])
    gs = gd.Series.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.notnull(), gs.notnull())


def test_ndim():
    pdf = pd.DataFrame({"x": range(5), "y": range(5, 10)})
    gdf = gd.DataFrame.from_pandas(pdf)
    assert pdf.ndim == gdf.ndim
    assert pdf.x.ndim == gdf.x.ndim

    s = pd.Series()
    gs = gd.Series()
    assert s.ndim == gs.ndim


@pytest.mark.parametrize(
    "arr",
    [
        np.random.normal(-100, 100, 1000),
        np.random.randint(-50, 50, 1000),
        np.zeros(100),
        np.repeat([-0.6459412758761901], 100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
    ],
)
@pytest.mark.parametrize(
    "decimal",
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        pytest.param(
            -1,
            marks=[
                pytest.mark.xfail(reason="NotImplementedError: decimals < 0")
            ],
        ),
    ],
)
def test_round(arr, decimal):
    pser = pd.Series(arr)
    ser = gd.Series(arr)
    result = ser.round(decimal)
    expected = pser.round(decimal)

    assert_eq(result, expected)

    # with nulls, maintaining existing null mask
    arr = arr.astype("float64")  # for pandas nulls
    mask = np.random.randint(0, 2, arr.shape[0])
    arr[mask == 1] = np.nan

    pser = pd.Series(arr)
    ser = gd.Series(arr)
    result = ser.round(decimal)
    expected = pser.round(decimal)

    assert_eq(result, expected)
    np.array_equal(ser.nullmask.to_array(), result.to_array())


@pytest.mark.parametrize(
    "series",
    [
        gd.Series([1.0, None, np.nan, 4.0], nan_as_null=False),
        gd.Series([1.24430, None, np.nan, 4.423530], nan_as_null=False),
        gd.Series([1.24430, np.nan, 4.423530], nan_as_null=False),
        gd.Series([-1.24430, np.nan, -4.423530], nan_as_null=False),
        gd.Series(np.repeat(np.nan, 100)),
    ],
)
@pytest.mark.parametrize("decimal", [0, 1, 2, 3])
def test_round_nan_as_null_false(series, decimal):
    pser = series.to_pandas()
    ser = gd.Series(series)
    result = ser.round(decimal)
    expected = pser.round(decimal)
    np.testing.assert_array_almost_equal(
        result.to_pandas(), expected, decimal=10
    )


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3],
        [-2, -1, 2, 3, 5],
        [-2, -1, 0, 3, 5],
        [True, False, False],
        [True],
        [False],
        [],
        [True, None, False],
        [True, True, None],
        [None, None],
        [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]],
        [[1, True], [2, False], [3, False]],
        pytest.param(
            [["a", True], ["b", False], ["c", False]],
            marks=[
                pytest.mark.xfail(
                    reason="NotImplementedError: all does not "
                    "support columns of object dtype."
                )
            ],
        ),
    ],
)
def test_all(data):
    # Pandas treats `None` in object type columns as True for some reason, so
    # replacing with `False`
    if np.array(data).ndim <= 1:
        pdata = pd.Series(data).replace([None], False)
        gdata = gd.Series.from_pandas(pdata)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"]).replace([None], False)
        gdata = gd.DataFrame.from_pandas(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.all(bool_only=True)
            expected = pdata.all(bool_only=True)
            assert_eq(got, expected)
        else:
            with pytest.raises(NotImplementedError):
                gdata.all(bool_only=False)
            with pytest.raises(NotImplementedError):
                gdata.all(level="a")

    got = gdata.all()
    expected = pdata.all()
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3],
        [-2, -1, 2, 3, 5],
        [-2, -1, 0, 3, 5],
        [0, 0, 0, 0, 0],
        [0, 0, None, 0],
        [True, False, False],
        [True],
        [False],
        [],
        [True, None, False],
        [True, True, None],
        [None, None],
        [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]],
        [[1, True], [2, False], [3, False]],
        pytest.param(
            [["a", True], ["b", False], ["c", False]],
            marks=[
                pytest.mark.xfail(
                    reason="NotImplementedError: any does not "
                    "support columns of object dtype."
                )
            ],
        ),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_any(data, axis):
    if np.array(data).ndim <= 1:
        pdata = pd.Series(data)
        gdata = gd.Series.from_pandas(pdata)

        if axis == 1:
            with pytest.raises(NotImplementedError):
                gdata.any(axis=axis)
        else:
            got = gdata.any(axis=axis)
            expected = pdata.any(axis=axis)
            assert_eq(got, expected)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"])
        gdata = gd.DataFrame.from_pandas(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.any(bool_only=True)
            expected = pdata.any(bool_only=True)
            assert_eq(got, expected)
        else:
            with pytest.raises(NotImplementedError):
                gdata.any(bool_only=False)
            with pytest.raises(NotImplementedError):
                gdata.any(level="a")

        got = gdata.any(axis=axis)
        expected = pdata.any(axis=axis)
        assert_eq(got, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_empty_dataframe_any(axis):
    pdf = pd.DataFrame({}, columns=["a", "b"])
    gdf = gd.DataFrame.from_pandas(pdf)
    got = gdf.any(axis=axis)
    expected = pdf.any(axis=axis)
    assert_eq(got, expected, check_index_type=False)


@pytest.mark.parametrize("indexed", [False, True])
def test_dataframe_sizeof(indexed):
    rows = int(1e6)
    index = list(i for i in range(rows)) if indexed else None

    gdf = gd.DataFrame({"A": [8] * rows, "B": [32] * rows}, index=index)

    for c in gdf._data.columns:
        assert gdf._index.__sizeof__() == gdf._index.__sizeof__()
    cols_sizeof = sum(c.__sizeof__() for c in gdf._data.columns)
    assert gdf.__sizeof__() == (gdf._index.__sizeof__() + cols_sizeof)


@pytest.mark.parametrize("a", [[], ["123"]])
@pytest.mark.parametrize("b", ["123", ["123"]])
@pytest.mark.parametrize(
    "misc_data",
    ["123", ["123"] * 20, 123, [1, 2, 0.8, 0.9] * 50, 0.9, 0.00001],
)
@pytest.mark.parametrize("non_list_data", [123, "abc", "zyx", "rapids", 0.8])
def test_create_dataframe_cols_empty_data(a, b, misc_data, non_list_data):
    expected = pd.DataFrame({"a": a})
    actual = gd.DataFrame.from_pandas(expected)
    expected["b"] = b
    actual["b"] = b
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": []})
    actual = gd.DataFrame.from_pandas(expected)
    expected["b"] = misc_data
    actual["b"] = misc_data
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": a})
    actual = gd.DataFrame.from_pandas(expected)
    expected["b"] = non_list_data
    actual["b"] = non_list_data
    assert_eq(actual, expected)


def test_empty_dataframe_describe():
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = gd.from_pandas(pdf)

    expected = pdf.describe()
    actual = gdf.describe()

    assert_eq(expected, actual)


def test_as_column_types():
    from cudf.core.column import column

    col = column.as_column(gd.Series([]))
    assert_eq(col.dtype, np.dtype("float64"))
    gds = gd.Series(col)
    pds = pd.Series(pd.Series([]))

    assert_eq(pds, gds)

    col = column.as_column(gd.Series([]), dtype="float32")
    assert_eq(col.dtype, np.dtype("float32"))
    gds = gd.Series(col)
    pds = pd.Series(pd.Series([], dtype="float32"))

    assert_eq(pds, gds)

    col = column.as_column(gd.Series([]), dtype="str")
    assert_eq(col.dtype, np.dtype("object"))
    gds = gd.Series(col)
    pds = pd.Series(pd.Series([], dtype="str"))

    assert_eq(pds, gds)

    col = column.as_column(gd.Series([]), dtype="object")
    assert_eq(col.dtype, np.dtype("object"))
    gds = gd.Series(col)
    pds = pd.Series(pd.Series([], dtype="object"))

    assert_eq(pds, gds)

    pds = pd.Series(np.array([1, 2, 3]), dtype="float32")
    gds = gd.Series(column.as_column(np.array([1, 2, 3]), dtype="float32"))

    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 3], dtype="float32")
    gds = gd.Series([1, 2, 3], dtype="float32")

    assert_eq(pds, gds)

    pds = pd.Series([])
    gds = gd.Series(column.as_column(pds))
    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 4], dtype="int64")
    gds = gd.Series(column.as_column(gd.Series([1, 2, 4]), dtype="int64"))

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="float32")
    gds = gd.Series(
        column.as_column(gd.Series([1.2, 18.0, 9.0]), dtype="float32")
    )

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="str")
    gds = gd.Series(column.as_column(gd.Series([1.2, 18.0, 9.0]), dtype="str"))

    assert_eq(pds, gds)

    pds = pd.Series(pd.Index(["1", "18", "9"]), dtype="int")
    gds = gd.Series(gd.core.index.StringIndex(["1", "18", "9"]), dtype="int")

    assert_eq(pds, gds)


def test_one_row_head():
    gdf = gd.DataFrame({"name": ["carl"], "score": [100]}, index=[123])
    pdf = gdf.to_pandas()

    head_gdf = gdf.head()
    head_pdf = pdf.head()

    assert_eq(head_pdf, head_gdf)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("as_dtype", NUMERIC_TYPES)
def test_series_astype_numeric_to_numeric(dtype, as_dtype):
    psr = pd.Series([1, 2, 4, 3], dtype=dtype)
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("as_dtype", NUMERIC_TYPES)
def test_series_astype_numeric_to_numeric_nulls(dtype, as_dtype):
    data = [1, 2, None, 3]
    sr = gd.Series(data, dtype=dtype)
    got = sr.astype(as_dtype)
    expect = gd.Series([1, 2, None, 3], dtype=as_dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize(
    "as_dtype",
    [
        "str",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_numeric_to_other(dtype, as_dtype):
    psr = pd.Series([1, 2, 3], dtype=dtype)
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "as_dtype",
    [
        "str",
        "int32",
        "uint32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_string_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05"]
    else:
        data = ["1", "2", "3"]
    psr = pd.Series(data)
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "as_dtype",
    [
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_datetime_to_other(as_dtype):
    data = ["2001-01-01", "2002-02-02", "2001-01-05"]
    psr = pd.Series(data)
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "inp",
    [
        ("datetime64[ns]", "2011-01-01 00:00:00.000000000"),
        ("datetime64[us]", "2011-01-01 00:00:00.000000"),
        ("datetime64[ms]", "2011-01-01 00:00:00.000"),
        ("datetime64[s]", "2011-01-01 00:00:00"),
    ],
)
def test_series_astype_datetime_to_string(inp):
    dtype, expect = inp
    base_date = "2011-01-01"
    sr = gd.Series([base_date], dtype=dtype)
    got = sr.astype(str)[0]
    assert expect == got


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "uint32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
    ],
)
def test_series_astype_categorical_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05", "2001-01-01"]
    else:
        data = [1, 2, 3, 1]
    psr = pd.Series(data, dtype="category")
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize("ordered", [True, False])
def test_series_astype_to_categorical_ordered(ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    gsr = gd.from_pandas(psr)

    ordered_dtype_pd = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=ordered
    )
    ordered_dtype_gd = gd.CategoricalDtype.from_pandas(ordered_dtype_pd)
    assert_eq(
        psr.astype("int32").astype(ordered_dtype_pd).astype("int32"),
        gsr.astype("int32").astype(ordered_dtype_gd).astype("int32"),
    )


@pytest.mark.parametrize("ordered", [True, False])
def test_series_astype_cat_ordered_to_unordered(ordered):
    pd_dtype = pd.CategoricalDtype(categories=[1, 2, 3], ordered=ordered)
    pd_to_dtype = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=not ordered
    )
    gd_dtype = gd.CategoricalDtype.from_pandas(pd_dtype)
    gd_to_dtype = gd.CategoricalDtype.from_pandas(pd_to_dtype)

    psr = pd.Series([1, 2, 3], dtype=pd_dtype)
    gsr = gd.Series([1, 2, 3], dtype=gd_dtype)

    expect = psr.astype(pd_to_dtype)
    got = gsr.astype(gd_to_dtype)

    assert_eq(expect, got)


def test_series_astype_null_cases():
    data = [1, 2, None, 3]

    # numerical to other
    assert_eq(gd.Series(data, dtype="str"), gd.Series(data).astype("str"))

    assert_eq(
        gd.Series(data, dtype="category"), gd.Series(data).astype("category")
    )

    assert_eq(
        gd.Series(data, dtype="float32"),
        gd.Series(data, dtype="int32").astype("float32"),
    )

    assert_eq(
        gd.Series(data, dtype="float32"),
        gd.Series(data, dtype="uint32").astype("float32"),
    )

    assert_eq(
        gd.Series(data, dtype="datetime64[ms]"),
        gd.Series(data).astype("datetime64[ms]"),
    )

    # categorical to other
    assert_eq(
        gd.Series(data, dtype="str"),
        gd.Series(data, dtype="category").astype("str"),
    )

    assert_eq(
        gd.Series(data, dtype="float32"),
        gd.Series(data, dtype="category").astype("float32"),
    )

    assert_eq(
        gd.Series(data, dtype="datetime64[ms]"),
        gd.Series(data, dtype="category").astype("datetime64[ms]"),
    )

    # string to other
    assert_eq(
        gd.Series([1, 2, None, 3], dtype="int32"),
        gd.Series(["1", "2", None, "3"]).astype("int32"),
    )

    assert_eq(
        gd.Series(
            ["2001-01-01", "2001-02-01", None, "2001-03-01"],
            dtype="datetime64[ms]",
        ),
        gd.Series(["2001-01-01", "2001-02-01", None, "2001-03-01"]).astype(
            "datetime64[ms]"
        ),
    )

    assert_eq(
        gd.Series(["a", "b", "c", None], dtype="category").to_pandas(),
        gd.Series(["a", "b", "c", None]).astype("category").to_pandas(),
    )

    # datetime to other
    data = [
        "2001-01-01 00:00:00.000000",
        "2001-02-01 00:00:00.000000",
        None,
        "2001-03-01 00:00:00.000000",
    ]
    assert_eq(
        gd.Series(data), gd.Series(data, dtype="datetime64[us]").astype("str"),
    )

    assert_eq(
        pd.Series(data, dtype="datetime64[ns]").astype("category"),
        gd.from_pandas(pd.Series(data, dtype="datetime64[ns]")).astype(
            "category"
        ),
    )


def test_series_astype_null_categorical():
    sr = gd.Series([None, None, None], dtype="category")
    expect = gd.Series([None, None, None], dtype="int32")
    got = sr.astype("int32")
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        (
            pd.Series([3, 3.0]),
            pd.Series([2.3, 3.9]),
            pd.Series([1.5, 3.9]),
            pd.Series([1.0, 2]),
        ),
        [
            pd.Series([3, 3.0]),
            pd.Series([2.3, 3.9]),
            pd.Series([1.5, 3.9]),
            pd.Series([1.0, 2]),
        ],
    ],
)
def test_create_dataframe_from_list_like(data):
    pdf = pd.DataFrame(data, index=["count", "mean", "std", "min"])
    gdf = gd.DataFrame(data, index=["count", "mean", "std", "min"])

    assert_eq(pdf, gdf)

    pdf = pd.DataFrame(data)
    gdf = gd.DataFrame(data)

    assert_eq(pdf, gdf)


def test_create_dataframe_column():
    pdf = pd.DataFrame(columns=["a", "b", "c"], index=["A", "Z", "X"])
    gdf = gd.DataFrame(columns=["a", "b", "c"], index=["A", "Z", "X"])

    assert_eq(pdf, gdf)

    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [2, 3, 5]},
        columns=["a", "b", "c"],
        index=["A", "Z", "X"],
    )
    gdf = gd.DataFrame(
        {"a": [1, 2, 3], "b": [2, 3, 5]},
        columns=["a", "b", "c"],
        index=["A", "Z", "X"],
    )

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4],
        [],
        [5.0, 7.0, 8.0],
        pd.Categorical(["a", "b", "c"]),
        ["m", "a", "d", "v"],
    ],
)
def test_series_values_host_property(data):
    pds = pd.Series(data)
    gds = gd.Series(data)

    np.testing.assert_array_equal(pds.values, gds.values_host)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4],
        [],
        [5.0, 7.0, 8.0],
        pytest.param(
            pd.Categorical(["a", "b", "c"]),
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
        pytest.param(
            ["m", "a", "d", "v"],
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
    ],
)
def test_series_values_property(data):
    pds = pd.Series(data)
    gds = gd.Series(data)
    gds_vals = gds.values
    assert isinstance(gds_vals, cupy.ndarray)
    np.testing.assert_array_equal(gds_vals.get(), pds.values)


@pytest.mark.parametrize(
    "data",
    [
        {"A": [1, 2, 3], "B": [4, 5, 6]},
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
        {"A": [1, 2, 3], "B": [1.0, 2.0, 3.0]},
        {"A": np.float32(np.arange(3)), "B": np.float64(np.arange(3))},
        pytest.param(
            {"A": [1, None, 3], "B": [1, 2, None]},
            marks=pytest.mark.xfail(
                reason="Nulls not supported by as_gpu_matrix"
            ),
        ),
        pytest.param(
            {"A": [None, None, None], "B": [None, None, None]},
            marks=pytest.mark.xfail(
                reason="Nulls not supported by as_gpu_matrix"
            ),
        ),
        pytest.param(
            {"A": [], "B": []},
            marks=pytest.mark.xfail(reason="Requires at least 1 row"),
        ),
        pytest.param(
            {"A": [1, 2, 3], "B": ["a", "b", "c"]},
            marks=pytest.mark.xfail(
                reason="str or categorical not supported by as_gpu_matrix"
            ),
        ),
        pytest.param(
            {"A": pd.Categorical(["a", "b", "c"]), "B": ["d", "e", "f"]},
            marks=pytest.mark.xfail(
                reason="str or categorical not supported by as_gpu_matrix"
            ),
        ),
    ],
)
def test_df_values_property(data):
    pdf = pd.DataFrame.from_dict(data)
    gdf = gd.DataFrame.from_pandas(pdf)

    pmtr = pdf.values
    gmtr = gdf.values.get()

    np.testing.assert_array_equal(pmtr, gmtr)


def test_value_counts():
    pdf = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    gdf = gd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    assert_eq(
        pdf.numeric.value_counts().sort_index(),
        gdf.numeric.value_counts().sort_index(),
        check_dtype=False,
    )
    assert_eq(
        pdf.alpha.value_counts().sort_index(),
        gdf.alpha.value_counts().sort_index(),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "data",
    [
        [],
        [0, 12, 14],
        [0, 14, 12, 12, 3, 10, 12, 14],
        np.random.randint(-100, 100, 200),
        pd.Series([0.0, 1.0, None, 10.0]),
        [None, None, None, None],
        [np.nan, None, -1, 2, 3],
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.random.randint(-100, 100, 10),
        [],
        [np.nan, None, -1, 2, 3],
        [1.0, 12.0, None, None, 120],
        [0, 14, 12, 12, 3, 10, 12, 14, None],
        [None, None, None],
        ["0", "12", "14"],
        ["0", "12", "14", "a"],
    ],
)
def test_isin_numeric(data, values):
    index = np.random.randint(0, 100, len(data))
    psr = pd.Series(data, index=index)
    gsr = gd.Series.from_pandas(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(
            ["2018-01-01", "2019-04-03", None, "2019-12-30"],
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [
                "2018-01-01",
                "2019-04-03",
                None,
                "2019-12-30",
                "2018-01-01",
                "2018-01-01",
            ],
            dtype="datetime64[ns]",
        ),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        [1514764800000000000, 1577664000000000000],
        [
            1514764800000000000,
            1577664000000000000,
            1577664000000000000,
            1577664000000000000,
            1514764800000000000,
        ],
        ["2019-04-03", "2019-12-30", "2012-01-01"],
        [
            "2012-01-01",
            "2012-01-01",
            "2012-01-01",
            "2019-04-03",
            "2019-12-30",
            "2012-01-01",
        ],
    ],
)
def test_isin_datetime(data, values):
    psr = pd.Series(data)
    gsr = gd.Series.from_pandas(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(["this", "is", None, "a", "test"]),
        pd.Series(["test", "this", "test", "is", None, "test", "a", "test"]),
        pd.Series(["0", "12", "14"]),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        [None, None, None],
        ["12", "14", "19"],
        pytest.param(
            [12, 14, 19],
            marks=[
                pytest.mark.xfail(
                    reason="pandas's failure here seems like a bug "
                    "given the reverse succeeds"
                )
            ],
        ),
        ["is", "this", "is", "this", "is"],
    ],
)
def test_isin_string(data, values):
    psr = pd.Series(data)
    gsr = gd.Series.from_pandas(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(["a", "b", "c", "c", "c", "d", "e"], dtype="category"),
        pd.Series(["a", "b", None, "c", "d", "e"], dtype="category"),
        pd.Series([0, 3, 10, 12], dtype="category"),
        pd.Series([0, 3, 10, 12, 0, 10, 3, 0, 0, 3, 3], dtype="category"),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["a", "b", None, "f", "words"],
        ["0", "12", None, "14"],
        [0, 10, 12, None, 39, 40, 1000],
        [0, 0, 0, 0, 3, 3, 3, None, 1, 2, 3],
    ],
)
def test_isin_categorical(data, values):
    psr = pd.Series(data)
    gsr = gd.Series.from_pandas(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(
            ["this", "is", None, "a", "test"], index=["a", "b", "c", "d", "e"]
        ),
        pd.Series([0, 15, 10], index=[0, None, 9]),
        pd.Series(
            range(25),
            index=pd.date_range(
                start="2019-01-01", end="2019-01-02", freq="H"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        [0, 19, 13],
        ["2019-01-01 04:00:00", "2019-01-01 06:00:00", "2018-03-02"],
    ],
)
def test_isin_index(data, values):
    psr = pd.Series(data)
    gsr = gd.Series.from_pandas(psr)

    got = gsr.index.isin(values)
    expected = psr.index.isin(values)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        pd.MultiIndex.from_arrays(
            [[1, 2, 3], ["red", "blue", "green"]], names=("number", "color")
        ),
        pd.MultiIndex.from_arrays([[], []], names=("number", "color")),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 10, 100], ["red", "blue", "green", "pink", "white"]],
            names=("number", "color"),
        ),
    ],
)
@pytest.mark.parametrize(
    "values,level,err",
    [
        (["red", "orange", "yellow"], "color", None),
        (["red", "white", "yellow"], "color", None),
        ([0, 1, 2, 10, 11, 15], "number", None),
        ([0, 1, 2, 10, 11, 15], None, TypeError),
        (pd.Series([0, 1, 2, 10, 11, 15]), None, TypeError),
        (pd.Index([0, 1, 2, 10, 11, 15]), None, TypeError),
        (pd.Index([0, 1, 2, 8, 11, 15]), "number", None),
        (pd.Index(["red", "white", "yellow"]), "color", None),
        ([(1, "red"), (3, "red")], None, None),
        (((1, "red"), (3, "red")), None, None),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3], ["red", "blue", "green"]],
                names=("number", "color"),
            ),
            None,
            None,
        ),
        (
            pd.MultiIndex.from_arrays([[], []], names=("number", "color")),
            None,
            None,
        ),
        (
            pd.MultiIndex.from_arrays(
                [
                    [1, 2, 3, 10, 100],
                    ["red", "blue", "green", "pink", "white"],
                ],
                names=("number", "color"),
            ),
            None,
            None,
        ),
    ],
)
def test_isin_multiindex(data, values, level, err):
    pmdx = data
    gmdx = gd.from_pandas(data)

    if err is None:
        expected = pmdx.isin(values, level=level)
        if isinstance(values, pd.MultiIndex):
            values = gd.from_pandas(values)
        got = gmdx.isin(values, level=level)

        assert_eq(got, expected)
    else:
        with pytest.raises((ValueError, TypeError)):
            expected = pmdx.isin(values, level=level)

        with pytest.raises(err):
            got = gmdx.isin(values, level=level)


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(
            {
                "num_legs": [2, 4],
                "num_wings": [2, 0],
                "bird_cats": pd.Series(
                    ["sparrow", "pigeon"],
                    dtype="category",
                    index=["falcon", "dog"],
                ),
            },
            index=["falcon", "dog"],
        ),
        pd.DataFrame(
            {"num_legs": [8, 2], "num_wings": [0, 2]},
            index=["spider", "falcon"],
        ),
        pd.DataFrame(
            {
                "num_legs": [8, 2, 1, 0, 2, 4, 5],
                "num_wings": [2, 0, 2, 1, 2, 4, -1],
            }
        ),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [0, 2],
        {"num_wings": [0, 3]},
        pd.DataFrame(
            {"num_legs": [8, 2], "num_wings": [0, 2]},
            index=["spider", "falcon"],
        ),
        pd.DataFrame(
            {
                "num_legs": [2, 4],
                "num_wings": [2, 0],
                "bird_cats": pd.Series(
                    ["sparrow", "pigeon"],
                    dtype="category",
                    index=["falcon", "dog"],
                ),
            },
            index=["falcon", "dog"],
        ),
        ["sparrow", "pigeon"],
        pd.Series(["sparrow", "pigeon"], dtype="category"),
        pd.Series([1, 2, 3, 4, 5]),
        "abc",
        123,
    ],
)
def test_isin_dataframe(data, values):
    from cudf.utils.dtypes import is_scalar

    pdf = data
    gdf = gd.from_pandas(pdf)

    if is_scalar(values):
        with pytest.raises(TypeError):
            pdf.isin(values)
        with pytest.raises(TypeError):
            gdf.isin(values)
    else:
        try:
            expected = pdf.isin(values)
        except ValueError as e:
            if str(e) == "Lengths must match.":
                # xref https://github.com/pandas-dev/pandas/issues/34256
                pytest.xfail(
                    "https://github.com/pandas-dev/pandas/issues/34256"
                )
        if isinstance(values, (pd.DataFrame, pd.Series)):
            values = gd.from_pandas(values)
        got = gdf.isin(values)
        assert_eq(got, expected)


def test_constructor_properties():
    df = gd.DataFrame()
    key1 = "a"
    key2 = "b"
    val1 = np.array([123], dtype=np.float64)
    val2 = np.array([321], dtype=np.float64)
    df[key1] = val1
    df[key2] = val2

    # Correct use of _constructor (for DataFrame)
    assert_eq(df, df._constructor({key1: val1, key2: val2}))

    # Correct use of _constructor (for gd.Series)
    assert_eq(df[key1], df[key2]._constructor(val1, name=key1))

    # Correct use of _constructor_sliced (for DataFrame)
    assert_eq(df[key1], df._constructor_sliced(val1, name=key1))

    # Correct use of _constructor_expanddim (for gd.Series)
    assert_eq(df, df[key2]._constructor_expanddim({key1: val1, key2: val2}))

    # Incorrect use of _constructor_sliced (Raises for gd.Series)
    with pytest.raises(NotImplementedError):
        df[key1]._constructor_sliced

    # Incorrect use of _constructor_expanddim (Raises for DataFrame)
    with pytest.raises(NotImplementedError):
        df._constructor_expanddim


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("as_dtype", ALL_TYPES)
def test_df_astype_numeric_to_all(dtype, as_dtype):
    if "uint" in dtype:
        data = [1, 2, None, 4, 7]
    elif "int" in dtype or "longlong" in dtype:
        data = [1, 2, None, 4, -7]
    elif "float" in dtype:
        data = [1.0, 2.0, None, 4.0, np.nan, -7.0]

    gdf = gd.DataFrame()

    gdf["foo"] = gd.Series(data, dtype=dtype)
    gdf["bar"] = gd.Series(data, dtype=dtype)

    insert_data = gd.Series(data, dtype=dtype)

    expect = gd.DataFrame()
    expect["foo"] = insert_data.astype(as_dtype)
    expect["bar"] = insert_data.astype(as_dtype)

    got = gdf.astype(as_dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_df_astype_string_to_other(as_dtype):
    if "datetime64" in as_dtype:
        # change None to "NaT" after this issue is fixed:
        # https://github.com/rapidsai/cudf/issues/5117
        data = ["2001-01-01", "2002-02-02", "2000-01-05", None]
    elif as_dtype == "int32":
        data = [1, 2, 3]
    elif as_dtype == "category":
        data = ["1", "2", "3", None]
    elif "float" in as_dtype:
        data = [1.0, 2.0, 3.0, np.nan]

    insert_data = gd.Series.from_pandas(pd.Series(data, dtype="str"))
    expect_data = gd.Series(data, dtype=as_dtype)

    gdf = gd.DataFrame()
    expect = gd.DataFrame()

    gdf["foo"] = insert_data
    gdf["bar"] = insert_data

    expect["foo"] = expect_data
    expect["bar"] = expect_data

    got = gdf.astype(as_dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int64",
        "datetime64[s]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
        "category",
    ],
)
def test_df_astype_datetime_to_other(as_dtype):
    data = [
        "1991-11-20 00:00:00.000",
        "2004-12-04 00:00:00.000",
        "2016-09-13 00:00:00.000",
        None,
    ]

    gdf = gd.DataFrame()
    expect = gd.DataFrame()

    gdf["foo"] = gd.Series(data, dtype="datetime64[ms]")
    gdf["bar"] = gd.Series(data, dtype="datetime64[ms]")

    if as_dtype == "int64":
        expect["foo"] = gd.Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
        expect["bar"] = gd.Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
    elif as_dtype == "str":
        expect["foo"] = gd.Series(data, dtype="str")
        expect["bar"] = gd.Series(data, dtype="str")
    elif as_dtype == "category":
        expect["foo"] = gd.Series(gdf["foo"], dtype="category")
        expect["bar"] = gd.Series(gdf["bar"], dtype="category")
    else:
        expect["foo"] = gd.Series(data, dtype=as_dtype)
        expect["bar"] = gd.Series(data, dtype=as_dtype)

    got = gdf.astype(as_dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
    ],
)
def test_df_astype_categorical_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05", "2001-01-01"]
    else:
        data = [1, 2, 3, 1]
    psr = pd.Series(data, dtype="category")
    pdf = pd.DataFrame()
    pdf["foo"] = psr
    pdf["bar"] = psr
    gdf = gd.DataFrame.from_pandas(pdf)
    assert_eq(pdf.astype(as_dtype), gdf.astype(as_dtype))


@pytest.mark.parametrize("ordered", [True, False])
def test_df_astype_to_categorical_ordered(ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    pdf = pd.DataFrame()
    pdf["foo"] = psr
    pdf["bar"] = psr
    gdf = gd.DataFrame.from_pandas(pdf)

    ordered_dtype_pd = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=ordered
    )
    ordered_dtype_gd = gd.CategoricalDtype.from_pandas(ordered_dtype_pd)

    assert_eq(
        pdf.astype(ordered_dtype_pd).astype("int32"),
        gdf.astype(ordered_dtype_gd).astype("int32"),
    )


@pytest.mark.parametrize(
    "dtype,args",
    [(dtype, {}) for dtype in ALL_TYPES]
    + [("category", {"ordered": True}), ("category", {"ordered": False})],
)
def test_empty_df_astype(dtype, args):
    df = gd.DataFrame()
    kwargs = {}
    kwargs.update(args)
    assert_eq(df, df.astype(dtype=dtype, **kwargs))


@pytest.mark.parametrize(
    "errors",
    [
        pytest.param(
            "raise", marks=pytest.mark.xfail(reason="should raise error here")
        ),
        pytest.param("other", marks=pytest.mark.xfail(raises=ValueError)),
        "ignore",
        pytest.param(
            "warn", marks=pytest.mark.filterwarnings("ignore:Traceback")
        ),
    ],
)
def test_series_astype_error_handling(errors):
    sr = gd.Series(["random", "words"])
    got = sr.astype("datetime64", errors=errors)
    assert_eq(sr, got)


@pytest.mark.parametrize("dtype", ALL_TYPES)
def test_df_constructor_dtype(dtype):
    if "datetime" in dtype:
        data = ["1991-11-20", "2004-12-04", "2016-09-13", None]
    elif dtype == "str":
        data = ["a", "b", "c", None]
    elif "float" in dtype:
        data = [1.0, 0.5, -1.1, np.nan, None]
    elif "bool" in dtype:
        data = [True, False, None]
    else:
        data = [1, 2, 3, None]

    sr = gd.Series(data, dtype=dtype)

    expect = gd.DataFrame()
    expect["foo"] = sr
    expect["bar"] = sr
    got = gd.DataFrame({"foo": data, "bar": data}, dtype=dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        gd.datasets.randomdata(
            nrows=10, dtypes={"a": "category", "b": int, "c": float, "d": int}
        ),
        gd.datasets.randomdata(
            nrows=10, dtypes={"a": "category", "b": int, "c": float, "d": str}
        ),
        gd.datasets.randomdata(
            nrows=10, dtypes={"a": bool, "b": int, "c": float, "d": str}
        ),
        gd.DataFrame(),
        gd.DataFrame({"a": [0, 1, 2], "b": [1, None, 3]}),
        gd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [7, np.NaN, 9, 10],
                "c": [np.NaN, np.NaN, np.NaN, np.NaN],
                "d": gd.Series([None, None, None, None], dtype="int64"),
                "e": [100, None, 200, None],
                "f": gd.Series([10, None, np.NaN, 11], nan_as_null=False),
            }
        ),
        gd.DataFrame(
            {
                "a": [10, 11, 12, 13, 14, 15],
                "b": gd.Series(
                    [10, None, np.NaN, 2234, None, np.NaN], nan_as_null=False
                ),
            }
        ),
    ],
)
@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "product", "mean", "var", "std"]
)
@pytest.mark.parametrize("skipna", [True, False])
def test_rowwise_ops(data, op, skipna):
    gdf = data
    pdf = gdf.to_pandas()

    if op in ("var", "std"):
        expected = getattr(pdf, op)(axis=1, ddof=0, skipna=skipna)
        got = getattr(gdf, op)(axis=1, ddof=0, skipna=skipna)
    else:
        expected = getattr(pdf, op)(axis=1, skipna=skipna)
        got = getattr(gdf, op)(axis=1, skipna=skipna)

    assert_eq(expected, got, check_less_precise=7)


@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "product", "mean", "var", "std"]
)
def test_rowwise_ops_nullable_dtypes_all_null(op):
    gdf = gd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [7, np.NaN, 9, 10],
            "c": [np.NaN, np.NaN, np.NaN, np.NaN],
            "d": gd.Series([None, None, None, None], dtype="int64"),
            "e": [100, None, 200, None],
            "f": gd.Series([10, None, np.NaN, 11], nan_as_null=False),
        }
    )

    expected = gd.Series([None, None, None, None], dtype="float64")

    if op in ("var", "std"):
        got = getattr(gdf, op)(axis=1, ddof=0, skipna=False)
    else:
        got = getattr(gdf, op)(axis=1, skipna=False)

    assert_eq(got.null_count, expected.null_count)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "op,expected",
    [
        (
            "max",
            gd.Series(
                [10.0, None, np.NaN, 2234.0, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "min",
            gd.Series(
                [10.0, None, np.NaN, 13.0, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "sum",
            gd.Series(
                [20.0, None, np.NaN, 2247.0, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "product",
            gd.Series(
                [100.0, None, np.NaN, 29042.0, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "mean",
            gd.Series(
                [10.0, None, np.NaN, 1123.5, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "var",
            gd.Series(
                [0.0, None, np.NaN, 1233210.25, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "std",
            gd.Series(
                [0.0, None, np.NaN, 1110.5, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
    ],
)
def test_rowwise_ops_nullable_dtypes_partial_null(op, expected):
    gdf = gd.DataFrame(
        {
            "a": [10, 11, 12, 13, 14, 15],
            "b": gd.Series(
                [10, None, np.NaN, 2234, None, np.NaN], nan_as_null=False,
            ),
        }
    )

    if op in ("var", "std"):
        got = getattr(gdf, op)(axis=1, ddof=0, skipna=False)
    else:
        got = getattr(gdf, op)(axis=1, skipna=False)

    assert_eq(got.null_count, expected.null_count)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "op,expected",
    [
        ("max", gd.Series([10, None, None, 2234, None, 453], dtype="int64",),),
        ("min", gd.Series([10, None, None, 13, None, 15], dtype="int64",),),
        ("sum", gd.Series([20, None, None, 2247, None, 468], dtype="int64",),),
        (
            "product",
            gd.Series([100, None, None, 29042, None, 6795], dtype="int64",),
        ),
        (
            "mean",
            gd.Series(
                [10.0, None, None, 1123.5, None, 234.0], dtype="float32",
            ),
        ),
        (
            "var",
            gd.Series(
                [0.0, None, None, 1233210.25, None, 47961.0], dtype="float32",
            ),
        ),
        (
            "std",
            gd.Series(
                [0.0, None, None, 1110.5, None, 219.0], dtype="float32",
            ),
        ),
    ],
)
def test_rowwise_ops_nullable_int_dtypes(op, expected):
    gdf = gd.DataFrame(
        {
            "a": [10, 11, None, 13, None, 15],
            "b": gd.Series(
                [10, None, 323, 2234, None, 453], nan_as_null=False,
            ),
        }
    )

    if op in ("var", "std"):
        got = getattr(gdf, op)(axis=1, ddof=0, skipna=False)
    else:
        got = getattr(gdf, op)(axis=1, skipna=False)

    assert_eq(got.null_count, expected.null_count)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [5.0, 6.0, 7.0],
        "single value",
        np.array(1, dtype="int64"),
        np.array(0.6273643, dtype="float64"),
    ],
)
def test_insert(data):
    pdf = pd.DataFrame.from_dict({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    gdf = gd.DataFrame.from_pandas(pdf)

    # insertion by index

    pdf.insert(0, "foo", data)
    gdf.insert(0, "foo", data)

    assert_eq(pdf, gdf)

    pdf.insert(3, "bar", data)
    gdf.insert(3, "bar", data)

    assert_eq(pdf, gdf)

    pdf.insert(1, "baz", data)
    gdf.insert(1, "baz", data)

    assert_eq(pdf, gdf)

    # pandas insert doesn't support negative indexing
    pdf.insert(len(pdf.columns), "qux", data)
    gdf.insert(-1, "qux", data)

    assert_eq(pdf, gdf)


def test_cov():
    gdf = gd.datasets.randomdata(10)
    pdf = gdf.to_pandas()

    assert_eq(pdf.cov(), gdf.cov())


@pytest.mark.xfail(reason="cupy-based cov does not support nulls")
def test_cov_nans():
    pdf = pd.DataFrame()
    pdf["a"] = [None, None, None, 2.00758632, None]
    pdf["b"] = [0.36403686, None, None, None, None]
    pdf["c"] = [None, None, None, 0.64882227, None]
    pdf["d"] = [None, -1.46863125, None, 1.22477948, -0.06031689]
    gdf = gd.from_pandas(pdf)

    assert_eq(pdf.cov(), gdf.cov())


@pytest.mark.parametrize(
    "gsr",
    [
        gd.Series([1, 2, 3]),
        gd.Series([1, 2, 3], index=["a", "b", "c"]),
        gd.Series([1, 2, 3], index=["a", "b", "d"]),
        gd.Series([1, 2], index=["a", "b"]),
        gd.Series([1, 2, 3], index=gd.core.index.RangeIndex(0, 3)),
        pytest.param(
            gd.Series([1, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"]),
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.parametrize("colnames", [["a", "b", "c"], [0, 1, 2]])
@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.ne,
    ],
)
def test_df_sr_binop(gsr, colnames, op):
    data = [[0, 2, 5], [3, None, 5], [6, 7, np.nan]]
    data = dict(zip(colnames, data))

    gdf = gd.DataFrame(data)
    pdf = pd.DataFrame.from_dict(data)

    psr = gsr.to_pandas()

    expect = op(pdf, psr)
    got = op(gdf, gsr)
    assert_eq(expect.astype(float), got.astype(float))

    expect = op(psr, pdf)
    got = op(psr, pdf)
    assert_eq(expect.astype(float), got.astype(float))


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.ne,
    ],
)
@pytest.mark.parametrize(
    "gsr", [gd.Series([1, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"])]
)
def test_df_sr_binop_col_order(gsr, op):
    colnames = [0, 1, 2]
    data = [[0, 2, 5], [3, None, 5], [6, 7, np.nan]]
    data = dict(zip(colnames, data))

    gdf = gd.DataFrame(data)
    pdf = pd.DataFrame.from_dict(data)

    psr = gsr.to_pandas()

    expect = op(pdf, psr).astype("float")
    out = op(gdf, gsr).astype("float")
    got = out[expect.columns]

    assert_eq(expect, got)


@pytest.mark.parametrize("set_index", [None, "A", "C", "D"])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("deep", [True, False])
def test_memory_usage(deep, index, set_index):
    # Testing numerical/datetime by comparing with pandas
    # (string and categorical columns will be different)
    rows = int(100)
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int64"),
            "B": np.arange(rows, dtype="int32"),
            "C": np.arange(rows, dtype="float64"),
        }
    )
    df["D"] = pd.to_datetime(df.A)
    if set_index:
        df = df.set_index(set_index)

    gdf = gd.from_pandas(df)

    if index and set_index is None:

        # Special Case: Assume RangeIndex size == 0
        assert gdf.index.memory_usage(deep=deep) == 0

    else:

        # Check for Series only
        assert df["B"].memory_usage(index=index, deep=deep) == gdf[
            "B"
        ].memory_usage(index=index, deep=deep)

        # Check for entire DataFrame
        assert_eq(
            df.memory_usage(index=index, deep=deep).sort_index(),
            gdf.memory_usage(index=index, deep=deep).sort_index(),
        )


@pytest.mark.xfail
def test_memory_usage_string():
    rows = int(100)
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": np.random.choice(["apple", "banana", "orange"], rows),
        }
    )
    gdf = gd.from_pandas(df)

    # Check deep=False (should match pandas)
    assert gdf.B.memory_usage(deep=False, index=False) == df.B.memory_usage(
        deep=False, index=False
    )

    # Check string column
    assert gdf.B.memory_usage(deep=True, index=False) == df.B.memory_usage(
        deep=True, index=False
    )

    # Check string index
    assert gdf.set_index("B").index.memory_usage(
        deep=True
    ) == df.B.memory_usage(deep=True, index=False)


def test_memory_usage_cat():
    rows = int(100)
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": np.random.choice(["apple", "banana", "orange"], rows),
        }
    )
    df["B"] = df.B.astype("category")
    gdf = gd.from_pandas(df)

    expected = (
        gdf.B._column.cat().categories.__sizeof__()
        + gdf.B._column.cat().codes.__sizeof__()
    )

    # Check cat column
    assert gdf.B.memory_usage(deep=True, index=False) == expected

    # Check cat index
    assert gdf.set_index("B").index.memory_usage(deep=True) == expected


@pytest.mark.xfail
def test_memory_usage_multi():
    rows = int(100)
    deep = True
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": np.random.choice(np.arange(3, dtype="int64"), rows),
            "C": np.random.choice(np.arange(3, dtype="float64"), rows),
        }
    ).set_index(["B", "C"])
    gdf = gd.from_pandas(df)

    # Assume MultiIndex memory footprint is just that
    # of the underlying columns, levels, and codes
    expect = rows * 16  # Source Columns
    expect += rows * 16  # Codes
    expect += 3 * 8  # Level 0
    expect += 3 * 8  # Level 1

    assert expect == gdf.index.memory_usage(deep=deep)


@pytest.mark.parametrize(
    "list_input",
    [
        pytest.param([1, 2, 3, 4], id="smaller"),
        pytest.param([1, 2, 3, 4, 5, 6], id="larger"),
    ],
)
@pytest.mark.parametrize(
    "key",
    [
        pytest.param("list_test", id="new_column"),
        pytest.param("id", id="existing_column"),
    ],
)
def test_setitem_diff_size_list(list_input, key):
    gdf = gd.datasets.randomdata(5)
    with pytest.raises(
        ValueError, match=("All values must be of equal length")
    ):
        gdf[key] = list_input


@pytest.mark.parametrize(
    "series_input",
    [
        pytest.param(gd.Series([1, 2, 3, 4]), id="smaller_cudf"),
        pytest.param(gd.Series([1, 2, 3, 4, 5, 6]), id="larger_cudf"),
        pytest.param(gd.Series([1, 2, 3], index=[4, 5, 6]), id="index_cudf"),
        pytest.param(pd.Series([1, 2, 3, 4]), id="smaller_pandas"),
        pytest.param(pd.Series([1, 2, 3, 4, 5, 6]), id="larger_pandas"),
        pytest.param(pd.Series([1, 2, 3], index=[4, 5, 6]), id="index_pandas"),
    ],
)
@pytest.mark.parametrize(
    "key",
    [
        pytest.param("list_test", id="new_column"),
        pytest.param("id", id="existing_column"),
    ],
)
def test_setitem_diff_size_series(series_input, key):
    gdf = gd.datasets.randomdata(5)
    pdf = gdf.to_pandas()

    pandas_input = series_input
    if isinstance(pandas_input, gd.Series):
        pandas_input = pandas_input.to_pandas()

    expect = pdf
    expect[key] = pandas_input

    got = gdf
    got[key] = series_input

    # Pandas uses NaN and typecasts to float64 if there's missing values on
    # alignment, so need to typecast to float64 for equality comparison
    expect = expect.astype("float64")
    got = got.astype("float64")

    assert_eq(expect, got)


def test_tupleize_cols_False_set():
    pdf = pd.DataFrame()
    gdf = gd.DataFrame()
    pdf[("a", "b")] = [1]
    gdf[("a", "b")] = [1]
    assert_eq(pdf, gdf)
    assert_eq(pdf.columns, gdf.columns)


def test_init_multiindex_from_dict():
    pdf = pd.DataFrame({("a", "b"): [1]})
    gdf = gd.DataFrame({("a", "b"): [1]})
    assert_eq(pdf, gdf)
    assert_eq(pdf.columns, gdf.columns)


def test_change_column_dtype_in_empty():
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = gd.from_pandas(pdf)
    assert_eq(pdf, gdf)
    pdf["b"] = pdf["b"].astype("int64")
    gdf["b"] = gdf["b"].astype("int64")
    assert_eq(pdf, gdf)


def test_dataframe_from_table_empty_index():
    from cudf._lib.table import Table

    df = gd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    odict = df._data
    tbl = Table(odict)

    result = gd.DataFrame._from_table(tbl)  # noqa: F841


@pytest.mark.parametrize("dtype", ["int64", "str"])
def test_dataframe_from_dictionary_series_same_name_index(dtype):
    pd_idx1 = pd.Index([1, 2, 0], name="test_index").astype(dtype)
    pd_idx2 = pd.Index([2, 0, 1], name="test_index").astype(dtype)
    pd_series1 = pd.Series([1, 2, 3], index=pd_idx1)
    pd_series2 = pd.Series([1, 2, 3], index=pd_idx2)

    gd_idx1 = gd.from_pandas(pd_idx1)
    gd_idx2 = gd.from_pandas(pd_idx2)
    gd_series1 = gd.Series([1, 2, 3], index=gd_idx1)
    gd_series2 = gd.Series([1, 2, 3], index=gd_idx2)

    expect = pd.DataFrame({"a": pd_series1, "b": pd_series2})
    got = gd.DataFrame({"a": gd_series1, "b": gd_series2})

    if dtype == "str":
        # Pandas actually loses its index name erroneously here...
        expect.index.name = "test_index"

    assert_eq(expect, got)
    assert expect.index.names == got.index.names


@pytest.mark.parametrize(
    "arg", [slice(2, 8, 3), slice(1, 20, 4), slice(-2, -6, -2)]
)
def test_dataframe_strided_slice(arg):
    mul = pd.DataFrame(
        {
            "Index": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "AlphaIndex": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        }
    )
    pdf = pd.DataFrame(
        {"Val": [10, 9, 8, 7, 6, 5, 4, 3, 2]},
        index=pd.MultiIndex.from_frame(mul),
    )
    gdf = gd.DataFrame.from_pandas(pdf)

    expect = pdf[arg]
    got = gdf[arg]

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,condition,other,error",
    [
        (pd.Series(range(5)), pd.Series(range(5)) > 0, None, None),
        (pd.Series(range(5)), pd.Series(range(5)) > 1, None, None),
        (pd.Series(range(5)), pd.Series(range(5)) > 1, 10, None),
        (
            pd.Series(range(5)),
            pd.Series(range(5)) > 1,
            pd.Series(range(5, 10)),
            None,
        ),
        (
            pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"]),
            (
                pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"])
                % 3
            )
            == 0,
            -pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"]),
            None,
        ),
        (
            pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]}),
            pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]}) == 4,
            None,
            None,
        ),
        (
            pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]}),
            pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]}) != 4,
            None,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [True, True, True],
            None,
            ValueError,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [True, True, True, False],
            None,
            ValueError,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [[True, True, True, False], [True, True, True, False]],
            None,
            ValueError,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [[True, True], [False, True], [True, False], [False, True]],
            None,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            cuda.to_device(
                np.array(
                    [[True, True], [False, True], [True, False], [False, True]]
                )
            ),
            None,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            cupy.array(
                [[True, True], [False, True], [True, False], [False, True]]
            ),
            17,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [[True, True], [False, True], [True, False], [False, True]],
            17,
            None,
        ),
        (
            pd.DataFrame({"p": [-2, 3, -4, -79], "k": [9, 10, 11, 12]}),
            [
                [True, True, False, True],
                [True, True, False, True],
                [True, True, False, True],
                [True, True, False, True],
            ],
            None,
            ValueError,
        ),
        (
            pd.Series([1, 2, np.nan]),
            pd.Series([1, 2, np.nan]) == 4,
            None,
            None,
        ),
        (
            pd.Series([1, 2, np.nan]),
            pd.Series([1, 2, np.nan]) != 4,
            None,
            None,
        ),
        (
            pd.Series([4, np.nan, 6]),
            pd.Series([4, np.nan, 6]) == 4,
            None,
            None,
        ),
        (
            pd.Series([4, np.nan, 6]),
            pd.Series([4, np.nan, 6]) != 4,
            None,
            None,
        ),
        (
            pd.Series([4, np.nan, 6], dtype="category"),
            pd.Series([4, np.nan, 6], dtype="category") != 4,
            None,
            None,
        ),
        (
            pd.Series(["a", "b", "b", "d", "c", "s"], dtype="category"),
            pd.Series(["a", "b", "b", "d", "c", "s"], dtype="category") == "b",
            None,
            None,
        ),
        (
            pd.Series(["a", "b", "b", "d", "c", "s"], dtype="category"),
            pd.Series(["a", "b", "b", "d", "c", "s"], dtype="category") == "b",
            "s",
            None,
        ),
        (
            pd.Series([1, 2, 3, 2, 5]),
            pd.Series([1, 2, 3, 2, 5]) == 2,
            pd.DataFrame(
                {
                    "a": pd.Series([1, 2, 3, 2, 5]),
                    "b": pd.Series([1, 2, 3, 2, 5]),
                }
            ),
            NotImplementedError,
        ),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_df_sr_mask_where(data, condition, other, error, inplace):
    ps_where = data
    gs_where = gd.from_pandas(data)

    ps_mask = ps_where.copy(deep=True)
    gs_mask = gs_where.copy(deep=True)

    if hasattr(condition, "__cuda_array_interface__"):
        if type(condition).__module__.split(".")[0] == "cupy":
            ps_condition = cupy.asnumpy(condition)
        else:
            ps_condition = np.array(condition).astype("bool")
    else:
        ps_condition = condition

    if type(condition).__module__.split(".")[0] == "pandas":
        gs_condition = gd.from_pandas(condition)
    else:
        gs_condition = condition

    ps_other = other
    if type(other).__module__.split(".")[0] == "pandas":
        gs_other = gd.from_pandas(other)
    else:
        gs_other = other

    if error is None:
        expect_where = ps_where.where(
            ps_condition, other=ps_other, inplace=inplace
        )
        got_where = gs_where.where(
            gs_condition, other=gs_other, inplace=inplace
        )

        expect_mask = ps_mask.mask(
            ps_condition, other=ps_other, inplace=inplace
        )
        got_mask = gs_mask.mask(gs_condition, other=gs_other, inplace=inplace)

        if inplace:
            expect_where = ps_where
            got_where = gs_where

            expect_mask = ps_mask
            got_mask = gs_mask

        if pd.api.types.is_categorical_dtype(expect_where):
            np.testing.assert_array_equal(
                expect_where.cat.codes,
                got_where.cat.codes.astype(expect_where.cat.codes.dtype)
                .fillna(-1)
                .to_array(),
            )
            assert_eq(expect_where.cat.categories, got_where.cat.categories)

            np.testing.assert_array_equal(
                expect_mask.cat.codes,
                got_mask.cat.codes.astype(expect_mask.cat.codes.dtype)
                .fillna(-1)
                .to_array(),
            )
            assert_eq(expect_mask.cat.categories, got_mask.cat.categories)
        else:
            assert_eq(
                expect_where.fillna(-1),
                got_where.fillna(-1),
                check_dtype=False,
            )
            assert_eq(
                expect_mask.fillna(-1), got_mask.fillna(-1), check_dtype=False
            )
    else:
        with pytest.raises(error):
            ps_where.where(ps_condition, other=ps_other, inplace=inplace)
        with pytest.raises(error):
            gs_where.where(gs_condition, other=gs_other, inplace=inplace)

        with pytest.raises(error):
            ps_mask.mask(ps_condition, other=ps_other, inplace=inplace)
        with pytest.raises(error):
            gs_mask.mask(gs_condition, other=gs_other, inplace=inplace)


@pytest.mark.parametrize(
    "data,condition,other,has_cat",
    [
        (
            pd.DataFrame(
                {
                    "a": pd.Series(["a", "a", "b", "c", "a", "d", "d", "a"]),
                    "b": pd.Series(["o", "p", "q", "e", "p", "p", "a", "a"]),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(["a", "a", "b", "c", "a", "d", "d", "a"]),
                    "b": pd.Series(["o", "p", "q", "e", "p", "p", "a", "a"]),
                }
            )
            != "a",
            None,
            None,
        ),
        (
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            )
            != "a",
            None,
            True,
        ),
        (
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            )
            == "a",
            None,
            True,
        ),
        (
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            )
            != "a",
            "a",
            True,
        ),
        (
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        ["a", "a", "b", "c", "a", "d", "d", "a"],
                        dtype="category",
                    ),
                    "b": pd.Series(
                        ["o", "p", "q", "e", "p", "p", "a", "a"],
                        dtype="category",
                    ),
                }
            )
            == "a",
            "a",
            True,
        ),
    ],
)
def test_df_string_cat_types_mask_where(data, condition, other, has_cat):
    ps = data
    gs = gd.from_pandas(data)

    ps_condition = condition
    if type(condition).__module__.split(".")[0] == "pandas":
        gs_condition = gd.from_pandas(condition)
    else:
        gs_condition = condition

    ps_other = other
    if type(other).__module__.split(".")[0] == "pandas":
        gs_other = gd.from_pandas(other)
    else:
        gs_other = other

    expect_where = ps.where(ps_condition, other=ps_other)
    got_where = gs.where(gs_condition, other=gs_other)

    expect_mask = ps.mask(ps_condition, other=ps_other)
    got_mask = gs.mask(gs_condition, other=gs_other)

    if has_cat is None:
        assert_eq(
            expect_where.fillna(-1).astype("str"),
            got_where.fillna(-1),
            check_dtype=False,
        )
        assert_eq(
            expect_mask.fillna(-1).astype("str"),
            got_mask.fillna(-1),
            check_dtype=False,
        )
    else:
        assert_eq(expect_where, got_where, check_dtype=False)
        assert_eq(expect_mask, got_mask, check_dtype=False)


@pytest.mark.parametrize(
    "data,expected_upcast_type,error",
    [
        (
            pd.Series([random.random() for _ in range(10)], dtype="float32"),
            np.dtype("float32"),
            None,
        ),
        (
            pd.Series([random.random() for _ in range(10)], dtype="float16"),
            np.dtype("float32"),
            None,
        ),
        (
            pd.Series([random.random() for _ in range(10)], dtype="float64"),
            np.dtype("float64"),
            None,
        ),
        (
            pd.Series([random.random() for _ in range(10)], dtype="float128"),
            None,
            NotImplementedError,
        ),
    ],
)
def test_from_pandas_unsupported_types(data, expected_upcast_type, error):
    pdf = pd.DataFrame({"one_col": data})
    if error == NotImplementedError:
        with pytest.raises(error):
            df = gd.from_pandas(data)

        with pytest.raises(error):
            df = gd.Series(data)

        with pytest.raises(error):
            df = gd.from_pandas(pdf)

        with pytest.raises(error):
            df = gd.DataFrame(pdf)
    else:
        df = gd.from_pandas(data)

        assert_eq(data, df, check_dtype=False)
        assert df.dtype == expected_upcast_type

        df = gd.Series(data)
        assert_eq(data, df, check_dtype=False)
        assert df.dtype == expected_upcast_type

        df = gd.from_pandas(pdf)
        assert_eq(pdf, df, check_dtype=False)
        assert df["one_col"].dtype == expected_upcast_type

        df = gd.DataFrame(pdf)
        assert_eq(pdf, df, check_dtype=False)
        assert df["one_col"].dtype == expected_upcast_type


@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.parametrize("index", [None, "a", ["a", "b"]])
def test_from_pandas_nan_as_null(nan_as_null, index):

    data = [np.nan, 2.0, 3.0]

    if index is None:
        pdf = pd.DataFrame({"a": data, "b": data})
        expected = gd.DataFrame(
            {
                "a": column.as_column(data, nan_as_null=nan_as_null),
                "b": column.as_column(data, nan_as_null=nan_as_null),
            }
        )
    else:
        pdf = pd.DataFrame({"a": data, "b": data}).set_index(index)
        expected = gd.DataFrame(
            {
                "a": column.as_column(data, nan_as_null=nan_as_null),
                "b": column.as_column(data, nan_as_null=nan_as_null),
            }
        )
        expected = gd.DataFrame(
            {
                "a": column.as_column(data, nan_as_null=nan_as_null),
                "b": column.as_column(data, nan_as_null=nan_as_null),
            }
        )
        expected = expected.set_index(index)

    got = gd.from_pandas(pdf, nan_as_null=nan_as_null)

    assert_eq(expected, got)


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_from_pandas_for_series_nan_as_null(nan_as_null):

    data = [np.nan, 2.0, 3.0]
    psr = pd.Series(data)

    expected = gd.Series(column.as_column(data, nan_as_null=nan_as_null))
    got = gd.from_pandas(psr, nan_as_null=nan_as_null)

    assert_eq(expected, got)


@pytest.mark.parametrize("copy", [True, False])
def test_df_series_dataframe_astype_copy(copy):
    gdf = gd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    pdf = gdf.to_pandas()

    assert_eq(
        gdf.astype(dtype="float", copy=copy),
        pdf.astype(dtype="float", copy=copy),
    )
    assert_eq(gdf, pdf)

    gsr = gd.Series([1, 2])
    psr = gsr.to_pandas()

    assert_eq(
        gsr.astype(dtype="float", copy=copy),
        psr.astype(dtype="float", copy=copy),
    )
    assert_eq(gsr, psr)

    gsr = gd.Series([1, 2])
    psr = gsr.to_pandas()

    actual = gsr.astype(dtype="int64", copy=copy)
    expected = psr.astype(dtype="int64", copy=copy)
    assert_eq(expected, actual)
    assert_eq(gsr, psr)
    actual[0] = 3
    expected[0] = 3
    assert_eq(gsr, psr)


@pytest.mark.parametrize("copy", [True, False])
def test_df_series_dataframe_astype_dtype_dict(copy):
    gdf = gd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    pdf = gdf.to_pandas()

    assert_eq(
        gdf.astype(dtype={"col1": "float"}, copy=copy),
        pdf.astype(dtype={"col1": "float"}, copy=copy),
    )
    assert_eq(gdf, pdf)

    gsr = gd.Series([1, 2])
    psr = gsr.to_pandas()

    assert_eq(
        gsr.astype(dtype={None: "float"}, copy=copy),
        psr.astype(dtype={None: "float"}, copy=copy),
    )
    assert_eq(gsr, psr)

    with pytest.raises(KeyError):
        gsr.astype(dtype={"a": "float"}, copy=copy)

    with pytest.raises(KeyError):
        psr.astype(dtype={"a": "float"}, copy=copy)

    gsr = gd.Series([1, 2])
    psr = gsr.to_pandas()

    actual = gsr.astype({None: "int64"}, copy=copy)
    expected = psr.astype({None: "int64"}, copy=copy)
    assert_eq(expected, actual)
    assert_eq(gsr, psr)

    actual[0] = 3
    expected[0] = 3
    assert_eq(gsr, psr)


@pytest.mark.parametrize(
    "data,columns",
    [
        ([1, 2, 3, 100, 112, 35464], ["a"]),
        (range(100), None),
        ([], None),
        ((-10, 21, 32, 32, 1, 2, 3), ["p"]),
        ((), None),
        ([[1, 2, 3], [1, 2, 3]], ["col1", "col2", "col3"]),
        ([range(100), range(100)], ["range" + str(i) for i in range(100)]),
        (((1, 2, 3), (1, 2, 3)), ["tuple0", "tuple1", "tuple2"]),
        ([[1, 2, 3]], ["list col1", "list col2", "list col3"]),
        ([range(100)], ["range" + str(i) for i in range(100)]),
        (((1, 2, 3),), ["k1", "k2", "k3"]),
    ],
)
def test_dataframe_init_1d_list(data, columns):
    expect = pd.DataFrame(data, columns=columns)
    actual = gd.DataFrame(data, columns=columns)

    assert_eq(
        expect, actual, check_index_type=False if len(data) == 0 else True
    )

    expect = pd.DataFrame(data, columns=None)
    actual = gd.DataFrame(data, columns=None)

    assert_eq(
        expect, actual, check_index_type=False if len(data) == 0 else True
    )


@pytest.mark.parametrize(
    "data,cols,index",
    [
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            ["a", "b", "c", "d"],
        ),
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            [0, 20, 30, 10],
        ),
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            [0, 1, 2, 3],
        ),
        (np.array([11, 123, -2342, 232]), ["a"], [1, 2, 11, 12]),
        (np.array([11, 123, -2342, 232]), ["a"], ["khsdjk", "a", "z", "kk"]),
        (
            cupy.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "z"],
            ["a", "z", "a", "z"],
        ),
        (cupy.array([11, 123, -2342, 232]), ["z"], [0, 1, 1, 0]),
        (cupy.array([11, 123, -2342, 232]), ["z"], [1, 2, 3, 4]),
        (cupy.array([11, 123, -2342, 232]), ["z"], ["a", "z", "d", "e"]),
        (np.random.randn(2, 4), ["a", "b", "c", "d"], ["a", "b"]),
        (np.random.randn(2, 4), ["a", "b", "c", "d"], [1, 0]),
        (cupy.random.randn(2, 4), ["a", "b", "c", "d"], ["a", "b"]),
        (cupy.random.randn(2, 4), ["a", "b", "c", "d"], [1, 0]),
    ],
)
def test_dataframe_init_from_arrays_cols(data, cols, index):

    gd_data = data
    if isinstance(data, cupy.core.ndarray):
        # pandas can't handle cupy arrays in general
        pd_data = data.get()

        # additional test for building DataFrame with gpu array whose
        # cuda array interface has no `descr` attribute
        numba_data = cuda.as_cuda_array(data)
    else:
        pd_data = data
        numba_data = None

    # verify with columns & index
    pdf = pd.DataFrame(pd_data, columns=cols, index=index)
    gdf = gd.DataFrame(gd_data, columns=cols, index=index)

    assert_eq(pdf, gdf, check_dtype=False)

    # verify with columns
    pdf = pd.DataFrame(pd_data, columns=cols)
    gdf = gd.DataFrame(gd_data, columns=cols)

    assert_eq(pdf, gdf, check_dtype=False)

    pdf = pd.DataFrame(pd_data)
    gdf = gd.DataFrame(gd_data)

    assert_eq(pdf, gdf, check_dtype=False)

    if numba_data is not None:
        gdf = gd.DataFrame(numba_data)
        assert_eq(pdf, gdf, check_dtype=False)


@pytest.mark.parametrize(
    "col_data",
    [
        range(5),
        ["a", "b", "x", "y", "z"],
        [1.0, 0.213, 0.34332],
        ["a"],
        [1],
        [0.2323],
        [],
    ],
)
@pytest.mark.parametrize(
    "assign_val",
    [
        1,
        2,
        np.array(2),
        cupy.array(2),
        0.32324,
        np.array(0.34248),
        cupy.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
def test_dataframe_assign_scalar(col_data, assign_val):
    pdf = pd.DataFrame({"a": col_data})
    gdf = gd.DataFrame({"a": col_data})

    pdf["b"] = (
        cupy.asnumpy(assign_val)
        if isinstance(assign_val, cupy.ndarray)
        else assign_val
    )
    gdf["b"] = assign_val

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "col_data",
    [
        1,
        2,
        np.array(2),
        cupy.array(2),
        0.32324,
        np.array(0.34248),
        cupy.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
@pytest.mark.parametrize(
    "assign_val",
    [
        1,
        2,
        np.array(2),
        cupy.array(2),
        0.32324,
        np.array(0.34248),
        cupy.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
def test_dataframe_assign_scalar_with_scalar_cols(col_data, assign_val):
    pdf = pd.DataFrame(
        {
            "a": cupy.asnumpy(col_data)
            if isinstance(col_data, cupy.ndarray)
            else col_data
        },
        index=["dummy_mandatory_index"],
    )
    gdf = gd.DataFrame({"a": col_data}, index=["dummy_mandatory_index"])

    pdf["b"] = (
        cupy.asnumpy(assign_val)
        if isinstance(assign_val, cupy.ndarray)
        else assign_val
    )
    gdf["b"] = assign_val

    assert_eq(pdf, gdf)


def test_dataframe_info_basic():

    buffer = io.StringIO()
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    StringIndex: 10 entries, a to 1111
    Data columns (total 10 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   0       10 non-null     float64
     1   1       10 non-null     float64
     2   2       10 non-null     float64
     3   3       10 non-null     float64
     4   4       10 non-null     float64
     5   5       10 non-null     float64
     6   6       10 non-null     float64
     7   7       10 non-null     float64
     8   8       10 non-null     float64
     9   9       10 non-null     float64
    dtypes: float64(10)
    memory usage: 859.0+ bytes
    """
    )
    df = pd.DataFrame(
        np.random.randn(10, 10),
        index=["a", "2", "3", "4", "5", "6", "7", "8", "100", "1111"],
    )
    gd.from_pandas(df).info(buf=buffer, verbose=True)
    s = buffer.getvalue()
    assert str_cmp == s


def test_dataframe_info_verbose_mem_usage():
    buffer = io.StringIO()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["safdas", "assa", "asdasd"]})
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       3 non-null      int64
     1   b       3 non-null      object
    dtypes: int64(1), object(1)
    memory usage: 56.0+ bytes
    """
    )
    gd.from_pandas(df).info(buf=buffer, verbose=True)
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Columns: 2 entries, a to b
    dtypes: int64(1), object(1)
    memory usage: 56.0+ bytes
    """
    )
    gd.from_pandas(df).info(buf=buffer, verbose=False)
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["safdas", "assa", "asdasd"]},
        index=["sdfdsf", "sdfsdfds", "dsfdf"],
    )
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    StringIndex: 3 entries, sdfdsf to dsfdf
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       3 non-null      int64
     1   b       3 non-null      object
    dtypes: int64(1), object(1)
    memory usage: 91.0 bytes
    """
    )
    gd.from_pandas(df).info(buf=buffer, verbose=True, memory_usage="deep")
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    int_values = [1, 2, 3, 4, 5]
    text_values = ["alpha", "beta", "gamma", "delta", "epsilon"]
    float_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    df = gd.DataFrame(
        {
            "int_col": int_values,
            "text_col": text_values,
            "float_col": float_values,
        }
    )
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   int_col    5 non-null      int64
     1   text_col   5 non-null      object
     2   float_col  5 non-null      float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 130.0 bytes
    """
    )
    df.info(buf=buffer, verbose=True, memory_usage="deep")
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)


def test_dataframe_info_null_counts():
    int_values = [1, 2, 3, 4, 5]
    text_values = ["alpha", "beta", "gamma", "delta", "epsilon"]
    float_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    df = gd.DataFrame(
        {
            "int_col": int_values,
            "text_col": text_values,
            "float_col": float_values,
        }
    )
    buffer = io.StringIO()
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Dtype
    ---  ------     -----
     0   int_col    int64
     1   text_col   object
     2   float_col  float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 130.0+ bytes
    """
    )
    df.info(buf=buffer, verbose=True, null_counts=False)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df.info(buf=buffer, verbose=True, max_cols=0)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df = gd.DataFrame()

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 0 entries
    Empty DataFrame"""
    )
    df.info(buf=buffer, verbose=True)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df = gd.DataFrame(
        {
            "a": [1, 2, 3, None, 10, 11, 12, None],
            "b": ["a", "b", "c", "sd", "sdf", "sd", None, None],
        }
    )

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 2 columns):
     #   Column  Dtype
    ---  ------  -----
     0   a       int64
     1   b       object
    dtypes: int64(1), object(1)
    memory usage: 238.0+ bytes
    """
    )
    pd.options.display.max_info_rows = 2
    df.info(buf=buffer, max_cols=2, null_counts=None)
    pd.reset_option("display.max_info_rows")
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       6 non-null      int64
     1   b       6 non-null      object
    dtypes: int64(1), object(1)
    memory usage: 238.0+ bytes
    """
    )

    df.info(buf=buffer, max_cols=2, null_counts=None)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df.info(buf=buffer, null_counts=True)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string


@pytest.mark.parametrize(
    "data1",
    [
        [1, 2, 3, 4, 5, 6, 7],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [
            1.9876543,
            2.9876654,
            3.9876543,
            4.1234587,
            5.23,
            6.88918237,
            7.00001,
        ],
        [
            -1.9876543,
            -2.9876654,
            -3.9876543,
            -4.1234587,
            -5.23,
            -6.88918237,
            -7.00001,
        ],
        [
            1.987654321,
            2.987654321,
            3.987654321,
            0.1221,
            2.1221,
            0.112121,
            -21.1212,
        ],
        [
            -1.987654321,
            -2.987654321,
            -3.987654321,
            -0.1221,
            -2.1221,
            -0.112121,
            21.1212,
        ],
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        [1, 2, 3, 4, 5, 6, 7],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [
            1.9876543,
            2.9876654,
            3.9876543,
            4.1234587,
            5.23,
            6.88918237,
            7.00001,
        ],
        [
            -1.9876543,
            -2.9876654,
            -3.9876543,
            -4.1234587,
            -5.23,
            -6.88918237,
            -7.00001,
        ],
        [
            1.987654321,
            2.987654321,
            3.987654321,
            0.1221,
            2.1221,
            0.112121,
            -21.1212,
        ],
        [
            -1.987654321,
            -2.987654321,
            -3.987654321,
            -0.1221,
            -2.1221,
            -0.112121,
            21.1212,
        ],
    ],
)
@pytest.mark.parametrize("rtol", [0, 0.01, 1e-05, 1e-08, 5e-1, 50.12])
@pytest.mark.parametrize("atol", [0, 0.01, 1e-05, 1e-08, 50.12])
def test_cudf_isclose(data1, data2, rtol, atol):
    array1 = cupy.array(data1)
    array2 = cupy.array(data2)

    expected = gd.Series(cupy.isclose(array1, array2, rtol=rtol, atol=atol))

    actual = gd.isclose(
        gd.Series(data1), gd.Series(data2), rtol=rtol, atol=atol
    )

    assert_eq(expected, actual)
    actual = gd.isclose(data1, data2, rtol=rtol, atol=atol)

    assert_eq(expected, actual)

    actual = gd.isclose(
        cupy.array(data1), cupy.array(data2), rtol=rtol, atol=atol
    )

    assert_eq(expected, actual)

    actual = gd.isclose(np.array(data1), np.array(data2), rtol=rtol, atol=atol)

    assert_eq(expected, actual)

    actual = gd.isclose(
        pd.Series(data1), pd.Series(data2), rtol=rtol, atol=atol
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data1",
    [
        [
            -1.9876543,
            -2.9876654,
            np.nan,
            -4.1234587,
            -5.23,
            -6.88918237,
            -7.00001,
        ],
        [
            1.987654321,
            2.987654321,
            3.987654321,
            0.1221,
            2.1221,
            np.nan,
            -21.1212,
        ],
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        [
            -1.9876543,
            -2.9876654,
            -3.9876543,
            -4.1234587,
            -5.23,
            -6.88918237,
            -7.00001,
        ],
        [
            1.987654321,
            2.987654321,
            3.987654321,
            0.1221,
            2.1221,
            0.112121,
            -21.1212,
        ],
        [
            -1.987654321,
            -2.987654321,
            -3.987654321,
            np.nan,
            np.nan,
            np.nan,
            21.1212,
        ],
    ],
)
@pytest.mark.parametrize("equal_nan", [True, False])
def test_cudf_isclose_nulls(data1, data2, equal_nan):
    array1 = cupy.array(data1)
    array2 = cupy.array(data2)

    expected = gd.Series(cupy.isclose(array1, array2, equal_nan=equal_nan))

    actual = gd.isclose(
        gd.Series(data1), gd.Series(data2), equal_nan=equal_nan
    )
    assert_eq(expected, actual, check_dtype=False)
    actual = gd.isclose(data1, data2, equal_nan=equal_nan)
    assert_eq(expected, actual, check_dtype=False)


def test_cudf_isclose_different_index():
    s1 = gd.Series(
        [-1.9876543, -2.9876654, -3.9876543, -4.1234587, -5.23, -7.00001],
        index=[0, 1, 2, 3, 4, 5],
    )
    s2 = gd.Series(
        [-1.9876543, -2.9876654, -7.00001, -4.1234587, -5.23, -3.9876543],
        index=[0, 1, 5, 3, 4, 2],
    )

    expected = gd.Series([True] * 6, index=s1.index)
    assert_eq(expected, gd.isclose(s1, s2))

    s1 = gd.Series(
        [-1.9876543, -2.9876654, -3.9876543, -4.1234587, -5.23, -7.00001],
        index=[0, 1, 2, 3, 4, 5],
    )
    s2 = gd.Series(
        [-1.9876543, -2.9876654, -7.00001, -4.1234587, -5.23, -3.9876543],
        index=[0, 1, 5, 10, 4, 2],
    )

    expected = gd.Series([True, True, True, False, True, True], index=s1.index)
    assert_eq(expected, gd.isclose(s1, s2))

    s1 = gd.Series(
        [-1.9876543, -2.9876654, -3.9876543, -4.1234587, -5.23, -7.00001],
        index=[100, 1, 2, 3, 4, 5],
    )
    s2 = gd.Series(
        [-1.9876543, -2.9876654, -7.00001, -4.1234587, -5.23, -3.9876543],
        index=[0, 1, 100, 10, 4, 2],
    )

    expected = gd.Series(
        [False, True, True, False, True, False], index=s1.index
    )
    assert_eq(expected, gd.isclose(s1, s2))


def test_dataframe_to_dict_error():
    df = gd.DataFrame({"a": [1, 2, 3], "b": [9, 5, 3]})
    with pytest.raises(
        TypeError,
        match=re.escape(
            r"cuDF does not support conversion to host memory "
            r"via `to_dict()` method. Consider using "
            r"`.to_pandas().to_dict()` to construct a Python dictionary."
        ),
    ):
        df.to_dict()

    with pytest.raises(
        TypeError,
        match=re.escape(
            r"cuDF does not support conversion to host memory "
            r"via `to_dict()` method. Consider using "
            r"`.to_pandas().to_dict()` to construct a Python dictionary."
        ),
    ):
        df["a"].to_dict()


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, 3, 4, 5, 10, 11, 12, 33, 55, 19]}),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            }
        ),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            },
            index=[10, 20, 30, 40, 50, 60],
        ),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            },
            index=["a", "b", "c", "d", "e", "f"],
        ),
        pd.DataFrame(index=["a", "b", "c", "d", "e", "f"]),
        pd.DataFrame(columns=["a", "b", "c", "d", "e", "f"]),
        pd.DataFrame(index=[10, 11, 12]),
        pd.DataFrame(columns=[10, 11, 12]),
        pd.DataFrame(),
        pd.DataFrame({"one": [], "two": []}),
        pd.DataFrame({2: [], 1: []}),
        pd.DataFrame(
            {
                0: [1, 2, 3, 4, 5, 10],
                1: ["abc", "def", "ghi", "xyz", "pqr", "abc"],
                100: ["a", "b", "b", "x", "z", "a"],
            },
            index=[10, 20, 30, 40, 50, 60],
        ),
    ],
)
def test_dataframe_keys(df):
    gdf = gd.from_pandas(df)

    assert_eq(df.keys(), gdf.keys())


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series([1, 2, 3, 4, 5, 10, 11, 12, 33, 55, 19]),
        pd.Series(["abc", "def", "ghi", "xyz", "pqr", "abc"]),
        pd.Series(
            [1, 2, 3, 4, 5, 10],
            index=["abc", "def", "ghi", "xyz", "pqr", "abc"],
        ),
        pd.Series(
            ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            index=[1, 2, 3, 4, 5, 10],
        ),
        pd.Series(index=["a", "b", "c", "d", "e", "f"]),
        pd.Series(index=[10, 11, 12]),
        pd.Series(),
        pd.Series([]),
    ],
)
def test_series_keys(ps):
    gds = gd.from_pandas(ps)

    if len(ps) == 0 and not isinstance(ps.index, pd.RangeIndex):
        assert_eq(ps.keys().astype("float64"), gds.keys())
    else:
        assert_eq(ps.keys(), gds.keys())


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("BD")),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("DE")),
        pd.DataFrame(),
        pd.DataFrame(
            {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
        ),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[200]),
        pd.DataFrame([]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([], index=[100]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[0, 100, 200, 300],
        ),
    ],
)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_dataframe_append_dataframe(df, other, sort, ignore_index):
    pdf = df
    other_pd = other

    gdf = gd.from_pandas(df)
    other_gd = gd.from_pandas(other)

    expected = pdf.append(other_pd, sort=sort, ignore_index=ignore_index)
    actual = gdf.append(other_gd, sort=sort, ignore_index=ignore_index)

    if expected.shape != df.shape:
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(
            expected, actual, check_index_type=False if gdf.empty else True
        )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({12: [], 22: []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=[0, 1], index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=[1, 0], index=[7, 8]),
        pd.DataFrame(
            {
                23: [315.3324, 3243.32432, 3232.332, -100.32],
                33: [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                0: [315.3324, 3243.32432, 3232.332, -100.32],
                1: [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.Series([10, 11, 23, 234, 13]),
        pytest.param(
            pd.Series([10, 11, 23, 234, 13], index=[11, 12, 13, 44, 33]),
            marks=pytest.mark.xfail(
                reason="pandas bug: "
                "https://github.com/pandas-dev/pandas/issues/35092"
            ),
        ),
        {1: 1},
        {0: 10, 1: 100, 2: 102},
    ],
)
@pytest.mark.parametrize("sort", [False, True])
def test_dataframe_append_series_dict(df, other, sort):
    pdf = df
    other_pd = other

    gdf = gd.from_pandas(df)
    if isinstance(other, pd.Series):
        other_gd = gd.from_pandas(other)
    else:
        other_gd = other

    expected = pdf.append(other_pd, ignore_index=True, sort=sort)
    actual = gdf.append(other_gd, ignore_index=True, sort=sort)

    if expected.shape != df.shape:
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(
            expected, actual, check_index_type=False if gdf.empty else True
        )


def test_dataframe_append_series_mixed_index():
    df = gd.DataFrame({"first": [], "d": []})
    sr = gd.Series([1, 2, 3, 4])

    with pytest.raises(
        TypeError,
        match=re.escape(
            "cudf does not support mixed types, please type-cast "
            "the column index of dataframe and index of series "
            "to same dtypes."
        ),
    ):
        df.append(sr, ignore_index=True)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [pd.DataFrame([[5, 6], [7, 8]], columns=list("AB"))],
        [
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("BD")),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("DE")),
        ],
        [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        [
            pd.DataFrame(
                {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        ],
        [
            pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"l": [10]}),
            pd.DataFrame({"l": [10]}, index=[200]),
        ],
        [pd.DataFrame([]), pd.DataFrame([], index=[100])],
        [
            pd.DataFrame([]),
            pd.DataFrame([], index=[100]),
            pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                }
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        ],
    ],
)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_dataframe_append_dataframe_lists(df, other, sort, ignore_index):
    pdf = df
    other_pd = other

    gdf = gd.from_pandas(df)
    other_gd = [
        gd.from_pandas(o) if isinstance(o, pd.DataFrame) else o for o in other
    ]

    expected = pdf.append(other_pd, sort=sort, ignore_index=ignore_index)
    actual = gdf.append(other_gd, sort=sort, ignore_index=ignore_index)
    if expected.shape != df.shape:
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(
            expected, actual, check_index_type=False if gdf.empty else True
        )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [[1, 2], [10, 100]],
        [[1, 2, 10, 100, 0.1, 0.2, 0.0021]],
        [[]],
        [[], [], [], []],
        [[0.23, 0.00023, -10.00, 100, 200, 1000232, 1232.32323]],
    ],
)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_dataframe_append_lists(df, other, sort, ignore_index):
    pdf = df
    other_pd = other

    gdf = gd.from_pandas(df)
    other_gd = [
        gd.from_pandas(o) if isinstance(o, pd.DataFrame) else o for o in other
    ]

    expected = pdf.append(other_pd, sort=sort, ignore_index=ignore_index)
    actual = gdf.append(other_gd, sort=sort, ignore_index=ignore_index)

    if expected.shape != df.shape:
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(
            expected, actual, check_index_type=False if gdf.empty else True
        )


def test_dataframe_append_error():
    df = gd.DataFrame({"a": [1, 2, 3]})
    ps = gd.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match="Can only append a Series if ignore_index=True "
        "or if the Series has a name",
    ):
        df.append(ps)


def test_cudf_arrow_array_error():
    df = gd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow Table via __arrow_array__"
        " is not allowed, To explicitly construct a PyArrow Table, consider "
        "using .to_arrow()",
    ):
        df.__arrow_array__()

    sr = gd.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow Array via __arrow_array__"
        " is not allowed, To explicitly construct a PyArrow Array, consider "
        "using .to_arrow()",
    ):
        sr.__arrow_array__()

    sr = gd.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow Array via __arrow_array__"
        " is not allowed, To explicitly construct a PyArrow Array, consider "
        "using .to_arrow()",
    ):
        sr.__arrow_array__()


@pytest.mark.parametrize("n", [0, 2, 5, 10, None])
@pytest.mark.parametrize("frac", [0.1, 0.5, 1, 2, None])
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("axis", [0, 1])
def test_dataframe_sample_basic(n, frac, replace, axis):
    # as we currently don't support column with same name
    if axis == 1 and replace:
        return
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "float": [0.05, 0.2, 0.3, 0.2, 0.25],
            "int": [1, 3, 5, 4, 2],
        },
        index=[1, 2, 3, 4, 5],
    )
    df = gd.DataFrame.from_pandas(pdf)
    random_state = 0

    kind = None

    try:
        pout = pdf.sample(
            n=n,
            frac=frac,
            replace=replace,
            random_state=random_state,
            axis=axis,
        )
    except BaseException as e:
        kind = type(e)
        msg = str(e)

    if kind is not None:
        with pytest.raises(kind, match=msg):
            gout = df.sample(
                n=n,
                frac=frac,
                replace=replace,
                random_state=random_state,
                axis=axis,
            )
    else:
        gout = df.sample(
            n=n,
            frac=frac,
            replace=replace,
            random_state=random_state,
            axis=axis,
        )

    if kind is not None:
        return

    assert pout.shape == gout.shape


@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("random_state", [1, np.random.mtrand.RandomState(10)])
def test_dataframe_reproducibility(replace, random_state):
    df = gd.DataFrame({"a": cupy.arange(0, 1024)})

    expected = df.sample(1024, replace=replace, random_state=random_state)
    out = df.sample(1024, replace=replace, random_state=random_state)

    assert_eq(expected, out)


@pytest.mark.parametrize("n", [0, 2, 5, 10, None])
@pytest.mark.parametrize("frac", [0.1, 0.5, 1, 2, None])
@pytest.mark.parametrize("replace", [True, False])
def test_series_sample_basic(n, frac, replace):
    psr = pd.Series([1, 2, 3, 4, 5])
    sr = gd.Series.from_pandas(psr)
    random_state = 0

    kind = None

    try:
        pout = psr.sample(
            n=n, frac=frac, replace=replace, random_state=random_state
        )
    except BaseException as e:
        kind = type(e)
        msg = str(e)

    if kind is not None:
        with pytest.raises(kind, match=msg):
            gout = sr.sample(
                n=n, frac=frac, replace=replace, random_state=random_state
            )
    else:
        gout = sr.sample(
            n=n, frac=frac, replace=replace, random_state=random_state
        )

    if kind is not None:
        return

    assert pout.shape == gout.shape


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[100, 10, 1, 0]),
        pd.DataFrame(columns=["a", "b", "c", "d"]),
        pd.DataFrame(columns=["a", "b", "c", "d"], index=[100]),
        pd.DataFrame(
            columns=["a", "b", "c", "d"], index=[100, 10000, 2131, 133]
        ),
        pd.DataFrame({"a": [1, 2, 3], "b": ["abc", "xyz", "klm"]}),
    ],
)
def test_dataframe_empty(df):
    pdf = df
    gdf = gd.from_pandas(pdf)

    assert_eq(pdf.empty, gdf.empty)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[100, 10, 1, 0]),
        pd.DataFrame(columns=["a", "b", "c", "d"]),
        pd.DataFrame(columns=["a", "b", "c", "d"], index=[100]),
        pd.DataFrame(
            columns=["a", "b", "c", "d"], index=[100, 10000, 2131, 133]
        ),
        pd.DataFrame({"a": [1, 2, 3], "b": ["abc", "xyz", "klm"]}),
    ],
)
def test_dataframe_size(df):
    pdf = df
    gdf = gd.from_pandas(pdf)

    assert_eq(pdf.size, gdf.size)


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(),
        pd.Series(index=[100, 10, 1, 0]),
        pd.Series([]),
        pd.Series(["a", "b", "c", "d"]),
        pd.Series(["a", "b", "c", "d"], index=[0, 1, 10, 11]),
    ],
)
def test_series_empty(ps):
    ps = ps
    gs = gd.from_pandas(ps)

    assert_eq(ps.empty, gs.empty)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1],
        {"a": [10, 11, 12]},
        {
            "a": [10, 11, 12],
            "another column name": [12, 22, 34],
            "xyz": [0, 10, 11],
        },
    ],
)
@pytest.mark.parametrize("columns", [["a"], ["another column name"], None])
def test_dataframe_init_with_columns(data, columns):
    pdf = pd.DataFrame(data, columns=columns)
    gdf = gd.DataFrame(data, columns=columns)

    assert_eq(
        pdf,
        gdf,
        check_index_type=False if len(pdf.index) == 0 else True,
        check_dtype=False if pdf.empty and len(pdf.columns) else True,
    )


@pytest.mark.parametrize(
    "data, ignore_dtype",
    [
        ([pd.Series([1, 2, 3])], False),
        ([pd.Series(index=[1, 2, 3])], False),
        ([pd.Series(name="empty series name")], False),
        ([pd.Series([1]), pd.Series([]), pd.Series([3])], False),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([]),
                pd.Series([3], name="series that is named"),
            ],
            False,
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, False),
        ([pd.Series([1, 2, 3], name=None, index=[10, 11, 12])] * 10, False),
        (
            [
                pd.Series([1, 2, 3], name=None, index=[10, 11, 12]),
                pd.Series([1, 2, 30], name=None, index=[13, 144, 15]),
            ],
            True,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([]),
                pd.Series(index=[10, 11, 12]),
            ],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], name="abc"),
                pd.Series(index=[10, 11, 12]),
            ],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([1, -100, 200, -399, 400], name="abc"),
                pd.Series([111, 222, 333], index=[10, 11, 12]),
            ],
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    "columns", [None, ["0"], [0], ["abc"], [144, 13], [2, 1, 0]]
)
def test_dataframe_init_from_series_list(data, ignore_dtype, columns):
    gd_data = [gd.from_pandas(obj) for obj in data]

    expected = pd.DataFrame(data, columns=columns)
    actual = gd.DataFrame(gd_data, columns=columns)

    if ignore_dtype:
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data, ignore_dtype, index",
    [
        ([pd.Series([1, 2, 3])], False, ["a", "b", "c"]),
        ([pd.Series(index=[1, 2, 3])], False, ["a", "b"]),
        ([pd.Series(name="empty series name")], False, ["index1"]),
        (
            [pd.Series([1]), pd.Series([]), pd.Series([3])],
            False,
            ["0", "2", "1"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([]),
                pd.Series([3], name="series that is named"),
            ],
            False,
            ["_", "+", "*"],
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, False, ["mean"] * 10),
        (
            [pd.Series([1, 2, 3], name=None, index=[10, 11, 12])] * 10,
            False,
            ["abc"] * 10,
        ),
        (
            [
                pd.Series([1, 2, 3], name=None, index=[10, 11, 12]),
                pd.Series([1, 2, 30], name=None, index=[13, 144, 15]),
            ],
            True,
            ["set_index_a", "set_index_b"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([]),
                pd.Series(index=[10, 11, 12]),
            ],
            False,
            ["a", "b", "c"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], name="abc"),
                pd.Series(index=[10, 11, 12]),
            ],
            False,
            ["a", "v", "z"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([1, -100, 200, -399, 400], name="abc"),
                pd.Series([111, 222, 333], index=[10, 11, 12]),
            ],
            False,
            ["a", "v", "z"],
        ),
    ],
)
@pytest.mark.parametrize(
    "columns", [None, ["0"], [0], ["abc"], [144, 13], [2, 1, 0]]
)
def test_dataframe_init_from_series_list_with_index(
    data, ignore_dtype, index, columns
):
    gd_data = [gd.from_pandas(obj) for obj in data]

    expected = pd.DataFrame(data, columns=columns, index=index)
    actual = gd.DataFrame(gd_data, columns=columns, index=index)

    if ignore_dtype:
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data, index",
    [
        ([pd.Series([1, 2]), pd.Series([1, 2])], ["a", "b", "c"]),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([]),
                pd.Series([3], name="series that is named"),
            ],
            ["_", "+"],
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, ["mean"] * 9),
    ],
)
def test_dataframe_init_from_series_list_with_index_error(data, index):
    gd_data = [gd.from_pandas(obj) for obj in data]
    try:
        pd.DataFrame(data, index=index)
    except Exception as e:
        with pytest.raises(type(e), match=re.escape(str(e))):
            gd.DataFrame(gd_data, index=index)
    else:
        raise AssertionError(
            "expected pd.DataFrame to because of index mismatch "
            "with data dimensions"
        )


@pytest.mark.parametrize(
    "data",
    [
        [pd.Series([1, 2, 3], index=["a", "a", "a"])],
        [pd.Series([1, 2, 3], index=["a", "a", "a"])] * 4,
        [
            pd.Series([1, 2, 3], index=["a", "b", "a"]),
            pd.Series([1, 2, 3], index=["b", "b", "a"]),
        ],
        [
            pd.Series([1, 2, 3], index=["a", "b", "z"]),
            pd.Series([1, 2, 3], index=["u", "b", "a"]),
            pd.Series([1, 2, 3], index=["u", "b", "u"]),
        ],
    ],
)
def test_dataframe_init_from_series_list_duplicate_index_error(data):
    gd_data = [gd.from_pandas(obj) for obj in data]
    try:
        pd.DataFrame(data)
    except Exception as e:
        with pytest.raises(ValueError, match=re.escape(str(e))):
            gd.DataFrame(gd_data)
    else:
        raise AssertionError(
            "expected pd.DataFrame to because of duplicates in index"
        )


def test_dataframe_iterrows_itertuples():
    df = gd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(
        TypeError,
        match=re.escape(
            "cuDF does not support iteration of DataFrame "
            "via itertuples. Consider using "
            "`.to_pandas().itertuples()` "
            "if you wish to iterate over namedtuples."
        ),
    ):
        df.itertuples()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "cuDF does not support iteration of DataFrame "
            "via iterrows. Consider using "
            "`.to_pandas().iterrows()` "
            "if you wish to iterate over each row."
        ),
    ):
        df.iterrows()


@pytest.mark.parametrize(
    "df",
    [
        gd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [10, 22, 33],
                "c": [0.3234, 0.23432, 0.0],
                "d": ["hello", "world", "hello"],
            }
        ),
        gd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["hello", "world", "hello"],
                "c": [0.3234, 0.23432, 0.0],
            }
        ),
        pytest.param(
            gd.DataFrame(
                {
                    "int_data": [1, 2, 3],
                    "str_data": ["hello", "world", "hello"],
                    "float_data": [0.3234, 0.23432, 0.0],
                    "timedelta_data": gd.Series(
                        [1, 2, 1], dtype="timedelta64[ns]"
                    ),
                    "datetime_data": gd.Series(
                        [1, 2, 1], dtype="datetime64[ns]"
                    ),
                }
            ),
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6219"
            ),
        ),
        pytest.param(
            gd.DataFrame(
                {
                    "int_data": [1, 2, 3],
                    "str_data": ["hello", "world", "hello"],
                    "float_data": [0.3234, 0.23432, 0.0],
                    "timedelta_data": gd.Series(
                        [1, 2, 1], dtype="timedelta64[ns]"
                    ),
                    "datetime_data": gd.Series(
                        [1, 2, 1], dtype="datetime64[ns]"
                    ),
                    "category_data": gd.Series(
                        ["a", "a", "b"], dtype="category"
                    ),
                }
            ),
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6219"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "include",
    [None, "all", ["object"], ["int"], ["object", "int", "category"]],
)
def test_describe_misc_include(df, include):
    pdf = df.to_pandas()

    expected = pdf.describe(include=include, datetime_is_numeric=True)
    actual = df.describe(include=include, datetime_is_numeric=True)

    for col in expected.columns:
        if expected[col].dtype == np.dtype("object"):
            expected[col] = expected[col].fillna(-1).astype("str")
            actual[col] = actual[col].fillna(-1).astype("str")

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "df",
    [
        gd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [10, 22, 33],
                "c": [0.3234, 0.23432, 0.0],
                "d": ["hello", "world", "hello"],
            }
        ),
        gd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["hello", "world", "hello"],
                "c": [0.3234, 0.23432, 0.0],
            }
        ),
        pytest.param(
            gd.DataFrame(
                {
                    "int_data": [1, 2, 3],
                    "str_data": ["hello", "world", "hello"],
                    "float_data": [0.3234, 0.23432, 0.0],
                    "timedelta_data": gd.Series(
                        [1, 2, 1], dtype="timedelta64[ns]"
                    ),
                    "datetime_data": gd.Series(
                        [1, 2, 1], dtype="datetime64[ns]"
                    ),
                }
            ),
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6219"
            ),
        ),
        pytest.param(
            gd.DataFrame(
                {
                    "int_data": [1, 2, 3],
                    "str_data": ["hello", "world", "hello"],
                    "float_data": [0.3234, 0.23432, 0.0],
                    "timedelta_data": gd.Series(
                        [1, 2, 1], dtype="timedelta64[ns]"
                    ),
                    "datetime_data": gd.Series(
                        [1, 2, 1], dtype="datetime64[ns]"
                    ),
                    "category_data": gd.Series(
                        ["a", "a", "b"], dtype="category"
                    ),
                }
            ),
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6219"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "exclude", [None, ["object"], ["int"], ["object", "int", "category"]]
)
def test_describe_misc_exclude(df, exclude):
    pdf = df.to_pandas()

    expected = pdf.describe(exclude=exclude, datetime_is_numeric=True)
    actual = df.describe(exclude=exclude, datetime_is_numeric=True)

    for col in expected.columns:
        if expected[col].dtype == np.dtype("object"):
            expected[col] = expected[col].fillna(-1).astype("str")
            actual[col] = actual[col].fillna(-1).astype("str")

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "df",
    [
        gd.DataFrame({"a": [1, 2, 3]}),
        gd.DataFrame(
            {"a": [1, 2, 3], "b": ["a", "z", "c"]}, index=["a", "z", "x"]
        ),
        gd.DataFrame(
            {
                "a": [1, 2, 3, None, 2, 1, None],
                "b": ["a", "z", "c", "a", "v", "z", "z"],
            }
        ),
        gd.DataFrame({"a": [], "b": []}),
        gd.DataFrame({"a": [None, None], "b": [None, None]}),
        gd.DataFrame(
            {
                "a": ["hello", "world", "rapids", "ai", "nvidia"],
                "b": gd.Series([1, 21, 21, 11, 11], dtype="timedelta64[s]"),
            }
        ),
        gd.DataFrame(
            {
                "a": ["hello", None, "world", "rapids", None, "ai", "nvidia"],
                "b": gd.Series(
                    [1, 21, None, 11, None, 11, None], dtype="datetime64[s]"
                ),
            }
        ),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
def test_dataframe_mode(df, numeric_only, dropna):
    pdf = df.to_pandas()

    expected = pdf.mode(numeric_only=numeric_only, dropna=dropna)
    actual = df.mode(numeric_only=numeric_only, dropna=dropna)

    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize("lhs, rhs", [("a", "a"), ("a", "b"), (1, 1.0)])
def test_equals_names(lhs, rhs):
    lhs = gd.DataFrame({lhs: [1, 2]})
    rhs = gd.DataFrame({rhs: [1, 2]})

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)


def test_equals_dtypes():
    lhs = gd.DataFrame({"a": [1, 2.0]})
    rhs = gd.DataFrame({"a": [1, 2]})

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)
