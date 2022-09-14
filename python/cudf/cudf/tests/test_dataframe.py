# Copyright (c) 2018-2022, NVIDIA CORPORATION.

import array as arr
import datetime
import io
import operator
import random
import re
import string
import textwrap
import warnings
from contextlib import contextmanager
from copy import copy

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numba import cuda
from packaging import version

import cudf
from cudf.core._compat import (
    PANDAS_GE_110,
    PANDAS_GE_120,
    PANDAS_GE_134,
    PANDAS_LT_140,
)
from cudf.core.column import column
from cudf.testing import _utils as utils
from cudf.testing._utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_eq,
    assert_exceptions_equal,
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
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("columns", [["a", "b"], pd.Series(["a", "b"])])
def test_init_via_list_of_series(columns):
    data = [pd.Series([1, 2]), pd.Series([3, 4])]

    pdf = cudf.DataFrame(data, columns=columns)
    gdf = cudf.DataFrame(data, columns=columns)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("index", [None, [0, 1, 2]])
def test_init_with_missing_columns(index):
    """Test initialization when columns and data keys are disjoint."""
    data = {"a": [1, 2, 3], "b": [2, 3, 4]}
    columns = ["c", "d"]

    pdf = cudf.DataFrame(data, columns=columns, index=index)
    gdf = cudf.DataFrame(data, columns=columns, index=index)

    assert_eq(pdf, gdf)


def _dataframe_na_data():
    return [
        pd.DataFrame(
            {
                "a": [0, 1, 2, np.nan, 4, None, 6],
                "b": [np.nan, None, "u", "h", "d", "a", "m"],
            },
            index=["q", "w", "e", "r", "t", "y", "u"],
        ),
        pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": ["a", "b", "u", "h", "d"]}),
        pd.DataFrame(
            {
                "a": [None, None, np.nan, None],
                "b": [np.nan, None, np.nan, None],
            }
        ),
        pd.DataFrame({"a": []}),
        pd.DataFrame({"a": [np.nan], "b": [None]}),
        pd.DataFrame({"a": ["a", "b", "c", None, "e"]}),
        pd.DataFrame({"a": ["a", "b", "c", "d", "e"]}),
    ]


@pytest.mark.parametrize("rows", [0, 1, 2, 100])
def test_init_via_list_of_empty_tuples(rows):
    data = [()] * rows

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(
        pdf,
        gdf,
        check_like=True,
        check_column_type=False,
        check_index_type=False,
    )


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
    gdf = cudf.DataFrame(dict_of_series)

    assert_eq(pdf, gdf)

    for key in dict_of_series:
        if isinstance(dict_of_series[key], pd.Series):
            dict_of_series[key] = cudf.Series(dict_of_series[key])

    gdf = cudf.DataFrame(dict_of_series)

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
        gdf = cudf.DataFrame(dict_of_series)

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
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": cudf.Series([1.0, 2.0, 3.0], index=[1, 2, 3]),
        },
        index=[7, 8, 9],
    )

    assert_eq(pdf, gdf, check_dtype=False)


def test_series_basic():
    # Make series from buffer
    a1 = np.arange(10, dtype=np.float64)
    series = cudf.Series(a1)
    assert len(series) == 10
    np.testing.assert_equal(series.to_numpy(), np.hstack([a1]))


def test_series_from_cupy_scalars():
    data = [0.1, 0.2, 0.3]
    data_np = np.array(data)
    data_cp = cupy.array(data)
    s_np = cudf.Series([data_np[0], data_np[2]])
    s_cp = cudf.Series([data_cp[0], data_cp[2]])
    assert_eq(s_np, s_cp)


@pytest.mark.parametrize("a", [[1, 2, 3], [1, 10, 30]])
@pytest.mark.parametrize("b", [[4, 5, 6], [-11, -100, 30]])
def test_append_index(a, b):

    df = pd.DataFrame()
    df["a"] = a
    df["b"] = b

    gdf = cudf.DataFrame()
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


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2]},
        {"a": [1, 2, 3], "b": [3, 4, 5]},
        {"a": [1, 2, 3, 4], "b": [3, 4, 5, 6], "c": [1, 3, 5, 7]},
        {"a": [np.nan, 2, 3, 4], "b": [3, 4, np.nan, 6], "c": [1, 3, 5, 7]},
        {1: [1, 2, 3], 2: [3, 4, 5]},
        {"a": [1, None, None], "b": [3, np.nan, np.nan]},
        {1: ["a", "b", "c"], 2: ["q", "w", "u"]},
        {1: ["a", np.nan, "c"], 2: ["q", None, "u"]},
        pytest.param(
            {},
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/11080"
            ),
        ),
        pytest.param(
            {1: [], 2: [], 3: []},
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/11080"
            ),
        ),
        pytest.param(
            [1, 2, 3],
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/11080"
            ),
        ),
    ],
)
def test_axes(data):
    csr = cudf.DataFrame(data)
    psr = pd.DataFrame(data)

    expected = psr.axes
    actual = csr.axes

    for e, a in zip(expected, actual):
        assert_eq(e, a)


def test_series_init_none():

    # test for creating empty series
    # 1: without initializing
    sr1 = cudf.Series()
    got = sr1.to_string()

    expect = repr(sr1.to_pandas())
    assert got == expect

    # 2: Using `None` as an initializer
    sr2 = cudf.Series(None)
    got = sr2.to_string()

    expect = repr(sr2.to_pandas())
    assert got == expect


def test_dataframe_basic():
    np.random.seed(0)
    df = cudf.DataFrame()

    # Populate with cuda memory
    df["keys"] = np.arange(10, dtype=np.float64)
    np.testing.assert_equal(df["keys"].to_numpy(), np.arange(10))
    assert len(df) == 10

    # Populate with numpy array
    rnd_vals = np.random.random(10)
    df["vals"] = rnd_vals
    np.testing.assert_equal(df["vals"].to_numpy(), rnd_vals)
    assert len(df) == 10
    assert tuple(df.columns) == ("keys", "vals")

    # Make another dataframe
    df2 = cudf.DataFrame()
    df2["keys"] = np.array([123], dtype=np.float64)
    df2["vals"] = np.array([321], dtype=np.float64)

    # Concat
    df = cudf.concat([df, df2])
    assert len(df) == 11

    hkeys = np.asarray(np.arange(10, dtype=np.float64).tolist() + [123])
    hvals = np.asarray(rnd_vals.tolist() + [321])

    np.testing.assert_equal(df["keys"].to_numpy(), hkeys)
    np.testing.assert_equal(df["vals"].to_numpy(), hvals)

    # As matrix
    mat = df.values_host

    expect = np.vstack([hkeys, hvals]).T

    np.testing.assert_equal(mat, expect)

    # test dataframe with tuple name
    df_tup = cudf.DataFrame()
    data = np.arange(10)
    df_tup[(1, "foobar")] = data
    np.testing.assert_equal(data, df_tup[(1, "foobar")].to_numpy())

    df = cudf.DataFrame(pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]}))
    pdf = pd.DataFrame(pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]}))
    assert_eq(df, pdf)

    gdf = cudf.DataFrame({"id": [0, 1], "val": [None, None]})
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
    "columns",
    [["a"], ["b"], "a", "b", ["a", "b"]],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_columns(pdf, columns, inplace):
    pdf = pdf.copy()
    gdf = cudf.from_pandas(pdf)

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
    gdf = cudf.from_pandas(pdf)

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
    gdf = cudf.from_pandas(pdf)

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
        ("cow", None),
        (
            "lama",
            None,
        ),
        (
            "falcon",
            None,
        ),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_multiindex(pdf, index, level, inplace):
    pdf = pdf.copy()
    gdf = cudf.from_pandas(pdf)

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
    "labels",
    [["a"], ["b"], "a", "b", ["a", "b"]],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_drop_labels_axis_1(pdf, labels, inplace):
    pdf = pdf.copy()
    gdf = cudf.from_pandas(pdf)

    expected = pdf.drop(labels=labels, axis=1, inplace=inplace)
    actual = gdf.drop(labels=labels, axis=1, inplace=inplace)

    if inplace:
        expected = pdf
        actual = gdf

    assert_eq(expected, actual)


def test_dataframe_drop_error():
    df = cudf.DataFrame({"a": [1], "b": [2], "c": [3]})
    pdf = df.to_pandas()

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"columns": "d"}),
        rfunc_args_and_kwargs=([], {"columns": "d"}),
        expected_error_message="column 'd' does not exist",
    )

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"columns": ["a", "d", "b"]}),
        rfunc_args_and_kwargs=([], {"columns": ["a", "d", "b"]}),
        expected_error_message="column 'd' does not exist",
    )

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=(["a"], {"columns": "a", "axis": 1}),
        rfunc_args_and_kwargs=(["a"], {"columns": "a", "axis": 1}),
        expected_error_message="Cannot specify both",
    )

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"axis": 1}),
        rfunc_args_and_kwargs=([], {"axis": 1}),
        expected_error_message="Need to specify at least",
    )

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([[2, 0]],),
        rfunc_args_and_kwargs=([[2, 0]],),
        expected_error_message="One or more values not found in axis",
    )


def test_dataframe_swaplevel_axis_0():
    midx = cudf.MultiIndex(
        levels=[
            ["Work"],
            ["Final exam", "Coursework"],
            ["History", "Geography"],
            ["January", "February", "March", "April"],
        ],
        codes=[[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, 3]],
        names=["a", "b", "c", "d"],
    )
    cdf = cudf.DataFrame(
        {
            "Grade": ["A", "B", "A", "C"],
            "Percentage": ["95", "85", "95", "75"],
        },
        index=midx,
    )
    pdf = cdf.to_pandas()

    assert_eq(pdf.swaplevel(), cdf.swaplevel())
    assert_eq(pdf.swaplevel(), cdf.swaplevel(-2, -1, 0))
    assert_eq(pdf.swaplevel(1, 2), cdf.swaplevel(1, 2))
    assert_eq(cdf.swaplevel(2, 1), cdf.swaplevel(1, 2))
    assert_eq(pdf.swaplevel(-1, -3), cdf.swaplevel(-1, -3))
    assert_eq(pdf.swaplevel("a", "b", 0), cdf.swaplevel("a", "b", 0))
    assert_eq(cdf.swaplevel("a", "b"), cdf.swaplevel("b", "a"))


def test_dataframe_swaplevel_TypeError():
    cdf = cudf.DataFrame(
        {"a": [1, 2, 3], "c": [10, 20, 30]}, index=["x", "y", "z"]
    )

    with pytest.raises(TypeError):
        cdf.swaplevel()


def test_dataframe_swaplevel_axis_1():
    midx = cudf.MultiIndex(
        levels=[
            ["b", "a"],
            ["bb", "aa"],
            ["bbb", "aaa"],
        ],
        codes=[[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1]],
        names=[None, "a", "b"],
    )
    cdf = cudf.DataFrame(
        data=[[45, 30, 100, 90], [200, 100, 50, 80]],
        columns=midx,
    )
    pdf = cdf.to_pandas()

    assert_eq(pdf.swaplevel(1, 2, 1), cdf.swaplevel(1, 2, 1))
    assert_eq(pdf.swaplevel("a", "b", 1), cdf.swaplevel("a", "b", 1))
    assert_eq(cdf.swaplevel(2, 1, 1), cdf.swaplevel(1, 2, 1))
    assert_eq(pdf.swaplevel(0, 2, 1), cdf.swaplevel(0, 2, 1))
    assert_eq(pdf.swaplevel(2, 0, 1), cdf.swaplevel(2, 0, 1))
    assert_eq(cdf.swaplevel("a", "a", 1), cdf.swaplevel("b", "b", 1))


def test_dataframe_drop_raises():
    df = cudf.DataFrame(
        {"a": [1, 2, 3], "c": [10, 20, 30]}, index=["x", "y", "z"]
    )
    pdf = df.to_pandas()
    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=(["p"],),
        rfunc_args_and_kwargs=(["p"],),
        expected_error_message="One or more values not found in axis",
    )

    # label dtype mismatch
    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([3],),
        rfunc_args_and_kwargs=([3],),
        expected_error_message="One or more values not found in axis",
    )

    expect = pdf.drop("p", errors="ignore")
    actual = df.drop("p", errors="ignore")

    assert_eq(actual, expect)

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"columns": "p"}),
        rfunc_args_and_kwargs=([], {"columns": "p"}),
        expected_error_message="column 'p' does not exist",
    )

    expect = pdf.drop(columns="p", errors="ignore")
    actual = df.drop(columns="p", errors="ignore")

    assert_eq(actual, expect)

    assert_exceptions_equal(
        lfunc=pdf.drop,
        rfunc=df.drop,
        lfunc_args_and_kwargs=([], {"labels": "p", "axis": 1}),
        rfunc_args_and_kwargs=([], {"labels": "p", "axis": 1}),
        expected_error_message="column 'p' does not exist",
    )

    expect = pdf.drop(labels="p", axis=1, errors="ignore")
    actual = df.drop(labels="p", axis=1, errors="ignore")

    assert_eq(actual, expect)


def test_dataframe_column_add_drop_via_setitem():
    df = cudf.DataFrame()
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
    df = cudf.DataFrame({"a": data_0, "b": data_1, "c": data_2})

    for i in range(10):
        df.c = df.a
        assert assert_eq(df.c, df.a, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")

        df.c = df.b
        assert assert_eq(df.c, df.b, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")


def test_dataframe_column_drop_via_attr():
    df = cudf.DataFrame({"a": []})

    with pytest.raises(AttributeError):
        del df.a

    assert tuple(df.columns) == tuple("a")


@pytest.mark.parametrize("axis", [0, "index"])
def test_dataframe_index_rename(axis):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf = cudf.DataFrame.from_pandas(pdf)

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
        gdf.rename(mapper={1: "x", 2: "y"}, axis=axis)


def test_dataframe_MI_rename():
    gdf = cudf.DataFrame(
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
    gdf = cudf.DataFrame.from_pandas(pdf)

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


def test_dataframe_pop():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [7.0, 8.0, 9.0]}
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

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
    empty_gdf = cudf.DataFrame(columns=["a", "b"])
    pb = empty_pdf.pop("b")
    gb = empty_gdf.pop("b")
    assert len(pb) == len(gb)
    assert empty_pdf.empty and empty_gdf.empty


@pytest.mark.parametrize("nelem", [0, 3, 100, 1000])
def test_dataframe_astype(nelem):
    df = cudf.DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df["a"].dtype is np.dtype(np.int32)
    df["b"] = df["a"].astype(np.float32)
    assert df["b"].dtype is np.dtype(np.float32)
    np.testing.assert_equal(df["a"].to_numpy(), df["b"].to_numpy())


def test_astype_dict():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": ["1", "2", "3"]})
    pdf = gdf.to_pandas()

    assert_eq(pdf.astype({"a": "str"}), gdf.astype({"a": "str"}))
    assert_eq(
        pdf.astype({"a": "str", "b": np.int64}),
        gdf.astype({"a": "str", "b": np.int64}),
    )


@pytest.mark.parametrize("nelem", [0, 100])
def test_index_astype(nelem):
    df = cudf.DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df.index.dtype is np.dtype(np.int64)
    df.index = df.index.astype(np.float32)
    assert df.index.dtype is np.dtype(np.float32)
    df["a"] = df["a"].astype(np.float32)
    np.testing.assert_equal(df.index.to_numpy(), df["a"].to_numpy())
    df["b"] = df["a"]
    df = df.set_index("b")
    df["a"] = df["a"].astype(np.int16)
    df.index = df.index.astype(np.int16)
    np.testing.assert_equal(df.index.to_numpy(), df["a"].to_numpy())


def test_dataframe_to_string_with_skipped_rows():
    # Test skipped rows
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]}
    )

    with pd.option_context("display.max_rows", 5):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
            a   b
        0   1  11
        1   2  12
        .. ..  ..
        4   5  15
        5   6  16

        [6 rows x 2 columns]"""
    )
    assert got == expect


def test_dataframe_to_string_with_skipped_rows_and_columns():
    # Test skipped rows and skipped columns
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [11, 12, 13, 14, 15, 16],
            "c": [11, 12, 13, 14, 15, 16],
            "d": [11, 12, 13, 14, 15, 16],
        }
    )

    with pd.option_context("display.max_rows", 5, "display.max_columns", 3):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
            a  ...   d
        0   1  ...  11
        1   2  ...  12
        .. ..  ...  ..
        4   5  ...  15
        5   6  ...  16

        [6 rows x 4 columns]"""
    )
    assert got == expect


def test_dataframe_to_string_with_masked_data():
    # Test masked data
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]}
    )

    data = np.arange(6)
    mask = np.zeros(1, dtype=cudf.utils.utils.mask_dtype)
    mask[0] = 0b00101101

    masked = cudf.Series.from_masked_array(data, mask)
    assert masked.null_count == 2
    df["c"] = masked

    # Check data
    values = masked.copy()
    validids = [0, 2, 3, 5]
    densearray = masked.dropna().to_numpy()
    np.testing.assert_equal(data[validids], densearray)
    # Valid position is correct
    for i in validids:
        assert data[i] == values[i]
    # Null position is correct
    for i in range(len(values)):
        if i not in validids:
            assert values[i] is cudf.NA

    with pd.option_context("display.max_rows", 10):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
           a   b     c
        0  1  11     0
        1  2  12  <NA>
        2  3  13     2
        3  4  14     3
        4  5  15  <NA>
        5  6  16     5"""
    )
    assert got == expect


def test_dataframe_to_string_wide(monkeypatch):
    monkeypatch.setenv("COLUMNS", "79")
    # Test basic
    df = cudf.DataFrame({f"a{i}": [0, 1, 2] for i in range(100)})
    with pd.option_context("display.max_columns", 0):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
           a0  a1  a2  a3  a4  a5  a6  a7  ...  a92  a93  a94  a95  a96  a97  a98  a99
        0   0   0   0   0   0   0   0   0  ...    0    0    0    0    0    0    0    0
        1   1   1   1   1   1   1   1   1  ...    1    1    1    1    1    1    1    1
        2   2   2   2   2   2   2   2   2  ...    2    2    2    2    2    2    2    2

        [3 rows x 100 columns]"""  # noqa: E501
    )
    assert got == expect


def test_dataframe_empty_to_string():
    # Test for printing empty dataframe
    df = cudf.DataFrame()
    got = df.to_string()

    expect = "Empty DataFrame\nColumns: []\nIndex: []"
    assert got == expect


def test_dataframe_emptycolumns_to_string():
    # Test for printing dataframe having empty columns
    df = cudf.DataFrame()
    df["a"] = []
    df["b"] = []
    got = df.to_string()

    expect = "Empty DataFrame\nColumns: [a, b]\nIndex: []"
    assert got == expect


def test_dataframe_copy():
    # Test for copying the dataframe using python copy pkg
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3]
    df2 = copy(df)
    df2["b"] = [4, 5, 6]
    got = df.to_string()

    expect = textwrap.dedent(
        """\
           a
        0  1
        1  2
        2  3"""
    )
    assert got == expect


def test_dataframe_copy_shallow():
    # Test for copy dataframe using class method
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3]
    df2 = df.copy()
    df2["b"] = [4, 2, 3]
    got = df.to_string()

    expect = textwrap.dedent(
        """\
           a
        0  1
        1  2
        2  3"""
    )
    assert got == expect


def test_dataframe_dtypes():
    dtypes = pd.Series(
        [np.int32, np.float32, np.float64], index=["c", "a", "b"]
    )
    df = cudf.DataFrame({k: np.ones(10, dtype=v) for k, v in dtypes.items()})
    assert df.dtypes.equals(dtypes)


def test_dataframe_add_col_to_object_dataframe():
    # Test for adding column to an empty object dataframe
    cols = ["a", "b", "c"]
    df = pd.DataFrame(columns=cols, dtype="str")

    data = {k: v for (k, v) in zip(cols, [["a"] for _ in cols])}

    gdf = cudf.DataFrame(data)
    gdf = gdf[:0]

    assert gdf.dtypes.equals(df.dtypes)
    gdf["a"] = [1]
    df["a"] = [10]
    assert gdf.dtypes.equals(df.dtypes)
    gdf["b"] = [1.0]
    df["b"] = [10.0]
    assert gdf.dtypes.equals(df.dtypes)


def test_dataframe_dir_and_getattr():
    df = cudf.DataFrame(
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


def test_empty_dataframe_to_cupy():
    df = cudf.DataFrame()

    # Check fully empty dataframe.
    mat = df.to_cupy()
    assert mat.shape == (0, 0)
    mat = df.to_numpy()
    assert mat.shape == (0, 0)

    df = cudf.DataFrame()
    nelem = 123
    for k in "abc":
        df[k] = np.random.random(nelem)

    # Check all columns in empty dataframe.
    mat = df.head(0).to_cupy()
    assert mat.shape == (0, 3)


def test_dataframe_to_cupy():
    df = cudf.DataFrame()

    nelem = 123
    for k in "abcd":
        df[k] = np.random.random(nelem)

    # Check all columns
    mat = df.to_cupy()
    assert mat.shape == (nelem, 4)
    assert mat.strides == (8, 984)

    mat = df.to_numpy()
    assert mat.shape == (nelem, 4)
    assert mat.strides == (8, 984)
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(df[k].to_numpy(), mat[:, i])

    # Check column subset
    mat = df[["a", "c"]].to_cupy().get()
    assert mat.shape == (nelem, 2)

    for i, k in enumerate("ac"):
        np.testing.assert_array_equal(df[k].to_numpy(), mat[:, i])


def test_dataframe_to_cupy_null_values():
    df = cudf.DataFrame()

    nelem = 123
    na = -10000

    refvalues = {}
    for k in "abcd":
        df[k] = data = np.random.random(nelem)
        bitmask = utils.random_bitmask(nelem)
        df[k] = df[k]._column.set_mask(bitmask)
        boolmask = np.asarray(
            utils.expand_bits_to_bytes(bitmask)[:nelem], dtype=np.bool_
        )
        data[~boolmask] = na
        refvalues[k] = data

    # Check null value causes error
    with pytest.raises(ValueError):
        df.to_cupy()
    with pytest.raises(ValueError):
        df.to_numpy()

    for k in df.columns:
        df[k] = df[k].fillna(na)

    mat = df.to_numpy()
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(refvalues[k], mat[:, i])


def test_dataframe_append_empty():
    pdf = pd.DataFrame(
        {
            "key": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

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

    test1_null = cudf.Series(ary, nan_as_null=True)
    assert test1_null.nullable
    assert test1_null.null_count == 20
    test1_nan = cudf.Series(ary, nan_as_null=False)
    assert test1_nan.null_count == 0

    test2_null = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ary}), nan_as_null=True
    )
    assert test2_null["a"].nullable
    assert test2_null["a"].null_count == 20
    test2_nan = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ary}), nan_as_null=False
    )
    assert test2_nan["a"].null_count == 0

    gpu_ary = cupy.asarray(ary)
    test3_null = cudf.Series(gpu_ary, nan_as_null=True)
    assert test3_null.nullable
    assert test3_null.null_count == 20
    test3_nan = cudf.Series(gpu_ary, nan_as_null=False)
    assert test3_nan.null_count == 0

    test4 = cudf.DataFrame()
    lst = [1, 2, None, 4, 5, 6, None, 8, 9]
    test4["lst"] = lst
    assert test4["lst"].nullable
    assert test4["lst"].null_count == 2


def test_dataframe_append_to_empty():
    pdf = pd.DataFrame()
    pdf["a"] = []
    pdf["b"] = [1, 2, 3]

    gdf = cudf.DataFrame()
    gdf["a"] = []
    gdf["b"] = [1, 2, 3]

    assert_eq(gdf, pdf)


def test_dataframe_setitem_index_len1():
    gdf = cudf.DataFrame()
    gdf["a"] = [1]
    gdf["b"] = gdf.index._values

    np.testing.assert_equal(gdf.b.to_numpy(), [0])


def test_empty_dataframe_setitem_df():
    gdf1 = cudf.DataFrame()
    gdf2 = cudf.DataFrame({"a": [1, 2, 3, 4, 5]})
    gdf1["a"] = gdf2["a"]
    assert_eq(gdf1, gdf2)


def test_assign():
    gdf = cudf.DataFrame({"x": [1, 2, 3]})
    gdf2 = gdf.assign(y=gdf.x + 1)
    assert list(gdf.columns) == ["x"]
    assert list(gdf2.columns) == ["x", "y"]

    np.testing.assert_equal(gdf2.y.to_numpy(), [2, 3, 4])


@pytest.mark.parametrize("nrows", [1, 8, 100, 1000])
@pytest.mark.parametrize("method", ["murmur3", "md5"])
def test_dataframe_hash_values(nrows, method):
    gdf = cudf.DataFrame()
    data = np.asarray(range(nrows))
    data[0] = data[-1]  # make first and last the same
    gdf["a"] = data
    gdf["b"] = gdf.a + 100
    out = gdf.hash_values()
    assert isinstance(out, cudf.Series)
    assert len(out) == nrows
    assert out.dtype == np.uint32

    # Check single column
    out_one = gdf[["a"]].hash_values(method=method)
    # First matches last
    assert out_one.iloc[0] == out_one.iloc[-1]
    # Equivalent to the cudf.Series.hash_values()
    assert_eq(gdf["a"].hash_values(method=method), out_one)


@pytest.mark.parametrize("nrows", [3, 10, 100, 1000])
@pytest.mark.parametrize("nparts", [1, 2, 8, 13])
@pytest.mark.parametrize("nkeys", [1, 2])
def test_dataframe_hash_partition(nrows, nparts, nkeys):
    np.random.seed(123)
    gdf = cudf.DataFrame()
    keycols = []
    for i in range(nkeys):
        keyname = f"key{i}"
        gdf[keyname] = np.random.randint(0, 7 - i, nrows)
        keycols.append(keyname)
    gdf["val1"] = np.random.randint(0, nrows * 2, nrows)

    got = gdf.partition_by_hash(keycols, nparts=nparts)
    # Must return a list
    assert isinstance(got, list)
    # Must have correct number of partitions
    assert len(got) == nparts
    # All partitions must be DataFrame type
    assert all(isinstance(p, cudf.DataFrame) for p in got)
    # Check that all partitions have unique keys
    part_unique_keys = set()
    for p in got:
        if len(p):
            # Take rows of the keycolumns and build a set of the key-values
            unique_keys = set(map(tuple, p[keycols].values_host))
            # Ensure that none of the key-values have occurred in other groups
            assert not (unique_keys & part_unique_keys)
            part_unique_keys |= unique_keys
    assert len(part_unique_keys)


@pytest.mark.parametrize("nrows", [3, 10, 50])
def test_dataframe_hash_partition_masked_value(nrows):
    gdf = cudf.DataFrame()
    gdf["key"] = np.arange(nrows)
    gdf["val"] = np.arange(nrows) + 100
    bitmask = utils.random_bitmask(nrows)
    bytemask = utils.expand_bits_to_bytes(bitmask)
    gdf["val"] = gdf["val"]._column.set_mask(bitmask)
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
    gdf = cudf.DataFrame()
    gdf["key"] = np.arange(nrows)
    gdf["val"] = np.arange(nrows) + 100
    bitmask = utils.random_bitmask(nrows)
    bytemask = utils.expand_bits_to_bytes(bitmask)
    gdf["key"] = gdf["key"]._column.set_mask(bitmask)
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

    gdf = cudf.DataFrame(
        {"val": [1, 2, 3, 4], "key": [3, 2, 1, 4]}, index=[4, 3, 2, 1]
    )

    expected_df1 = cudf.DataFrame(
        {"val": [1], "key": [3]}, index=[4] if keep_index else None
    )
    expected_df2 = cudf.DataFrame(
        {"val": [2, 3, 4], "key": [2, 1, 4]},
        index=[3, 2, 1] if keep_index else range(1, 4),
    )
    expected = [expected_df1, expected_df2]

    parts = gdf.partition_by_hash(["key"], nparts=2, keep_index=keep_index)

    for exp, got in zip(expected, parts):
        assert_eq(exp, got)


def test_dataframe_hash_partition_empty():
    gdf = cudf.DataFrame({"val": [1, 2], "key": [3, 2]}, index=["a", "b"])
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
        with pytest.raises(TypeError):
            cudf.concat([df1, df2])
    else:
        pres = pd.concat([df1, df2])
        gres = cudf.concat([cudf.from_pandas(df1), cudf.from_pandas(df2)])
        assert_eq(pres, gres, check_dtype=False, check_index_type=True)


def test_dataframe_concat_different_column_types():
    df1 = cudf.Series([42], dtype=np.float64)
    df2 = cudf.Series(["a"], dtype="category")
    with pytest.raises(ValueError):
        cudf.concat([df1, df2])

    df2 = cudf.Series(["a string"])
    with pytest.raises(TypeError):
        cudf.concat([df1, df2])


@pytest.mark.parametrize(
    "df_1", [cudf.DataFrame({"a": [1, 2], "b": [1, 3]}), cudf.DataFrame({})]
)
@pytest.mark.parametrize(
    "df_2", [cudf.DataFrame({"a": [], "b": []}), cudf.DataFrame({})]
)
def test_concat_empty_dataframe(df_1, df_2):

    got = cudf.concat([df_1, df_2])
    expect = pd.concat([df_1.to_pandas(), df_2.to_pandas()], sort=False)

    # ignoring dtypes as pandas upcasts int to float
    # on concatenation with empty dataframes

    assert_eq(got, expect, check_dtype=False, check_index_type=True)


@pytest.mark.parametrize(
    "df1_d",
    [
        {"a": [1, 2], "b": [1, 2], "c": ["s1", "s2"], "d": [1.0, 2.0]},
        {"b": [1.9, 10.9], "c": ["s1", "s2"]},
        {"c": ["s1"], "b": pd.Series([None], dtype="float"), "a": [False]},
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
    got = cudf.concat(
        [cudf.DataFrame(df1_d), cudf.DataFrame(df2_d), cudf.DataFrame(df1_d)],
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

    assert_eq(got, expect, check_dtype=False, check_index_type=True)


@pytest.mark.parametrize(
    "ser_1", [pd.Series([1, 2, 3]), pd.Series([], dtype="float64")]
)
@pytest.mark.parametrize("ser_2", [pd.Series([], dtype="float64")])
def test_concat_empty_series(ser_1, ser_2):
    got = cudf.concat([cudf.Series(ser_1), cudf.Series(ser_2)])
    expect = pd.concat([ser_1, ser_2])

    assert_eq(got, expect, check_index_type=True)


def test_concat_with_axis():
    df1 = pd.DataFrame(dict(x=np.arange(5), y=np.arange(5)))
    df2 = pd.DataFrame(dict(a=np.arange(5), b=np.arange(5)))

    concat_df = pd.concat([df1, df2], axis=1)
    cdf1 = cudf.from_pandas(df1)
    cdf2 = cudf.from_pandas(df2)

    # concat only dataframes
    concat_cdf = cudf.concat([cdf1, cdf2], axis=1)
    assert_eq(concat_cdf, concat_df, check_index_type=True)

    # concat only series
    concat_s = pd.concat([df1.x, df1.y], axis=1)
    cs1 = cudf.Series.from_pandas(df1.x)
    cs2 = cudf.Series.from_pandas(df1.y)
    concat_cdf_s = cudf.concat([cs1, cs2], axis=1)

    assert_eq(concat_cdf_s, concat_s, check_index_type=True)

    # concat series and dataframes
    s3 = pd.Series(np.random.random(5))
    cs3 = cudf.Series.from_pandas(s3)

    concat_cdf_all = cudf.concat([cdf1, cs3, cdf2], axis=1)
    concat_df_all = pd.concat([df1, s3, df2], axis=1)
    assert_eq(concat_cdf_all, concat_df_all, check_index_type=True)

    # concat manual multi index
    midf1 = cudf.from_pandas(df1)
    midf1.index = cudf.MultiIndex(
        levels=[[0, 1, 2, 3], [0, 1]], codes=[[0, 1, 2, 3, 2], [0, 1, 0, 1, 0]]
    )
    midf2 = midf1[2:]
    midf2.index = cudf.MultiIndex(
        levels=[[3, 4, 5], [2, 0]], codes=[[0, 1, 2], [1, 0, 1]]
    )
    mipdf1 = midf1.to_pandas()
    mipdf2 = midf2.to_pandas()

    assert_eq(
        cudf.concat([midf1, midf2]),
        pd.concat([mipdf1, mipdf2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([midf2, midf1]),
        pd.concat([mipdf2, mipdf1]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([midf1, midf2, midf1]),
        pd.concat([mipdf1, mipdf2, mipdf1]),
        check_index_type=True,
    )

    # concat groupby multi index
    gdf1 = cudf.DataFrame(
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

    assert_eq(
        cudf.concat([gdg1, gdg2]),
        pd.concat([pdg1, pdg2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([gdg2, gdg1]),
        pd.concat([pdg2, pdg1]),
        check_index_type=True,
    )

    # series multi index concat
    gdgz1 = gdg1.z
    gdgz2 = gdg2.z
    pdgz1 = gdgz1.to_pandas()
    pdgz2 = gdgz2.to_pandas()

    assert_eq(
        cudf.concat([gdgz1, gdgz2]),
        pd.concat([pdgz1, pdgz2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([gdgz2, gdgz1]),
        pd.concat([pdgz2, pdgz1]),
        check_index_type=True,
    )


@pytest.mark.parametrize("nrows", [0, 3, 10, 100, 1000])
def test_nonmatching_index_setitem(nrows):
    np.random.seed(0)

    gdf = cudf.DataFrame()
    gdf["a"] = np.random.randint(2147483647, size=nrows)
    gdf["b"] = np.random.randint(2147483647, size=nrows)
    gdf = gdf.set_index("b")

    test_values = np.random.randint(2147483647, size=nrows)
    gdf["c"] = test_values
    assert len(test_values) == len(gdf["c"])
    gdf_series = cudf.Series(test_values, index=gdf.index, name="c")
    assert_eq(gdf["c"].to_pandas(), gdf_series.to_pandas())


def test_from_pandas():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])
    gdf = cudf.DataFrame.from_pandas(df)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    s = df.x
    gs = cudf.Series.from_pandas(s)
    assert isinstance(gs, cudf.Series)

    assert_eq(s, gs)


@pytest.mark.parametrize("dtypes", [int, float])
def test_from_records(dtypes):
    h_ary = np.ndarray(shape=(10, 4), dtype=dtypes)
    rec_ary = h_ary.view(np.recarray)

    gdf = cudf.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    df = pd.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(df, gdf)

    gdf = cudf.DataFrame.from_records(rec_ary)
    df = pd.DataFrame.from_records(rec_ary)
    assert isinstance(gdf, cudf.DataFrame)
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
    gdf = cudf.DataFrame.from_records(rec_ary, columns=columns, index=index)
    df = pd.DataFrame.from_records(rec_ary, columns=columns, index=index)
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(df, gdf)


def test_dataframe_construction_from_cupy_arrays():
    h_ary = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    d_ary = cupy.asarray(h_ary)

    gdf = cudf.DataFrame(d_ary, columns=["a", "b", "c"])
    df = pd.DataFrame(h_ary, columns=["a", "b", "c"])
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    df = pd.DataFrame(h_ary)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary, index=["a", "b"])
    df = pd.DataFrame(h_ary, index=["a", "b"])
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    gdf = gdf.set_index(keys=0, drop=False)
    df = pd.DataFrame(h_ary)
    df = df.set_index(keys=0, drop=False)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    gdf = gdf.set_index(keys=1, drop=False)
    df = pd.DataFrame(h_ary)
    df = df.set_index(keys=1, drop=False)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)


def test_dataframe_cupy_wrong_dimensions():
    d_ary = cupy.empty((2, 3, 4), dtype=np.int32)
    with pytest.raises(
        ValueError, match="records dimension expected 1 or 2 but found: 3"
    ):
        cudf.DataFrame(d_ary)


def test_dataframe_cupy_array_wrong_index():
    d_ary = cupy.empty((2, 3), dtype=np.int32)

    with pytest.raises(ValueError):
        cudf.DataFrame(d_ary, index=["a"])

    with pytest.raises(ValueError):
        cudf.DataFrame(d_ary, index="a")


def test_index_in_dataframe_constructor():
    a = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])
    b = cudf.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])

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
    gdf = cudf.DataFrame.from_arrow(padf)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    s = pa.Array.from_pandas(df.a)
    gs = cudf.Series.from_arrow(s)
    assert isinstance(gs, cudf.Series)

    # For some reason PyArrow to_pandas() converts to numpy array and has
    # better type compatibility
    np.testing.assert_array_equal(s.to_pandas(), gs.to_numpy())


@pytest.mark.parametrize("nelem", [0, 2, 3, 100, 1000])
@pytest.mark.parametrize("data_type", dtypes)
def test_to_arrow(nelem, data_type):
    df = pd.DataFrame(
        {
            "a": np.random.randint(0, 1000, nelem).astype(data_type),
            "b": np.random.randint(0, 1000, nelem).astype(data_type),
        }
    )
    gdf = cudf.DataFrame.from_pandas(df)

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
    gs1 = cudf.Series.from_arrow(s1)
    assert isinstance(gs1, cudf.Series)
    # We have 64B padded buffers for nulls whereas Arrow returns a minimal
    # number of bytes, so only check the first byte in this case
    np.testing.assert_array_equal(
        np.asarray(s1.buffers()[0]).view("u1")[0],
        gs1._column.mask_array_view.copy_to_host().view("u1")[0],
    )
    assert pa.Array.equals(s1, gs1.to_arrow())

    s2 = pa.array([None, None, None, None, None], type=data_type)
    gs2 = cudf.Series.from_arrow(s2)
    assert isinstance(gs2, cudf.Series)
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
    gdf = cudf.DataFrame.from_pandas(df)

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
    gd_cat = cudf.Series(pa_cat)

    assert isinstance(gd_cat, cudf.Series)
    assert_eq(
        pd.Series(pa_cat.to_pandas()),  # PyArrow returns a pd.Categorical
        gd_cat.to_pandas(),
    )


def test_to_arrow_missing_categorical():
    pd_cat = pd.Categorical(["a", "b", "c"], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = cudf.Series(pa_cat)

    assert isinstance(gd_cat, cudf.Series)
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
        scalar = np.datetime64(datetime.date.today()).astype("datetime64[ms]")
        data_type = "datetime64[ms]"
    else:
        scalar = np.dtype(data_type).type(np.random.randint(0, 5))

    gdf = cudf.DataFrame()
    gdf["a"] = [1, 2, 3, 4, 5]
    gdf["b"] = scalar
    assert gdf["b"].dtype == np.dtype(data_type)
    assert len(gdf["b"]) == len(gdf["a"])


@pytest.mark.parametrize("data_type", NUMERIC_TYPES)
def test_from_python_array(data_type):
    np_arr = np.random.randint(0, 100, 10).astype(data_type)
    data = memoryview(np_arr)
    data = arr.array(data.format, data)

    gs = cudf.Series(data)

    np.testing.assert_equal(gs.to_numpy(), np_arr)


def test_series_shape():
    ps = pd.Series([1, 2, 3, 4])
    cs = cudf.Series([1, 2, 3, 4])

    assert ps.shape == cs.shape


def test_series_shape_empty():
    ps = pd.Series(dtype="float64")
    cs = cudf.Series([])

    assert ps.shape == cs.shape


def test_dataframe_shape():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    gdf = cudf.DataFrame.from_pandas(pdf)

    assert pdf.shape == gdf.shape


def test_dataframe_shape_empty():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    assert pdf.shape == gdf.shape


@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 20])
@pytest.mark.parametrize("dtype", dtypes + ["object"])
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_dataframe_transpose(nulls, num_cols, num_rows, dtype):
    # In case of `bool` dtype: pandas <= 1.2.5 type-casts
    # a boolean series to `float64` series if a `np.nan` is assigned to it:
    # >>> s = pd.Series([True, False, True])
    # >>> s
    # 0     True
    # 1    False
    # 2     True
    # dtype: bool
    # >>> s[[2]] = np.nan
    # >>> s
    # 0    1.0
    # 1    0.0
    # 2    NaN
    # dtype: float64
    # In pandas >= 1.3.2 this behavior is fixed:
    # >>> s = pd.Series([True, False, True])
    # >>> s
    # 0
    # True
    # 1
    # False
    # 2
    # True
    # dtype: bool
    # >>> s[[2]] = np.nan
    # >>> s
    # 0
    # True
    # 1
    # False
    # 2
    # NaN
    # dtype: object
    # In cudf we change `object` dtype to `str` type - for which there
    # is no transpose implemented yet. Hence we need to test transpose
    # against pandas nullable types as they are the ones that closely
    # resemble `cudf` dtypes behavior.
    pdf = pd.DataFrame()

    null_rep = np.nan if dtype in ["float32", "float64"] else None
    np_dtype = dtype
    dtype = np.dtype(dtype)
    dtype = cudf.utils.dtypes.np_dtypes_to_pandas_dtypes.get(dtype, dtype)
    for i in range(num_cols):
        colname = string.ascii_lowercase[i]
        data = pd.Series(
            np.random.randint(0, 26, num_rows).astype(np_dtype),
            dtype=dtype,
        )
        if nulls == "some":
            idx = np.random.choice(
                num_rows, size=int(num_rows / 2), replace=False
            )
            if len(idx):
                data[idx] = null_rep
        elif nulls == "all":
            data[:] = null_rep
        pdf[colname] = data

    gdf = cudf.DataFrame.from_pandas(pdf)

    got_function = gdf.transpose()
    got_property = gdf.T

    expect = pdf.transpose()

    assert_eq(expect, got_function.to_pandas(nullable=True))
    assert_eq(expect, got_property.to_pandas(nullable=True))


@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 20])
def test_dataframe_transpose_category(num_cols, num_rows):
    pdf = pd.DataFrame()

    for i in range(num_cols):
        colname = string.ascii_lowercase[i]
        data = pd.Series(list(string.ascii_lowercase), dtype="category")
        data = data.sample(num_rows, replace=True).reset_index(drop=True)
        pdf[colname] = data

    gdf = cudf.DataFrame.from_pandas(pdf)

    got_function = gdf.transpose()
    got_property = gdf.T

    expect = pdf.transpose()

    assert_eq(expect, got_function.to_pandas())
    assert_eq(expect, got_property.to_pandas())


def test_generated_column():
    gdf = cudf.DataFrame({"a": (i for i in range(5))})
    assert len(gdf) == 5


@pytest.fixture
def pdf():
    return pd.DataFrame({"x": range(10), "y": range(10)})


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


@pytest.mark.parametrize(
    "data",
    [
        {
            "x": [np.nan, 2, 3, 4, 100, np.nan],
            "y": [4, 5, 6, 88, 99, np.nan],
            "z": [7, 8, 9, 66, np.nan, 77],
        },
        {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]},
        {
            "x": [np.nan, np.nan, np.nan],
            "y": [np.nan, np.nan, np.nan],
            "z": [np.nan, np.nan, np.nan],
        },
        pytest.param(
            {"x": [], "y": [], "z": []},
            marks=pytest.mark.xfail(
                condition=version.parse("11")
                <= version.parse(cupy.__version__)
                < version.parse("11.1"),
                reason="Zero-sized array passed to cupy reduction, "
                "https://github.com/cupy/cupy/issues/6937",
            ),
        ),
        pytest.param(
            {"x": []},
            marks=pytest.mark.xfail(
                condition=version.parse("11")
                <= version.parse(cupy.__version__)
                < version.parse("11.1"),
                reason="Zero-sized array passed to cupy reduction, "
                "https://github.com/cupy/cupy/issues/6937",
            ),
        ),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "func",
    [
        "min",
        "max",
        "sum",
        "prod",
        "product",
        "cummin",
        "cummax",
        "cumsum",
        "cumprod",
        "mean",
        "median",
        "sum",
        "max",
        "std",
        "var",
        "kurt",
        "skew",
        "all",
        "any",
    ],
)
@pytest.mark.parametrize("skipna", [True, False])
def test_dataframe_reductions(data, axis, func, skipna):
    pdf = pd.DataFrame(data=data)
    gdf = cudf.DataFrame.from_pandas(pdf)

    # Reductions can fail in numerous possible ways when attempting row-wise
    # reductions, which are only partially supported. Catching the appropriate
    # exception here allows us to detect API breakage in the form of changing
    # exceptions.
    expected_exception = None
    if axis == 1:
        if func in ("kurt", "skew"):
            expected_exception = NotImplementedError
        elif func not in cudf.core.dataframe._cupy_nan_methods_map:
            if skipna is False:
                expected_exception = NotImplementedError
            elif any(col.nullable for name, col in gdf.items()):
                expected_exception = ValueError
            elif func in ("cummin", "cummax"):
                expected_exception = AttributeError

    # Test different degrees of freedom for var and std.
    all_kwargs = [{"ddof": 1}, {"ddof": 2}] if func in ("var", "std") else [{}]
    for kwargs in all_kwargs:
        if expected_exception is not None:
            with pytest.raises(expected_exception):
                getattr(gdf, func)(axis=axis, skipna=skipna, **kwargs),
        else:
            assert_eq(
                getattr(pdf, func)(axis=axis, skipna=skipna, **kwargs),
                getattr(gdf, func)(axis=axis, skipna=skipna, **kwargs),
                check_dtype=False,
            )


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
    gdf = cudf.DataFrame.from_pandas(pdf)

    assert_eq(func(pdf), func(gdf))


@pytest.mark.parametrize(
    "data",
    [
        {"x": [np.nan, 2, 3, 4, 100, np.nan], "y": [4, 5, 6, 88, 99, np.nan]},
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan]},
        {"x": pd.Series([], dtype="float"), "y": pd.Series([], dtype="float")},
        {"x": pd.Series([], dtype="int")},
    ],
)
@pytest.mark.parametrize("ops", ["sum", "product", "prod"])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [-10, -1, 0, 1, 2, 3, 10])
def test_dataframe_min_count_ops(data, ops, skipna, min_count):
    psr = pd.DataFrame(data)
    gsr = cudf.from_pandas(psr)

    assert_eq(
        getattr(psr, ops)(skipna=skipna, min_count=min_count),
        getattr(gsr, ops)(skipna=skipna, min_count=min_count),
        check_dtype=False,
    )


@contextmanager
def _hide_host_other_warning(other):
    if isinstance(other, (dict, list)):
        with pytest.warns(FutureWarning):
            yield
    else:
        yield


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
@pytest.mark.parametrize(
    "other",
    [
        1.0,
        [1.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0],
        {"x": 1.0},
        {"x": 1.0, "y": 2.0},
        {"x": 1.0, "y": 2.0, "z": 3.0},
        {"x": 1.0, "z": 3.0},
        pd.Series([1.0]),
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0], index=["x"]),
        pd.Series([1.0, 2.0], index=["x", "y"]),
        pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}),
    ],
)
def test_binops_df(pdf, gdf, binop, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        with warnings.catch_warnings(record=True) as w:
            d = binop(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)

        # TODO: When we remove support for binary operations with lists and
        # dicts, those cases should all be checked in a `pytest.raises` block
        # that returns before we enter this try-except.
        with _hide_host_other_warning(other):
            assert_exceptions_equal(
                lfunc=binop,
                rfunc=binop,
                lfunc_args_and_kwargs=([pdf, other], {}),
                rfunc_args_and_kwargs=([gdf, other], {}),
                compare_error_message=False,
            )
    else:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        with _hide_host_other_warning(other):
            g = binop(gdf, other)
        try:
            assert_eq(d, g)
        except AssertionError:
            # Currently we will not match pandas for equality/inequality
            # operators when there are columns that exist in a Series but not
            # the DataFrame because pandas returns True/False values whereas we
            # return NA. However, this reindexing is deprecated in pandas so we
            # opt not to add support.
            if w and "DataFrame vs Series comparisons is deprecated" in str(w):
                pass


def test_binops_df_invalid(gdf):
    with pytest.raises(TypeError):
        gdf + np.array([1, 2])


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


def test_df_abs(pdf):
    np.random.seed(0)
    disturbance = pd.Series(np.random.rand(10))
    pdf = pdf - 5 + disturbance
    d = pdf.apply(np.abs)
    g = cudf.from_pandas(pdf).abs()
    assert_eq(d, g)


def test_scale_df(gdf):
    got = (gdf - 5).scale()
    expect = cudf.DataFrame(
        {"x": np.linspace(0.0, 1.0, 10), "y": np.linspace(0.0, 1.0, 10)}
    )
    assert_eq(expect, got)


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
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert not gdf.index.is_monotonic
    assert not gdf.index.is_monotonic_increasing
    assert not gdf.index.is_monotonic_decreasing


def test_iter(pdf, gdf):
    assert list(pdf) == list(gdf)


def test_iteritems(gdf):
    for k, v in gdf.items():
        assert k in gdf.columns
        assert isinstance(v, cudf.Series)
        assert_eq(v, gdf[k])


@pytest.mark.parametrize("q", [0.5, 1, 0.001, [0.5], [], [0.005, 0.5, 1]])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_quantile(q, numeric_only):
    ts = pd.date_range("2018-08-24", periods=5, freq="D")
    td = pd.to_timedelta(np.arange(5), unit="h")
    pdf = pd.DataFrame(
        {"date": ts, "delta": td, "val": np.random.randn(len(ts))}
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    assert_eq(pdf["date"].quantile(q), gdf["date"].quantile(q))
    assert_eq(pdf["delta"].quantile(q), gdf["delta"].quantile(q))
    assert_eq(pdf["val"].quantile(q), gdf["val"].quantile(q))

    if numeric_only:
        assert_eq(pdf.quantile(q), gdf.quantile(q))
    else:
        q = q if isinstance(q, list) else [q]
        assert_eq(
            pdf.quantile(
                q if isinstance(q, list) else [q], numeric_only=False
            ),
            gdf.quantile(q, numeric_only=False),
        )


@pytest.mark.parametrize("q", [0.2, 1, 0.001, [0.5], [], [0.005, 0.8, 0.03]])
@pytest.mark.parametrize("interpolation", ["higher", "lower", "nearest"])
@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
)
def test_decimal_quantile(q, interpolation, decimal_type):
    data = ["244.8", "32.24", "2.22", "98.14", "453.23", "5.45"]
    gdf = cudf.DataFrame(
        {"id": np.random.randint(0, 10, size=len(data)), "val": data}
    )
    gdf["id"] = gdf["id"].astype("float64")
    gdf["val"] = gdf["val"].astype(decimal_type(7, 2))
    pdf = gdf.to_pandas()

    got = gdf.quantile(q, numeric_only=False, interpolation=interpolation)
    expected = pdf.quantile(
        q if isinstance(q, list) else [q],
        numeric_only=False,
        interpolation=interpolation,
    )

    assert_eq(got, expected)


def test_empty_quantile():
    pdf = pd.DataFrame({"x": []})
    df = cudf.DataFrame({"x": []})

    actual = df.quantile()
    expected = pdf.quantile()

    assert_eq(actual, expected)


def test_from_pandas_function(pdf):
    gdf = cudf.from_pandas(pdf)
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(pdf, gdf)

    gdf = cudf.from_pandas(pdf.x)
    assert isinstance(gdf, cudf.Series)
    assert_eq(pdf.x, gdf)

    with pytest.raises(TypeError):
        cudf.from_pandas(123)


@pytest.mark.parametrize("preserve_index", [True, False])
def test_arrow_pandas_compat(pdf, gdf, preserve_index):
    pdf["z"] = range(10)
    pdf = pdf.set_index("z")
    gdf["z"] = range(10)
    gdf = gdf.set_index("z")

    pdf_arrow_table = pa.Table.from_pandas(pdf, preserve_index=preserve_index)
    gdf_arrow_table = gdf.to_arrow(preserve_index=preserve_index)

    assert pa.Table.equals(pdf_arrow_table, gdf_arrow_table)

    gdf2 = cudf.DataFrame.from_arrow(pdf_arrow_table)
    pdf2 = pdf_arrow_table.to_pandas()

    assert_eq(pdf2, gdf2)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["bool"])
def test_cuda_array_interface(dtype):

    np_data = np.arange(10).astype(dtype)
    cupy_data = cupy.array(np_data)
    pd_data = pd.Series(np_data)

    cudf_data = cudf.Series(cupy_data)
    assert_eq(pd_data, cudf_data)

    gdf = cudf.DataFrame()
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
    got = cudf.Series(pa_chunk_array)

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
    got = cudf.DataFrame.from_arrow(pa_table)

    assert_eq(expect, got)


@pytest.mark.skip(reason="Test was designed to be run in isolation")
def test_gpu_memory_usage_with_boolmask():
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
    boolmask = cudf.Series(np.random.randint(1, 2, len(cudaDF)).astype("bool"))

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
    gdf = cudf.DataFrame.from_pandas(pdf)
    gdf_mask = cudf.DataFrame.from_pandas(pdf_mask)
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
            cudf.Series([True, False, True]),
            marks=pytest.mark.xfail(
                reason="Pandas can't index a multiindex with a Series"
            ),
        ),
    ],
)
def test_dataframe_multiindex_boolmask(mask):
    gdf = cudf.DataFrame(
        {"w": [3, 2, 1], "x": [1, 2, 3], "y": [0, 1, 0], "z": [1, 1, 1]}
    )
    gdg = gdf.groupby(["w", "x"]).count()
    pdg = gdg.to_pandas()
    assert_eq(gdg[mask], pdg[mask])


def test_dataframe_assignment():
    pdf = pd.DataFrame()
    for col in "abc":
        pdf[col] = np.array([0, 1, 1, -2, 10])
    gdf = cudf.DataFrame.from_pandas(pdf)
    gdf[gdf < 0] = 999
    pdf[pdf < 0] = 999
    assert_eq(gdf, pdf)


def test_1row_arrow_table():
    data = [pa.array([0]), pa.array([1])]
    batch = pa.RecordBatch.from_arrays(data, ["f0", "f1"])
    table = pa.Table.from_batches([batch])

    expect = table.to_pandas()
    got = cudf.DataFrame.from_arrow(table)
    assert_eq(expect, got)


def test_arrow_handle_no_index_name(pdf, gdf):
    gdf_arrow = gdf.to_arrow()
    pdf_arrow = pa.Table.from_pandas(pdf)
    assert pa.Table.equals(pdf_arrow, gdf_arrow)

    got = cudf.DataFrame.from_arrow(gdf_arrow)
    expect = pdf_arrow.to_pandas()
    assert_eq(expect, got)


def test_pandas_non_contiguious():
    arr1 = np.random.sample([5000, 10])
    assert arr1.flags["C_CONTIGUOUS"] is True
    df = pd.DataFrame(arr1)
    for col in df.columns:
        assert df[col].values.flags["C_CONTIGUOUS"] is False

    gdf = cudf.DataFrame.from_pandas(df)
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
    expect = pd.Series(data, dtype="float64")
    got = cudf.Series(data, dtype="float64")

    assert_eq(expect, got)


@pytest.mark.parametrize("num_elements", [0, 2, 10, 100])
def test_series_all_valid_nan(num_elements):
    data = [np.nan] * num_elements
    sr = cudf.Series(data, nan_as_null=False)
    np.testing.assert_equal(sr.null_count, 0)


def test_series_rename():
    pds = pd.Series([1, 2, 3], name="asdf")
    gds = cudf.Series([1, 2, 3], name="asdf")

    expect = pds.rename("new_name")
    got = gds.rename("new_name")

    assert_eq(expect, got)

    pds = pd.Series(expect)
    gds = cudf.Series(got)

    assert_eq(pds, gds)

    pds = pd.Series(expect, name="name name")
    gds = cudf.Series(got, name="name name")

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

    gdf = cudf.DataFrame(
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
    gdf = cudf.DataFrame()
    gdf["id"] = cudf.Series(["a", "b"], dtype=np.object_)
    gdf["v"] = cudf.Series([1, 2])
    assert_eq(gdf.tail(3), gdf.to_pandas().tail(3))


@pytest.mark.parametrize("level", [None, 0, "l0", 1, ["l0", 1]])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize(
    "column_names",
    [
        ["v0", "v1"],
        ["v0", "index"],
        pd.MultiIndex.from_tuples([("x0", "x1"), ("y0", "y1")]),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index(level, drop, column_names, inplace, col_level, col_fill):
    midx = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["l0", None]
    )
    pdf = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8]], index=midx, columns=column_names
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    got = gdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    if inplace:
        expect = pdf
        got = gdf

    assert_eq(expect, got)


@pytest.mark.parametrize("level", [None, 0, 1, [None]])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_dup_level_name(level, drop, inplace, col_level, col_fill):
    # midx levels are named [None, None]
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    pdf = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=midx)
    gdf = cudf.from_pandas(pdf)
    if level == [None]:
        assert_exceptions_equal(
            lfunc=pdf.reset_index,
            rfunc=gdf.reset_index,
            lfunc_args_and_kwargs=(
                [],
                {"level": level, "drop": drop, "inplace": inplace},
            ),
            rfunc_args_and_kwargs=(
                [],
                {"level": level, "drop": drop, "inplace": inplace},
            ),
            expected_error_message="occurs multiple times, use a level number",
        )
        return

    expect = pdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    got = gdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    if inplace:
        expect = pdf
        got = gdf

    assert_eq(expect, got)


@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_named(pdf, gdf, drop, inplace, col_level, col_fill):
    pdf.index.name = "cudf"
    gdf.index.name = "cudf"

    expect = pdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    got = gdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    if inplace:
        expect = pdf
        got = gdf
    assert_eq(expect, got)


@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("column_names", [["x", "y"], ["index", "y"]])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_unnamed(
    pdf, gdf, drop, inplace, column_names, col_level, col_fill
):
    pdf.columns = column_names
    gdf.columns = column_names

    expect = pdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    got = gdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    if inplace:
        expect = pdf
        got = gdf
    assert_eq(expect, got)


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
    gdf = cudf.DataFrame(data)
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
    gdf = cudf.DataFrame(data)
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
    gdf = cudf.DataFrame.from_pandas(df)

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


@pytest.fixture()
def reindex_data():
    return cudf.datasets.randomdata(
        nrows=6,
        dtypes={
            "a": "category",
            "c": float,
            "d": str,
        },
    )


@pytest.fixture()
def reindex_data_numeric():
    return cudf.datasets.randomdata(
        nrows=6,
        dtypes={"a": float, "b": float, "c": float},
    )


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize(
    "args,gd_kwargs",
    [
        ([], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {"axis": 0}),
        ([["a", "b", "c", "d", "e"]], {"axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": 0}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": "index"}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": "columns"}),
        ([], {"index": [-3, 0, 3, 0, -2, 1, 3, 4, 6]}),
        ([], {"columns": ["a", "b", "c", "d", "e"]}),
        (
            [],
            {
                "index": [-3, 0, 3, 0, -2, 1, 3, 4, 6],
                "columns": ["a", "b", "c", "d", "e"],
            },
        ),
    ],
)
def test_dataframe_reindex(copy, reindex_data, args, gd_kwargs):
    pdf, gdf = reindex_data.to_pandas(), reindex_data

    gd_kwargs["copy"] = copy
    pd_kwargs = gd_kwargs.copy()
    pd_kwargs["copy"] = True
    assert_eq(pdf.reindex(*args, **pd_kwargs), gdf.reindex(*args, **gd_kwargs))


@pytest.mark.parametrize("fill_value", [-1.0, 0.0, 1.5])
@pytest.mark.parametrize(
    "args,kwargs",
    [
        ([], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {"axis": 0}),
        ([["a", "b", "c", "d", "e"]], {"axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": 0}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": "index"}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": "columns"}),
        ([], {"index": [-3, 0, 3, 0, -2, 1, 3, 4, 6]}),
        ([], {"columns": ["a", "b", "c", "d", "e"]}),
        (
            [],
            {
                "index": [-3, 0, 3, 0, -2, 1, 3, 4, 6],
                "columns": ["a", "b", "c", "d", "e"],
            },
        ),
    ],
)
def test_dataframe_reindex_fill_value(
    reindex_data_numeric, args, kwargs, fill_value
):
    pdf, gdf = reindex_data_numeric.to_pandas(), reindex_data_numeric
    kwargs["fill_value"] = fill_value
    assert_eq(pdf.reindex(*args, **kwargs), gdf.reindex(*args, **kwargs))


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_change_dtype(copy):
    if PANDAS_GE_110:
        kwargs = {"check_freq": False}
    else:
        kwargs = {}
    index = pd.date_range("12/29/2009", periods=10, freq="D")
    columns = ["a", "b", "c", "d", "e"]
    gdf = cudf.datasets.randomdata(
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
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"a": "category"})
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
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"c": float})
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
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"d": str})
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
    gdf = cudf.DataFrame.from_pandas(pdf)

    expect = pdf.sort_index()
    got = gdf.sort_index()

    assert_eq(expect, got, check_index_type=True)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 3, 1),
        [3.0, 1.0, np.nan],
        # Test for single column MultiIndex
        pd.MultiIndex.from_arrays(
            [
                [2, 0, 1],
            ]
        ),
        pytest.param(
            pd.RangeIndex(2, -1, -1),
            marks=[
                pytest.mark.xfail(
                    condition=PANDAS_LT_140,
                    reason="https://github.com/pandas-dev/pandas/issues/43591",
                )
            ],
        ),
    ],
)
@pytest.mark.parametrize("axis", [0, 1, "index", "columns"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_sort_index(
    index, axis, ascending, inplace, ignore_index, na_position
):
    pdf = pd.DataFrame(
        {"b": [1, 3, 2], "a": [1, 4, 3], "c": [4, 1, 5]},
        index=index,
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

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
        assert_eq(pdf, gdf, check_index_type=True)
    else:
        assert_eq(expected, got, check_index_type=True)


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
    gdf = cudf.DataFrame.from_pandas(pdf)

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

    expect = cudf.DataFrame()
    expect["x"] = data
    expect["y"] = data
    got = expect.head(0)

    for col_name in got.columns:
        assert expect[col_name].dtype == got[col_name].dtype

    expect = cudf.Series(data)
    got = expect.head(0)

    assert expect.dtype == got.dtype


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_series_list_nanasnull(nan_as_null):
    data = [1.0, 2.0, 3.0, np.nan, None]

    expect = pa.array(data, from_pandas=nan_as_null)
    got = cudf.Series(data, nan_as_null=nan_as_null).to_arrow()

    # Bug in Arrow 0.14.1 where NaNs aren't handled
    expect = expect.cast("int64", safe=False)
    got = got.cast("int64", safe=False)

    assert pa.Array.equals(expect, got)


def test_column_assignment():
    gdf = cudf.datasets.randomdata(
        nrows=20, dtypes={"a": "category", "b": int, "c": float}
    )
    new_cols = ["q", "r", "s"]
    gdf.columns = new_cols
    assert list(gdf.columns) == new_cols


def test_select_dtype():
    gdf = cudf.datasets.randomdata(
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

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
        lfunc_args_and_kwargs=([], {"includes": ["Foo"]}),
        rfunc_args_and_kwargs=([], {"includes": ["Foo"]}),
    )

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
        lfunc_args_and_kwargs=(
            [],
            {"exclude": np.number, "include": np.number},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"exclude": np.number, "include": np.number},
        ),
    )

    gdf = cudf.DataFrame(
        {"A": [3, 4, 5], "C": [1, 2, 3], "D": ["a", "b", "c"]}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(include=["object", "int", "category"]),
        gdf.select_dtypes(include=["object", "int", "category"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["object"], exclude=["category"]),
        gdf.select_dtypes(include=["object"], exclude=["category"]),
    )

    gdf = cudf.DataFrame({"a": range(10), "b": range(10, 20)})
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

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
    )

    gdf = cudf.DataFrame(
        {"a": cudf.Series([], dtype="int"), "b": cudf.Series([], dtype="str")}
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

    gdf = cudf.DataFrame(
        {"int_col": [0, 1, 2], "list_col": [[1, 2], [3, 4], [5, 6]]}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes("int64"),
        gdf.select_dtypes("int64"),
    )


def test_select_dtype_datetime():
    gdf = cudf.datasets.timeseries(
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
    gdf = cudf.datasets.timeseries(
        start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={"x": int}
    )
    gdf = gdf.reset_index()
    pdf = gdf.to_pandas()

    assert_exceptions_equal(
        pdf.select_dtypes,
        gdf.select_dtypes,
        (["datetime64[ms]"],),
        (["datetime64[ms]"],),
    )


def test_dataframe_describe_exclude():
    np.random.seed(12)
    data_length = 10000

    df = cudf.DataFrame()
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

    df = cudf.DataFrame()
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

    df = cudf.DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe()
    pdf_results = pdf.describe()

    assert_eq(pdf_results, gdf_results)


def test_series_describe_include_all():
    np.random.seed(12)
    data_length = 10000

    df = cudf.DataFrame()
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

    df = cudf.DataFrame()
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
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf._get_numeric_data(), gdf._get_numeric_data())


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("period", [-15, -1, 0, 1, 15])
@pytest.mark.parametrize("data_empty", [False, True])
def test_shift(dtype, period, data_empty):
    # TODO : this function currently tests for series.shift()
    # but should instead test for dataframe.shift()
    if data_empty:
        data = None
    else:
        if dtype == np.int8:
            # to keep data in range
            data = gen_rand(dtype, 10, low=-2, high=2)
        else:
            data = gen_rand(dtype, 10)

    gs = cudf.DataFrame({"a": cudf.Series(data, dtype=dtype)})
    ps = pd.DataFrame({"a": pd.Series(data, dtype=dtype)})

    shifted_outcome = gs.a.shift(period)
    expected_outcome = ps.a.shift(period)

    # pandas uses NaNs to signal missing value and force converts the
    # results columns to float types
    if data_empty:
        assert_eq(
            shifted_outcome,
            expected_outcome,
            check_index_type=False,
            check_dtype=False,
        )
    else:
        assert_eq(shifted_outcome, expected_outcome, check_dtype=False)


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

    gdf = cudf.DataFrame({"a": cudf.Series(data, dtype=dtype)})
    pdf = pd.DataFrame({"a": pd.Series(data, dtype=dtype)})

    expected_outcome = pdf.a.diff(period)
    diffed_outcome = gdf.a.diff(period).astype(expected_outcome.dtype)

    if data_empty:
        assert_eq(diffed_outcome, expected_outcome, check_index_type=False)
    else:
        assert_eq(diffed_outcome, expected_outcome)


@pytest.mark.parametrize("df", _dataframe_na_data())
@pytest.mark.parametrize("nan_as_null", [True, False, None])
def test_dataframe_isnull_isna(df, nan_as_null):

    gdf = cudf.DataFrame.from_pandas(df, nan_as_null=nan_as_null)

    assert_eq(df.isnull(), gdf.isnull())
    assert_eq(df.isna(), gdf.isna())

    # Test individual columns
    for col in df:
        assert_eq(df[col].isnull(), gdf[col].isnull())
        assert_eq(df[col].isna(), gdf[col].isna())


@pytest.mark.parametrize("df", _dataframe_na_data())
@pytest.mark.parametrize("nan_as_null", [True, False, None])
def test_dataframe_notna_notnull(df, nan_as_null):

    gdf = cudf.DataFrame.from_pandas(df, nan_as_null=nan_as_null)

    assert_eq(df.notnull(), gdf.notnull())
    assert_eq(df.notna(), gdf.notna())

    # Test individual columns
    for col in df:
        assert_eq(df[col].notnull(), gdf[col].notnull())
        assert_eq(df[col].notna(), gdf[col].notna())


def test_ndim():
    pdf = pd.DataFrame({"x": range(5), "y": range(5, 10)})
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert pdf.ndim == gdf.ndim
    assert pdf.x.ndim == gdf.x.ndim

    s = pd.Series(dtype="float64")
    gs = cudf.Series()
    assert s.ndim == gs.ndim


@pytest.mark.parametrize(
    "decimals",
    [
        -3,
        0,
        5,
        pd.Series([1, 4, 3, -6], index=["w", "x", "y", "z"]),
        cudf.Series([-4, -2, 12], index=["x", "y", "z"]),
        {"w": -1, "x": 15, "y": 2},
    ],
)
def test_dataframe_round(decimals):
    pdf = pd.DataFrame(
        {
            "w": np.arange(0.5, 10.5, 1),
            "x": np.random.normal(-100, 100, 10),
            "y": np.array(
                [
                    14.123,
                    2.343,
                    np.nan,
                    0.0,
                    -8.302,
                    np.nan,
                    94.313,
                    -112.236,
                    -8.029,
                    np.nan,
                ]
            ),
            "z": np.repeat([-0.6459412758761901], 10),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    if isinstance(decimals, cudf.Series):
        pdecimals = decimals.to_pandas()
    else:
        pdecimals = decimals

    result = gdf.round(decimals)
    expected = pdf.round(pdecimals)
    assert_eq(result, expected)

    # with nulls, maintaining existing null mask
    for c in pdf.columns:
        arr = pdf[c].to_numpy().astype("float64")  # for pandas nulls
        arr.ravel()[np.random.choice(10, 5, replace=False)] = np.nan
        pdf[c] = gdf[c] = arr

    result = gdf.round(decimals)
    expected = pdf.round(pdecimals)

    assert_eq(result, expected)


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
        pdata = pd.Series(data=data).replace([None], False)
        gdata = cudf.Series.from_pandas(pdata)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"]).replace([None], False)
        gdata = cudf.DataFrame.from_pandas(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.all(bool_only=True)
            expected = pdata.all(bool_only=True)
            assert_eq(got, expected)
        else:
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
        pdata = pd.Series(data=data)
        gdata = cudf.Series.from_pandas(pdata)

        if axis == 1:
            with pytest.raises(NotImplementedError):
                gdata.any(axis=axis)
        else:
            got = gdata.any(axis=axis)
            expected = pdata.any(axis=axis)
            assert_eq(got, expected)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"])
        gdata = cudf.DataFrame.from_pandas(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.any(bool_only=True)
            expected = pdata.any(bool_only=True)
            assert_eq(got, expected)
        else:
            with pytest.raises(NotImplementedError):
                gdata.any(level="a")

        got = gdata.any(axis=axis)
        expected = pdata.any(axis=axis)
        assert_eq(got, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_empty_dataframe_any(axis):
    pdf = pd.DataFrame({}, columns=["a", "b"])
    gdf = cudf.DataFrame.from_pandas(pdf)
    got = gdf.any(axis=axis)
    expected = pdf.any(axis=axis)
    assert_eq(got, expected, check_index_type=False)


@pytest.mark.parametrize("a", [[], ["123"]])
@pytest.mark.parametrize("b", ["123", ["123"]])
@pytest.mark.parametrize(
    "misc_data",
    ["123", ["123"] * 20, 123, [1, 2, 0.8, 0.9] * 50, 0.9, 0.00001],
)
@pytest.mark.parametrize("non_list_data", [123, "abc", "zyx", "rapids", 0.8])
def test_create_dataframe_cols_empty_data(a, b, misc_data, non_list_data):
    expected = pd.DataFrame({"a": a})
    actual = cudf.DataFrame.from_pandas(expected)
    expected["b"] = b
    actual["b"] = b
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": []})
    actual = cudf.DataFrame.from_pandas(expected)
    expected["b"] = misc_data
    actual["b"] = misc_data
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": a})
    actual = cudf.DataFrame.from_pandas(expected)
    expected["b"] = non_list_data
    actual["b"] = non_list_data
    assert_eq(actual, expected)


def test_empty_dataframe_describe():
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = cudf.from_pandas(pdf)

    expected = pdf.describe()
    actual = gdf.describe()

    assert_eq(expected, actual)


def test_as_column_types():
    col = column.as_column(cudf.Series([]))
    assert_eq(col.dtype, np.dtype("float64"))
    gds = cudf.Series(col)
    pds = pd.Series(pd.Series([], dtype="float64"))

    assert_eq(pds, gds)

    col = column.as_column(cudf.Series([]), dtype="float32")
    assert_eq(col.dtype, np.dtype("float32"))
    gds = cudf.Series(col)
    pds = pd.Series(pd.Series([], dtype="float32"))

    assert_eq(pds, gds)

    col = column.as_column(cudf.Series([]), dtype="str")
    assert_eq(col.dtype, np.dtype("object"))
    gds = cudf.Series(col)
    pds = pd.Series(pd.Series([], dtype="str"))

    assert_eq(pds, gds)

    col = column.as_column(cudf.Series([]), dtype="object")
    assert_eq(col.dtype, np.dtype("object"))
    gds = cudf.Series(col)
    pds = pd.Series(pd.Series([], dtype="object"))

    assert_eq(pds, gds)

    pds = pd.Series(np.array([1, 2, 3]), dtype="float32")
    gds = cudf.Series(column.as_column(np.array([1, 2, 3]), dtype="float32"))

    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 3], dtype="float32")
    gds = cudf.Series([1, 2, 3], dtype="float32")

    assert_eq(pds, gds)

    pds = pd.Series([], dtype="float64")
    gds = cudf.Series(column.as_column(pds))
    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 4], dtype="int64")
    gds = cudf.Series(column.as_column(cudf.Series([1, 2, 4]), dtype="int64"))

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="float32")
    gds = cudf.Series(
        column.as_column(cudf.Series([1.2, 18.0, 9.0]), dtype="float32")
    )

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="str")
    gds = cudf.Series(
        column.as_column(cudf.Series([1.2, 18.0, 9.0]), dtype="str")
    )

    assert_eq(pds, gds)

    pds = pd.Series(pd.Index(["1", "18", "9"]), dtype="int")
    gds = cudf.Series(cudf.StringIndex(["1", "18", "9"]), dtype="int")

    assert_eq(pds, gds)


def test_one_row_head():
    gdf = cudf.DataFrame({"name": ["carl"], "score": [100]}, index=[123])
    pdf = gdf.to_pandas()

    head_gdf = gdf.head()
    head_pdf = pdf.head()

    assert_eq(head_pdf, head_gdf)


@pytest.mark.parametrize("dtype", ALL_TYPES)
@pytest.mark.parametrize(
    "np_dtype,pd_dtype",
    [
        tuple(item)
        for item in cudf.utils.dtypes.np_dtypes_to_pandas_dtypes.items()
    ],
)
def test_series_astype_pandas_nullable(dtype, np_dtype, pd_dtype):
    source = cudf.Series([0, 1, None], dtype=dtype)

    expect = source.astype(np_dtype)
    got = source.astype(pd_dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("as_dtype", NUMERIC_TYPES)
def test_series_astype_numeric_to_numeric(dtype, as_dtype):
    psr = pd.Series([1, 2, 4, 3], dtype=dtype)
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("as_dtype", NUMERIC_TYPES)
def test_series_astype_numeric_to_numeric_nulls(dtype, as_dtype):
    data = [1, 2, None, 3]
    sr = cudf.Series(data, dtype=dtype)
    got = sr.astype(as_dtype)
    expect = cudf.Series([1, 2, None, 3], dtype=as_dtype)
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
    gsr = cudf.from_pandas(psr)
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
    gsr = cudf.from_pandas(psr)
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
    gsr = cudf.from_pandas(psr)
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
    sr = cudf.Series([base_date], dtype=dtype)
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
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize("ordered", [True, False])
def test_series_astype_to_categorical_ordered(ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    gsr = cudf.from_pandas(psr)

    ordered_dtype_pd = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=ordered
    )
    ordered_dtype_gd = cudf.CategoricalDtype.from_pandas(ordered_dtype_pd)
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
    gd_dtype = cudf.CategoricalDtype.from_pandas(pd_dtype)
    gd_to_dtype = cudf.CategoricalDtype.from_pandas(pd_to_dtype)

    psr = pd.Series([1, 2, 3], dtype=pd_dtype)
    gsr = cudf.Series([1, 2, 3], dtype=gd_dtype)

    expect = psr.astype(pd_to_dtype)
    got = gsr.astype(gd_to_dtype)

    assert_eq(expect, got)


def test_series_astype_null_cases():
    data = [1, 2, None, 3]

    # numerical to other
    assert_eq(cudf.Series(data, dtype="str"), cudf.Series(data).astype("str"))

    assert_eq(
        cudf.Series(data, dtype="category"),
        cudf.Series(data).astype("category"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="int32").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="uint32").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="datetime64[ms]"),
        cudf.Series(data).astype("datetime64[ms]"),
    )

    # categorical to other
    assert_eq(
        cudf.Series(data, dtype="str"),
        cudf.Series(data, dtype="category").astype("str"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="category").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="datetime64[ms]"),
        cudf.Series(data, dtype="category").astype("datetime64[ms]"),
    )

    # string to other
    assert_eq(
        cudf.Series([1, 2, None, 3], dtype="int32"),
        cudf.Series(["1", "2", None, "3"]).astype("int32"),
    )

    assert_eq(
        cudf.Series(
            ["2001-01-01", "2001-02-01", None, "2001-03-01"],
            dtype="datetime64[ms]",
        ),
        cudf.Series(["2001-01-01", "2001-02-01", None, "2001-03-01"]).astype(
            "datetime64[ms]"
        ),
    )

    assert_eq(
        cudf.Series(["a", "b", "c", None], dtype="category").to_pandas(),
        cudf.Series(["a", "b", "c", None]).astype("category").to_pandas(),
    )

    # datetime to other
    data = [
        "2001-01-01 00:00:00.000000",
        "2001-02-01 00:00:00.000000",
        None,
        "2001-03-01 00:00:00.000000",
    ]
    assert_eq(
        cudf.Series(data),
        cudf.Series(data, dtype="datetime64[us]").astype("str"),
    )

    assert_eq(
        pd.Series(data, dtype="datetime64[ns]").astype("category"),
        cudf.from_pandas(pd.Series(data, dtype="datetime64[ns]")).astype(
            "category"
        ),
    )


def test_series_astype_null_categorical():
    sr = cudf.Series([None, None, None], dtype="category")
    expect = cudf.Series([None, None, None], dtype="int32")
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
    gdf = cudf.DataFrame(data, index=["count", "mean", "std", "min"])

    assert_eq(pdf, gdf)

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)


def test_create_dataframe_column():
    pdf = pd.DataFrame(columns=["a", "b", "c"], index=["A", "Z", "X"])
    gdf = cudf.DataFrame(columns=["a", "b", "c"], index=["A", "Z", "X"])

    assert_eq(pdf, gdf)

    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [2, 3, 5]},
        columns=["a", "b", "c"],
        index=["A", "Z", "X"],
    )
    gdf = cudf.DataFrame(
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
    pds = pd.Series(data=data)
    gds = cudf.Series(data)

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
            marks=pytest.mark.xfail(raises=TypeError),
        ),
    ],
)
def test_series_values_property(data):
    pds = pd.Series(data=data)
    gds = cudf.Series(data)
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
        {"A": [], "B": []},
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
    gdf = cudf.DataFrame.from_pandas(pdf)

    pmtr = pdf.values
    gmtr = gdf.values.get()

    np.testing.assert_array_equal(pmtr, gmtr)


def test_numeric_alpha_value_counts():
    pdf = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    gdf = cudf.DataFrame(
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
        pd.DataFrame({"a": ["a", "b", "c"]}, dtype="category"),
        pd.DataFrame({"a": ["a", "b", "c"]}),
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
        pd.Series(["a", "b", "c"]),
        pd.Series(["a", "b", "c"], dtype="category"),
        pd.DataFrame({"a": ["a", "b", "c"]}, dtype="category"),
    ],
)
def test_isin_dataframe(data, values):
    pdf = data
    gdf = cudf.from_pandas(pdf)

    if cudf.api.types.is_scalar(values):
        assert_exceptions_equal(
            lfunc=pdf.isin,
            rfunc=gdf.isin,
            lfunc_args_and_kwargs=([values],),
            rfunc_args_and_kwargs=([values],),
        )
    else:
        try:
            expected = pdf.isin(values)
        except ValueError as e:
            if str(e) == "Lengths must match.":
                pytest.xfail(
                    not PANDAS_GE_110,
                    "https://github.com/pandas-dev/pandas/issues/34256",
                )
        except TypeError as e:
            # Can't do isin with different categories
            if str(e) == (
                "Categoricals can only be compared if 'categories' "
                "are the same."
            ):
                return

        if isinstance(values, (pd.DataFrame, pd.Series)):
            values = cudf.from_pandas(values)

        got = gdf.isin(values)
        assert_eq(got, expected)


def test_constructor_properties():
    df = cudf.DataFrame()
    key1 = "a"
    key2 = "b"
    val1 = np.array([123], dtype=np.float64)
    val2 = np.array([321], dtype=np.float64)
    df[key1] = val1
    df[key2] = val2

    # Correct use of _constructor_sliced (for DataFrame)
    assert_eq(df[key1], df._constructor_sliced(val1, name=key1))

    # Correct use of _constructor_expanddim (for cudf.Series)
    assert_eq(df, df[key2]._constructor_expanddim({key1: val1, key2: val2}))

    # Incorrect use of _constructor_sliced (Raises for cudf.Series)
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

    gdf = cudf.DataFrame()

    gdf["foo"] = cudf.Series(data, dtype=dtype)
    gdf["bar"] = cudf.Series(data, dtype=dtype)

    insert_data = cudf.Series(data, dtype=dtype)

    expect = cudf.DataFrame()
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

    insert_data = cudf.Series.from_pandas(pd.Series(data, dtype="str"))
    expect_data = cudf.Series(data, dtype=as_dtype)

    gdf = cudf.DataFrame()
    expect = cudf.DataFrame()

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

    gdf = cudf.DataFrame()
    expect = cudf.DataFrame()

    gdf["foo"] = cudf.Series(data, dtype="datetime64[ms]")
    gdf["bar"] = cudf.Series(data, dtype="datetime64[ms]")

    if as_dtype == "int64":
        expect["foo"] = cudf.Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
        expect["bar"] = cudf.Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
    elif as_dtype == "str":
        expect["foo"] = cudf.Series(data, dtype="str")
        expect["bar"] = cudf.Series(data, dtype="str")
    elif as_dtype == "category":
        expect["foo"] = cudf.Series(gdf["foo"], dtype="category")
        expect["bar"] = cudf.Series(gdf["bar"], dtype="category")
    else:
        expect["foo"] = cudf.Series(data, dtype=as_dtype)
        expect["bar"] = cudf.Series(data, dtype=as_dtype)

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
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_eq(pdf.astype(as_dtype), gdf.astype(as_dtype))


@pytest.mark.parametrize("ordered", [True, False])
def test_df_astype_to_categorical_ordered(ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    pdf = pd.DataFrame()
    pdf["foo"] = psr
    pdf["bar"] = psr
    gdf = cudf.DataFrame.from_pandas(pdf)

    ordered_dtype_pd = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=ordered
    )
    ordered_dtype_gd = cudf.CategoricalDtype.from_pandas(ordered_dtype_pd)

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
    df = cudf.DataFrame()
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
    ],
)
def test_series_astype_error_handling(errors):
    sr = cudf.Series(["random", "words"])
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

    sr = cudf.Series(data, dtype=dtype)

    expect = cudf.DataFrame()
    expect["foo"] = sr
    expect["bar"] = sr
    got = cudf.DataFrame({"foo": data, "bar": data}, dtype=dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        cudf.datasets.randomdata(
            nrows=10, dtypes={"a": "category", "b": int, "c": float, "d": int}
        ),
        cudf.datasets.randomdata(
            nrows=10, dtypes={"a": "category", "b": int, "c": float, "d": str}
        ),
        cudf.datasets.randomdata(
            nrows=10, dtypes={"a": bool, "b": int, "c": float, "d": str}
        ),
        cudf.DataFrame(),
        cudf.DataFrame({"a": [0, 1, 2], "b": [1, None, 3]}),
        cudf.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [7, np.NaN, 9, 10],
                "c": [np.NaN, np.NaN, np.NaN, np.NaN],
                "d": cudf.Series([None, None, None, None], dtype="int64"),
                "e": [100, None, 200, None],
                "f": cudf.Series([10, None, np.NaN, 11], nan_as_null=False),
            }
        ),
        cudf.DataFrame(
            {
                "a": [10, 11, 12, 13, 14, 15],
                "b": cudf.Series(
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

    assert_eq(expected, got, check_exact=False)


@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "product", "mean", "var", "std"]
)
def test_rowwise_ops_nullable_dtypes_all_null(op):
    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [7, np.NaN, 9, 10],
            "c": [np.NaN, np.NaN, np.NaN, np.NaN],
            "d": cudf.Series([None, None, None, None], dtype="int64"),
            "e": [100, None, 200, None],
            "f": cudf.Series([10, None, np.NaN, 11], nan_as_null=False),
        }
    )

    expected = cudf.Series([None, None, None, None], dtype="float64")

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
            cudf.Series(
                [10.0, None, np.NaN, 2234.0, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "min",
            cudf.Series(
                [10.0, None, np.NaN, 13.0, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "sum",
            cudf.Series(
                [20.0, None, np.NaN, 2247.0, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "product",
            cudf.Series(
                [100.0, None, np.NaN, 29042.0, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "mean",
            cudf.Series(
                [10.0, None, np.NaN, 1123.5, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "var",
            cudf.Series(
                [0.0, None, np.NaN, 1233210.25, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
        (
            "std",
            cudf.Series(
                [0.0, None, np.NaN, 1110.5, None, np.NaN],
                dtype="float64",
                nan_as_null=False,
            ),
        ),
    ],
)
def test_rowwise_ops_nullable_dtypes_partial_null(op, expected):
    gdf = cudf.DataFrame(
        {
            "a": [10, 11, 12, 13, 14, 15],
            "b": cudf.Series(
                [10, None, np.NaN, 2234, None, np.NaN],
                nan_as_null=False,
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
        (
            "max",
            cudf.Series(
                [10, None, None, 2234, None, 453],
                dtype="int64",
            ),
        ),
        (
            "min",
            cudf.Series(
                [10, None, None, 13, None, 15],
                dtype="int64",
            ),
        ),
        (
            "sum",
            cudf.Series(
                [20, None, None, 2247, None, 468],
                dtype="int64",
            ),
        ),
        (
            "product",
            cudf.Series(
                [100, None, None, 29042, None, 6795],
                dtype="int64",
            ),
        ),
        (
            "mean",
            cudf.Series(
                [10.0, None, None, 1123.5, None, 234.0],
                dtype="float32",
            ),
        ),
        (
            "var",
            cudf.Series(
                [0.0, None, None, 1233210.25, None, 47961.0],
                dtype="float32",
            ),
        ),
        (
            "std",
            cudf.Series(
                [0.0, None, None, 1110.5, None, 219.0],
                dtype="float32",
            ),
        ),
    ],
)
def test_rowwise_ops_nullable_int_dtypes(op, expected):
    gdf = cudf.DataFrame(
        {
            "a": [10, 11, None, 13, None, 15],
            "b": cudf.Series(
                [10, None, 323, 2234, None, 453],
                nan_as_null=False,
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
        {
            "t1": cudf.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": cudf.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ms]"
            ),
        },
        {
            "t1": cudf.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": cudf.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ns]"
            ),
            "t3": cudf.Series(
                ["1960-08-31 06:00:00", "2030-08-02 10:00:00"], dtype="<M8[s]"
            ),
        },
        {
            "t1": cudf.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": cudf.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[us]"
            ),
        },
        {
            "t1": cudf.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": cudf.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ms]"
            ),
            "i1": cudf.Series([1001, 2002], dtype="int64"),
        },
        {
            "t1": cudf.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": cudf.Series(["1940-08-31 06:00:00", None], dtype="<M8[ms]"),
            "i1": cudf.Series([1001, 2002], dtype="int64"),
        },
        {
            "t1": cudf.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "i1": cudf.Series([1001, 2002], dtype="int64"),
            "f1": cudf.Series([-100.001, 123.456], dtype="float64"),
        },
        {
            "t1": cudf.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "i1": cudf.Series([1001, 2002], dtype="int64"),
            "f1": cudf.Series([-100.001, 123.456], dtype="float64"),
            "b1": cudf.Series([True, False], dtype="bool"),
        },
    ],
)
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("skipna", [True, False])
def test_rowwise_ops_datetime_dtypes(data, op, skipna):

    gdf = cudf.DataFrame(data)

    pdf = gdf.to_pandas()

    got = getattr(gdf, op)(axis=1, skipna=skipna)
    expected = getattr(pdf, op)(axis=1, skipna=skipna)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data,op,skipna",
    [
        (
            {
                "t1": cudf.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": cudf.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "max",
            True,
        ),
        (
            {
                "t1": cudf.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": cudf.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "min",
            False,
        ),
        (
            {
                "t1": cudf.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": cudf.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "min",
            True,
        ),
    ],
)
def test_rowwise_ops_datetime_dtypes_2(data, op, skipna):

    gdf = cudf.DataFrame(data)

    pdf = gdf.to_pandas()

    got = getattr(gdf, op)(axis=1, skipna=skipna)
    expected = getattr(pdf, op)(axis=1, skipna=skipna)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        (
            {
                "t1": pd.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ns]",
                ),
                "t2": pd.Series(
                    ["1940-08-31 06:00:00", pd.NaT], dtype="<M8[ns]"
                ),
            }
        )
    ],
)
def test_rowwise_ops_datetime_dtypes_pdbug(data):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    expected = pdf.max(axis=1, skipna=False)
    got = gdf.max(axis=1, skipna=False)

    if PANDAS_GE_120:
        assert_eq(got, expected)
    else:
        # PANDAS BUG: https://github.com/pandas-dev/pandas/issues/36907
        with pytest.raises(AssertionError, match="numpy array are different"):
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
    gdf = cudf.DataFrame.from_pandas(pdf)

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


@pytest.mark.parametrize(
    "data",
    [{"A": [1, 2, 3], "B": ["a", "b", "c"]}],
)
def test_insert_NA(data):
    pdf = pd.DataFrame.from_dict(data)
    gdf = cudf.DataFrame.from_pandas(pdf)

    pdf["C"] = pd.NA
    gdf["C"] = cudf.NA
    assert_eq(pdf, gdf)


def test_cov():
    gdf = cudf.datasets.randomdata(10)
    pdf = gdf.to_pandas()

    assert_eq(pdf.cov(), gdf.cov())


@pytest.mark.xfail(reason="cupy-based cov does not support nulls")
def test_cov_nans():
    pdf = pd.DataFrame()
    pdf["a"] = [None, None, None, 2.00758632, None]
    pdf["b"] = [0.36403686, None, None, None, None]
    pdf["c"] = [None, None, None, 0.64882227, None]
    pdf["d"] = [None, -1.46863125, None, 1.22477948, -0.06031689]
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.cov(), gdf.cov())


@pytest.mark.parametrize(
    "gsr",
    [
        cudf.Series([4, 2, 3]),
        cudf.Series([4, 2, 3], index=["a", "b", "c"]),
        cudf.Series([4, 2, 3], index=["a", "b", "d"]),
        cudf.Series([4, 2], index=["a", "b"]),
        cudf.Series([4, 2, 3], index=cudf.core.index.RangeIndex(0, 3)),
        pytest.param(
            cudf.Series([4, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"]),
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
    data = [[3.0, 2.0, 5.0], [3.0, None, 5.0], [6.0, 7.0, np.nan]]
    data = dict(zip(colnames, data))

    gsr = gsr.astype("float64")

    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas(nullable=True)

    psr = gsr.to_pandas(nullable=True)

    expect = op(pdf, psr)
    got = op(gdf, gsr).to_pandas(nullable=True)
    assert_eq(expect, got, check_dtype=False)

    expect = op(psr, pdf)
    got = op(gsr, gdf).to_pandas(nullable=True)
    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        # comparison ops will temporarily XFAIL
        # see PR  https://github.com/rapidsai/cudf/pull/7491
        pytest.param(operator.eq, marks=pytest.mark.xfail()),
        pytest.param(operator.lt, marks=pytest.mark.xfail()),
        pytest.param(operator.le, marks=pytest.mark.xfail()),
        pytest.param(operator.gt, marks=pytest.mark.xfail()),
        pytest.param(operator.ge, marks=pytest.mark.xfail()),
        pytest.param(operator.ne, marks=pytest.mark.xfail()),
    ],
)
@pytest.mark.parametrize(
    "gsr", [cudf.Series([1, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"])]
)
def test_df_sr_binop_col_order(gsr, op):
    colnames = [0, 1, 2]
    data = [[0, 2, 5], [3, None, 5], [6, 7, np.nan]]
    data = dict(zip(colnames, data))

    gdf = cudf.DataFrame(data)
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

    gdf = cudf.from_pandas(df)

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
    gdf = cudf.from_pandas(df)

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
    gdf = cudf.from_pandas(df)

    expected = (
        gdf.B._column.categories.memory_usage
        + gdf.B._column.codes.memory_usage
    )

    # Check cat column
    assert gdf.B.memory_usage(deep=True, index=False) == expected

    # Check cat index
    assert gdf.set_index("B").index.memory_usage(deep=True) == expected


def test_memory_usage_list():
    df = cudf.DataFrame({"A": [[0, 1, 2, 3], [4, 5, 6], [7, 8], [9]]})
    expected = (
        df.A._column.offsets.memory_usage + df.A._column.elements.memory_usage
    )
    assert expected == df.A.memory_usage()


@pytest.mark.parametrize("rows", [10, 100])
def test_memory_usage_multi(rows):
    # We need to sample without replacement to guarantee that the size of the
    # levels are always the same.
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": np.random.choice(
                np.arange(rows, dtype="int64"), rows, replace=False
            ),
            "C": np.random.choice(
                np.arange(rows, dtype="float64"), rows, replace=False
            ),
        }
    ).set_index(["B", "C"])
    gdf = cudf.from_pandas(df)
    # Assume MultiIndex memory footprint is just that
    # of the underlying columns, levels, and codes
    expect = rows * 16  # Source Columns
    expect += rows * 16  # Codes
    expect += rows * 8  # Level 0
    expect += rows * 8  # Level 1

    assert expect == gdf.index.memory_usage(deep=True)


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
    gdf = cudf.datasets.randomdata(5)
    with pytest.raises(
        ValueError, match=("All columns must be of equal length")
    ):
        gdf[key] = list_input


@pytest.mark.parametrize(
    "series_input",
    [
        pytest.param(cudf.Series([1, 2, 3, 4]), id="smaller_cudf"),
        pytest.param(cudf.Series([1, 2, 3, 4, 5, 6]), id="larger_cudf"),
        pytest.param(cudf.Series([1, 2, 3], index=[4, 5, 6]), id="index_cudf"),
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
    gdf = cudf.datasets.randomdata(5)
    pdf = gdf.to_pandas()

    pandas_input = series_input
    if isinstance(pandas_input, cudf.Series):
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
    gdf = cudf.DataFrame()
    pdf[("a", "b")] = [1]
    gdf[("a", "b")] = [1]
    assert_eq(pdf, gdf)
    assert_eq(pdf.columns, gdf.columns)


def test_init_multiindex_from_dict():
    pdf = pd.DataFrame({("a", "b"): [1]})
    gdf = cudf.DataFrame({("a", "b"): [1]})
    assert_eq(pdf, gdf)
    assert_eq(pdf.columns, gdf.columns)


def test_change_column_dtype_in_empty():
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf, gdf)
    pdf["b"] = pdf["b"].astype("int64")
    gdf["b"] = gdf["b"].astype("int64")
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("dtype", ["int64", "str"])
def test_dataframe_from_dictionary_series_same_name_index(dtype):
    pd_idx1 = pd.Index([1, 2, 0], name="test_index").astype(dtype)
    pd_idx2 = pd.Index([2, 0, 1], name="test_index").astype(dtype)
    pd_series1 = pd.Series([1, 2, 3], index=pd_idx1)
    pd_series2 = pd.Series([1, 2, 3], index=pd_idx2)

    gd_idx1 = cudf.from_pandas(pd_idx1)
    gd_idx2 = cudf.from_pandas(pd_idx2)
    gd_series1 = cudf.Series([1, 2, 3], index=gd_idx1)
    gd_series2 = cudf.Series([1, 2, 3], index=gd_idx2)

    expect = pd.DataFrame({"a": pd_series1, "b": pd_series2})
    got = cudf.DataFrame({"a": gd_series1, "b": gd_series2})

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
    gdf = cudf.DataFrame.from_pandas(pdf)

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
    gs_where = cudf.from_pandas(data)

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
        gs_condition = cudf.from_pandas(condition)
    else:
        gs_condition = condition

    ps_other = other
    if type(other).__module__.split(".")[0] == "pandas":
        gs_other = cudf.from_pandas(other)
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
                .to_numpy(),
            )
            assert_eq(expect_where.cat.categories, got_where.cat.categories)

            np.testing.assert_array_equal(
                expect_mask.cat.codes,
                got_mask.cat.codes.astype(expect_mask.cat.codes.dtype)
                .fillna(-1)
                .to_numpy(),
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
        assert_exceptions_equal(
            lfunc=ps_where.where,
            rfunc=gs_where.where,
            lfunc_args_and_kwargs=(
                [ps_condition],
                {"other": ps_other, "inplace": inplace},
            ),
            rfunc_args_and_kwargs=(
                [gs_condition],
                {"other": gs_other, "inplace": inplace},
            ),
            compare_error_message=False
            if error is NotImplementedError
            else True,
        )

        assert_exceptions_equal(
            lfunc=ps_mask.mask,
            rfunc=gs_mask.mask,
            lfunc_args_and_kwargs=(
                [ps_condition],
                {"other": ps_other, "inplace": inplace},
            ),
            rfunc_args_and_kwargs=(
                [gs_condition],
                {"other": gs_other, "inplace": inplace},
            ),
            compare_error_message=False,
        )


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
    gs = cudf.from_pandas(data)

    ps_condition = condition
    if type(condition).__module__.split(".")[0] == "pandas":
        gs_condition = cudf.from_pandas(condition)
    else:
        gs_condition = condition

    ps_other = other
    if type(other).__module__.split(".")[0] == "pandas":
        gs_other = cudf.from_pandas(other)
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
            TypeError,
        ),
    ],
)
def test_from_pandas_unsupported_types(data, expected_upcast_type, error):
    pdf = pd.DataFrame({"one_col": data})
    if error is not None:
        with pytest.raises(ValueError):
            cudf.from_pandas(data)

        with pytest.raises(ValueError):
            cudf.Series(data)

        with pytest.raises(error):
            cudf.from_pandas(pdf)

        with pytest.raises(error):
            cudf.DataFrame(pdf)
    else:
        df = cudf.from_pandas(data)

        assert_eq(data, df, check_dtype=False)
        assert df.dtype == expected_upcast_type

        df = cudf.Series(data)
        assert_eq(data, df, check_dtype=False)
        assert df.dtype == expected_upcast_type

        df = cudf.from_pandas(pdf)
        assert_eq(pdf, df, check_dtype=False)
        assert df["one_col"].dtype == expected_upcast_type

        df = cudf.DataFrame(pdf)
        assert_eq(pdf, df, check_dtype=False)
        assert df["one_col"].dtype == expected_upcast_type


@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.parametrize("index", [None, "a", ["a", "b"]])
def test_from_pandas_nan_as_null(nan_as_null, index):

    data = [np.nan, 2.0, 3.0]

    if index is None:
        pdf = pd.DataFrame({"a": data, "b": data})
        expected = cudf.DataFrame(
            {
                "a": column.as_column(data, nan_as_null=nan_as_null),
                "b": column.as_column(data, nan_as_null=nan_as_null),
            }
        )
    else:
        pdf = pd.DataFrame({"a": data, "b": data}).set_index(index)
        expected = cudf.DataFrame(
            {
                "a": column.as_column(data, nan_as_null=nan_as_null),
                "b": column.as_column(data, nan_as_null=nan_as_null),
            }
        )
        expected = cudf.DataFrame(
            {
                "a": column.as_column(data, nan_as_null=nan_as_null),
                "b": column.as_column(data, nan_as_null=nan_as_null),
            }
        )
        expected = expected.set_index(index)

    got = cudf.from_pandas(pdf, nan_as_null=nan_as_null)

    assert_eq(expected, got)


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_from_pandas_for_series_nan_as_null(nan_as_null):

    data = [np.nan, 2.0, 3.0]
    psr = pd.Series(data)

    expected = cudf.Series(column.as_column(data, nan_as_null=nan_as_null))
    got = cudf.from_pandas(psr, nan_as_null=nan_as_null)

    assert_eq(expected, got)


@pytest.mark.parametrize("copy", [True, False])
def test_df_series_dataframe_astype_copy(copy):
    gdf = cudf.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    pdf = gdf.to_pandas()

    assert_eq(
        gdf.astype(dtype="float", copy=copy),
        pdf.astype(dtype="float", copy=copy),
    )
    assert_eq(gdf, pdf)

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    assert_eq(
        gsr.astype(dtype="float", copy=copy),
        psr.astype(dtype="float", copy=copy),
    )
    assert_eq(gsr, psr)

    gsr = cudf.Series([1, 2])
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
    gdf = cudf.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    pdf = gdf.to_pandas()

    assert_eq(
        gdf.astype(dtype={"col1": "float"}, copy=copy),
        pdf.astype(dtype={"col1": "float"}, copy=copy),
    )
    assert_eq(gdf, pdf)

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    assert_eq(
        gsr.astype(dtype={None: "float"}, copy=copy),
        psr.astype(dtype={None: "float"}, copy=copy),
    )
    assert_eq(gsr, psr)

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=([], {"dtype": {"a": "float"}, "copy": copy}),
        rfunc_args_and_kwargs=([], {"dtype": {"a": "float"}, "copy": copy}),
    )

    gsr = cudf.Series([1, 2])
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
    actual = cudf.DataFrame(data, columns=columns)

    assert_eq(expect, actual, check_index_type=len(data) != 0)

    expect = pd.DataFrame(data, columns=None)
    actual = cudf.DataFrame(data, columns=None)

    assert_eq(expect, actual, check_index_type=len(data) != 0)


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
    if isinstance(data, cupy.ndarray):
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
    gdf = cudf.DataFrame(gd_data, columns=cols, index=index)

    assert_eq(pdf, gdf, check_dtype=False)

    # verify with columns
    pdf = pd.DataFrame(pd_data, columns=cols)
    gdf = cudf.DataFrame(gd_data, columns=cols)

    assert_eq(pdf, gdf, check_dtype=False)

    pdf = pd.DataFrame(pd_data)
    gdf = cudf.DataFrame(gd_data)

    assert_eq(pdf, gdf, check_dtype=False)

    if numba_data is not None:
        gdf = cudf.DataFrame(numba_data)
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
    gdf = cudf.DataFrame({"a": col_data})

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
    gdf = cudf.DataFrame({"a": col_data}, index=["dummy_mandatory_index"])

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
    cudf.from_pandas(df).info(buf=buffer, verbose=True)
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
    cudf.from_pandas(df).info(buf=buffer, verbose=True)
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
    cudf.from_pandas(df).info(buf=buffer, verbose=False)
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
    cudf.from_pandas(df).info(buf=buffer, verbose=True, memory_usage="deep")
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    int_values = [1, 2, 3, 4, 5]
    text_values = ["alpha", "beta", "gamma", "delta", "epsilon"]
    float_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    df = cudf.DataFrame(
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

    df = cudf.DataFrame(
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

    df = cudf.DataFrame()

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

    df = cudf.DataFrame(
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

    expected = cudf.Series(cupy.isclose(array1, array2, rtol=rtol, atol=atol))

    actual = cudf.isclose(
        cudf.Series(data1), cudf.Series(data2), rtol=rtol, atol=atol
    )

    assert_eq(expected, actual)
    actual = cudf.isclose(data1, data2, rtol=rtol, atol=atol)

    assert_eq(expected, actual)

    actual = cudf.isclose(
        cupy.array(data1), cupy.array(data2), rtol=rtol, atol=atol
    )

    assert_eq(expected, actual)

    actual = cudf.isclose(
        np.array(data1), np.array(data2), rtol=rtol, atol=atol
    )

    assert_eq(expected, actual)

    actual = cudf.isclose(
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

    expected = cudf.Series(cupy.isclose(array1, array2, equal_nan=equal_nan))

    actual = cudf.isclose(
        cudf.Series(data1), cudf.Series(data2), equal_nan=equal_nan
    )
    assert_eq(expected, actual, check_dtype=False)
    actual = cudf.isclose(data1, data2, equal_nan=equal_nan)
    assert_eq(expected, actual, check_dtype=False)


def test_cudf_isclose_different_index():
    s1 = cudf.Series(
        [-1.9876543, -2.9876654, -3.9876543, -4.1234587, -5.23, -7.00001],
        index=[0, 1, 2, 3, 4, 5],
    )
    s2 = cudf.Series(
        [-1.9876543, -2.9876654, -7.00001, -4.1234587, -5.23, -3.9876543],
        index=[0, 1, 5, 3, 4, 2],
    )

    expected = cudf.Series([True] * 6, index=s1.index)
    assert_eq(expected, cudf.isclose(s1, s2))

    s1 = cudf.Series(
        [-1.9876543, -2.9876654, -3.9876543, -4.1234587, -5.23, -7.00001],
        index=[0, 1, 2, 3, 4, 5],
    )
    s2 = cudf.Series(
        [-1.9876543, -2.9876654, -7.00001, -4.1234587, -5.23, -3.9876543],
        index=[0, 1, 5, 10, 4, 2],
    )

    expected = cudf.Series(
        [True, True, True, False, True, True], index=s1.index
    )
    assert_eq(expected, cudf.isclose(s1, s2))

    s1 = cudf.Series(
        [-1.9876543, -2.9876654, -3.9876543, -4.1234587, -5.23, -7.00001],
        index=[100, 1, 2, 3, 4, 5],
    )
    s2 = cudf.Series(
        [-1.9876543, -2.9876654, -7.00001, -4.1234587, -5.23, -3.9876543],
        index=[0, 1, 100, 10, 4, 2],
    )

    expected = cudf.Series(
        [False, True, True, False, True, False], index=s1.index
    )
    assert_eq(expected, cudf.isclose(s1, s2))


def test_dataframe_to_dict_error():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [9, 5, 3]})
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
    gdf = cudf.from_pandas(df)

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
        pd.Series(index=["a", "b", "c", "d", "e", "f"], dtype="float64"),
        pd.Series(index=[10, 11, 12], dtype="float64"),
        pd.Series(dtype="float64"),
        pd.Series([], dtype="float64"),
    ],
)
def test_series_keys(ps):
    gds = cudf.from_pandas(ps)

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

    gdf = cudf.from_pandas(df)
    other_gd = cudf.from_pandas(other)

    expected = pdf.append(other_pd, sort=sort, ignore_index=ignore_index)
    actual = gdf.append(other_gd, sort=sort, ignore_index=ignore_index)

    if expected.shape != df.shape:
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(expected, actual, check_index_type=not gdf.empty)


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

    gdf = cudf.from_pandas(df)
    if isinstance(other, pd.Series):
        other_gd = cudf.from_pandas(other)
    else:
        other_gd = other

    expected = pdf.append(other_pd, ignore_index=True, sort=sort)
    actual = gdf.append(other_gd, ignore_index=True, sort=sort)

    if expected.shape != df.shape:
        # Ignore the column type comparison because pandas incorrectly
        # returns pd.Index([1, 2, 3], dtype="object") instead
        # of pd.Index([1, 2, 3], dtype="int64")
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=False,
            check_index_type=True,
        )
    else:
        assert_eq(expected, actual, check_index_type=not gdf.empty)


def test_dataframe_append_series_mixed_index():
    df = cudf.DataFrame({"first": [], "d": []})
    sr = cudf.Series([1, 2, 3, 4])

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

    gdf = cudf.from_pandas(df)
    other_gd = [
        cudf.from_pandas(o) if isinstance(o, pd.DataFrame) else o
        for o in other
    ]

    expected = pdf.append(other_pd, sort=sort, ignore_index=ignore_index)
    actual = gdf.append(other_gd, sort=sort, ignore_index=ignore_index)
    if expected.shape != df.shape:
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(expected, actual, check_index_type=not gdf.empty)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": [1, 2, 3, np.nan, None, 6]}),
        pd.Series([1, 2, 3, None, np.nan, 5, 6, np.nan]),
    ],
)
def test_dataframe_bfill(df):
    gdf = cudf.from_pandas(df)

    actual = df.bfill()
    expected = gdf.bfill()
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": [1, 2, 3, np.nan, None, 6]}),
        pd.Series([1, 2, 3, None, np.nan, 5, 6, np.nan]),
    ],
)
def test_dataframe_ffill(df):
    gdf = cudf.from_pandas(df)

    actual = df.ffill()
    expected = gdf.ffill()
    assert_eq(expected, actual)


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

    gdf = cudf.from_pandas(df)
    other_gd = [
        cudf.from_pandas(o) if isinstance(o, pd.DataFrame) else o
        for o in other
    ]

    expected = pdf.append(other_pd, sort=sort, ignore_index=ignore_index)
    actual = gdf.append(other_gd, sort=sort, ignore_index=ignore_index)

    if expected.shape != df.shape:
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=not gdf.empty,
        )
    else:
        assert_eq(expected, actual, check_index_type=not gdf.empty)


def test_dataframe_append_error():
    df = cudf.DataFrame({"a": [1, 2, 3]})
    ps = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match="Can only append a Series if ignore_index=True "
        "or if the Series has a name",
    ):
        df.append(ps)


def test_cudf_arrow_array_error():
    df = cudf.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        df.__arrow_array__()

    sr = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        sr.__arrow_array__()

    sr = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        sr.__arrow_array__()


@pytest.mark.parametrize(
    "make_weights_axis_1",
    [lambda _: None, lambda s: [1] * s, lambda s: np.ones(s)],
)
def test_sample_axis_1(
    sample_n_frac, random_state_tuple_axis_1, make_weights_axis_1
):
    n, frac = sample_n_frac
    pd_random_state, gd_random_state, checker = random_state_tuple_axis_1

    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "float": [0.05, 0.2, 0.3, 0.2, 0.25],
            "int": [1, 3, 5, 4, 2],
        },
    )
    df = cudf.DataFrame.from_pandas(pdf)

    weights = make_weights_axis_1(len(pdf.columns))

    expected = pdf.sample(
        n=n,
        frac=frac,
        replace=False,
        random_state=pd_random_state,
        weights=weights,
        axis=1,
    )
    got = df.sample(
        n=n,
        frac=frac,
        replace=False,
        random_state=gd_random_state,
        weights=weights,
        axis=1,
    )
    checker(expected, got)


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "float": [0.05, 0.2, 0.3, 0.2, 0.25],
                "int": [1, 3, 5, 4, 2],
            },
        ),
        pd.Series([1, 2, 3, 4, 5]),
    ],
)
@pytest.mark.parametrize("replace", [True, False])
def test_sample_axis_0(
    pdf, sample_n_frac, replace, random_state_tuple_axis_0, make_weights_axis_0
):
    n, frac = sample_n_frac
    pd_random_state, gd_random_state, checker = random_state_tuple_axis_0

    df = cudf.from_pandas(pdf)

    pd_weights, gd_weights = make_weights_axis_0(
        len(pdf), isinstance(gd_random_state, np.random.RandomState)
    )
    if (
        not replace
        and not isinstance(gd_random_state, np.random.RandomState)
        and gd_weights is not None
    ):
        pytest.skip(
            "`cupy.random.RandomState` doesn't support weighted sampling "
            "without replacement."
        )

    expected = pdf.sample(
        n=n,
        frac=frac,
        replace=replace,
        random_state=pd_random_state,
        weights=pd_weights,
        axis=0,
    )

    got = df.sample(
        n=n,
        frac=frac,
        replace=replace,
        random_state=gd_random_state,
        weights=gd_weights,
        axis=0,
    )
    checker(expected, got)


@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize(
    "random_state_lib", [cupy.random.RandomState, np.random.RandomState]
)
def test_sample_reproducibility(replace, random_state_lib):
    df = cudf.DataFrame({"a": cupy.arange(0, 1024)})

    n = 1024
    expected = df.sample(n, replace=replace, random_state=random_state_lib(10))
    out = df.sample(n, replace=replace, random_state=random_state_lib(10))

    assert_eq(expected, out)


@pytest.mark.parametrize("axis", [0, 1])
def test_sample_invalid_n_frac_combo(axis):
    n, frac = 2, 0.5
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "float": [0.05, 0.2, 0.3, 0.2, 0.25],
            "int": [1, 3, 5, 4, 2],
        },
    )
    df = cudf.DataFrame.from_pandas(pdf)

    assert_exceptions_equal(
        lfunc=pdf.sample,
        rfunc=df.sample,
        lfunc_args_and_kwargs=([], {"n": n, "frac": frac, "axis": axis}),
        rfunc_args_and_kwargs=([], {"n": n, "frac": frac, "axis": axis}),
    )


@pytest.mark.parametrize("n, frac", [(100, None), (None, 3)])
@pytest.mark.parametrize("axis", [0, 1])
def test_oversample_without_replace(n, frac, axis):
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    df = cudf.DataFrame.from_pandas(pdf)

    assert_exceptions_equal(
        lfunc=pdf.sample,
        rfunc=df.sample,
        lfunc_args_and_kwargs=(
            [],
            {"n": n, "frac": frac, "axis": axis, "replace": False},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"n": n, "frac": frac, "axis": axis, "replace": False},
        ),
    )


@pytest.mark.parametrize("random_state", [None, cupy.random.RandomState(42)])
def test_sample_unsupported_arguments(random_state):
    df = cudf.DataFrame({"float": [0.05, 0.2, 0.3, 0.2, 0.25]})
    with pytest.raises(
        NotImplementedError,
        match="Random sampling with cupy does not support these inputs.",
    ):
        df.sample(
            n=2, replace=False, random_state=random_state, weights=[1] * 5
        )


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
    gdf = cudf.from_pandas(pdf)

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
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.size, gdf.size)


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(dtype="float64"),
        pd.Series(index=[100, 10, 1, 0], dtype="float64"),
        pd.Series([], dtype="float64"),
        pd.Series(["a", "b", "c", "d"]),
        pd.Series(["a", "b", "c", "d"], index=[0, 1, 10, 11]),
    ],
)
def test_series_empty(ps):
    ps = ps
    gs = cudf.from_pandas(ps)

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
    gdf = cudf.DataFrame(data, columns=columns)

    assert_eq(
        pdf,
        gdf,
        check_index_type=len(pdf.index) != 0,
        check_dtype=not (pdf.empty and len(pdf.columns)),
    )


@pytest.mark.parametrize(
    "data, ignore_dtype",
    [
        ([pd.Series([1, 2, 3])], False),
        ([pd.Series(index=[1, 2, 3], dtype="float64")], False),
        ([pd.Series(name="empty series name", dtype="float64")], False),
        (
            [pd.Series([1]), pd.Series([], dtype="float64"), pd.Series([3])],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
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
                pd.Series([], dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], name="abc", dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
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
    gd_data = [cudf.from_pandas(obj) for obj in data]

    expected = pd.DataFrame(data, columns=columns)
    actual = cudf.DataFrame(gd_data, columns=columns)

    if ignore_dtype:
        # When a union is performed to generate columns,
        # the order is never guaranteed. Hence sort by
        # columns before comparison.
        if not expected.columns.equals(actual.columns):
            expected = expected.sort_index(axis=1)
            actual = actual.sort_index(axis=1)
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_index_type=True,
        )
    else:
        assert_eq(expected, actual, check_index_type=True)


@pytest.mark.parametrize(
    "data, ignore_dtype, index",
    [
        ([pd.Series([1, 2, 3])], False, ["a", "b", "c"]),
        ([pd.Series(index=[1, 2, 3], dtype="float64")], False, ["a", "b"]),
        (
            [pd.Series(name="empty series name", dtype="float64")],
            False,
            ["index1"],
        ),
        (
            [pd.Series([1]), pd.Series([], dtype="float64"), pd.Series([3])],
            False,
            ["0", "2", "1"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
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
                pd.Series([], dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
            ["a", "b", "c"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], name="abc", dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
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
    gd_data = [cudf.from_pandas(obj) for obj in data]

    expected = pd.DataFrame(data, columns=columns, index=index)
    actual = cudf.DataFrame(gd_data, columns=columns, index=index)

    if ignore_dtype:
        # When a union is performed to generate columns,
        # the order is never guaranteed. Hence sort by
        # columns before comparison.
        if not expected.columns.equals(actual.columns):
            expected = expected.sort_index(axis=1)
            actual = actual.sort_index(axis=1)
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
                pd.Series([], dtype="float64"),
                pd.Series([3], name="series that is named"),
            ],
            ["_", "+"],
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, ["mean"] * 9),
    ],
)
def test_dataframe_init_from_series_list_with_index_error(data, index):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    assert_exceptions_equal(
        pd.DataFrame,
        cudf.DataFrame,
        ([data], {"index": index}),
        ([gd_data], {"index": index}),
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
    gd_data = [cudf.from_pandas(obj) for obj in data]

    assert_exceptions_equal(
        lfunc=pd.DataFrame,
        rfunc=cudf.DataFrame,
        lfunc_args_and_kwargs=([], {"data": data}),
        rfunc_args_and_kwargs=([], {"data": gd_data}),
        check_exception_type=False,
    )


def test_dataframe_iterrows_itertuples():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

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
        cudf.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [10, 22, 33],
                "c": [0.3234, 0.23432, 0.0],
                "d": ["hello", "world", "hello"],
            }
        ),
        cudf.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["hello", "world", "hello"],
                "c": [0.3234, 0.23432, 0.0],
            }
        ),
        pytest.param(
            cudf.DataFrame(
                {
                    "int_data": [1, 2, 3],
                    "str_data": ["hello", "world", "hello"],
                    "float_data": [0.3234, 0.23432, 0.0],
                    "timedelta_data": cudf.Series(
                        [1, 2, 1], dtype="timedelta64[ns]"
                    ),
                    "datetime_data": cudf.Series(
                        [1, 2, 1], dtype="datetime64[ns]"
                    ),
                }
            ),
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6219"
            ),
        ),
        pytest.param(
            cudf.DataFrame(
                {
                    "int_data": [1, 2, 3],
                    "str_data": ["hello", "world", "hello"],
                    "float_data": [0.3234, 0.23432, 0.0],
                    "timedelta_data": cudf.Series(
                        [1, 2, 1], dtype="timedelta64[ns]"
                    ),
                    "datetime_data": cudf.Series(
                        [1, 2, 1], dtype="datetime64[ns]"
                    ),
                    "category_data": cudf.Series(
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
        cudf.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [10, 22, 33],
                "c": [0.3234, 0.23432, 0.0],
                "d": ["hello", "world", "hello"],
            }
        ),
        cudf.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["hello", "world", "hello"],
                "c": [0.3234, 0.23432, 0.0],
            }
        ),
        pytest.param(
            cudf.DataFrame(
                {
                    "int_data": [1, 2, 3],
                    "str_data": ["hello", "world", "hello"],
                    "float_data": [0.3234, 0.23432, 0.0],
                    "timedelta_data": cudf.Series(
                        [1, 2, 1], dtype="timedelta64[ns]"
                    ),
                    "datetime_data": cudf.Series(
                        [1, 2, 1], dtype="datetime64[ns]"
                    ),
                }
            ),
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6219"
            ),
        ),
        pytest.param(
            cudf.DataFrame(
                {
                    "int_data": [1, 2, 3],
                    "str_data": ["hello", "world", "hello"],
                    "float_data": [0.3234, 0.23432, 0.0],
                    "timedelta_data": cudf.Series(
                        [1, 2, 1], dtype="timedelta64[ns]"
                    ),
                    "datetime_data": cudf.Series(
                        [1, 2, 1], dtype="datetime64[ns]"
                    ),
                    "category_data": cudf.Series(
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
        cudf.DataFrame({"a": [1, 2, 3]}),
        cudf.DataFrame(
            {"a": [1, 2, 3], "b": ["a", "z", "c"]}, index=["a", "z", "x"]
        ),
        cudf.DataFrame(
            {
                "a": [1, 2, 3, None, 2, 1, None],
                "b": ["a", "z", "c", "a", "v", "z", "z"],
            }
        ),
        cudf.DataFrame({"a": [], "b": []}),
        cudf.DataFrame({"a": [None, None], "b": [None, None]}),
        cudf.DataFrame(
            {
                "a": ["hello", "world", "rapids", "ai", "nvidia"],
                "b": cudf.Series(
                    [1, 21, 21, 11, 11],
                    dtype="timedelta64[s]",
                    index=["a", "b", "c", "d", " e"],
                ),
            },
            index=["a", "b", "c", "d", " e"],
        ),
        cudf.DataFrame(
            {
                "a": ["hello", None, "world", "rapids", None, "ai", "nvidia"],
                "b": cudf.Series(
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


@pytest.mark.parametrize(
    "lhs, rhs", [("a", "a"), ("a", "b"), (1, 1.0), (None, None), (None, "a")]
)
def test_equals_names(lhs, rhs):
    lhs = cudf.DataFrame({lhs: [1, 2]})
    rhs = cudf.DataFrame({rhs: [1, 2]})

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)


def test_equals_dtypes():
    lhs = cudf.DataFrame({"a": [1, 2.0]})
    rhs = cudf.DataFrame({"a": [1, 2]})

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"a": [10, 11, 12]}, index=["a", "b", "z"]),
        pd.DataFrame({"z": ["a"]}),
        pd.DataFrame({"a": [], "b": []}),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame(),
        pd.DataFrame({"a": ["a", "a", "c", "z", "A"], "z": [1, 2, 3, 4, 5]}),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
)
def test_dataframe_error_equality(df1, df2, op):
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    assert_exceptions_equal(op, op, ([df1, df2],), ([gdf1, gdf2],))


@pytest.mark.parametrize(
    "df,expected_pdf",
    [
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series([1, 2, None, 3], dtype="uint8"),
                    "b": cudf.Series([23, None, None, 32], dtype="uint16"),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series([1, 2, None, 3], dtype=pd.UInt8Dtype()),
                    "b": pd.Series(
                        [23, None, None, 32], dtype=pd.UInt16Dtype()
                    ),
                }
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series([None, 123, None, 1], dtype="uint32"),
                    "b": cudf.Series(
                        [234, 2323, 23432, None, None, 224], dtype="uint64"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [None, 123, None, 1], dtype=pd.UInt32Dtype()
                    ),
                    "b": pd.Series(
                        [234, 2323, 23432, None, None, 224],
                        dtype=pd.UInt64Dtype(),
                    ),
                }
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [-10, 1, None, -1, None, 3], dtype="int8"
                    ),
                    "b": cudf.Series(
                        [111, None, 222, None, 13], dtype="int16"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [-10, 1, None, -1, None, 3], dtype=pd.Int8Dtype()
                    ),
                    "b": pd.Series(
                        [111, None, 222, None, 13], dtype=pd.Int16Dtype()
                    ),
                }
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [11, None, 22, 33, None, 2, None, 3], dtype="int32"
                    ),
                    "b": cudf.Series(
                        [32431, None, None, 32322, 0, 10, -32324, None],
                        dtype="int64",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [11, None, 22, 33, None, 2, None, 3],
                        dtype=pd.Int32Dtype(),
                    ),
                    "b": pd.Series(
                        [32431, None, None, 32322, 0, 10, -32324, None],
                        dtype=pd.Int64Dtype(),
                    ),
                }
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [True, None, False, None, False, True, True, False],
                        dtype="bool_",
                    ),
                    "b": cudf.Series(
                        [
                            "abc",
                            "a",
                            None,
                            "hello world",
                            "foo buzz",
                            "",
                            None,
                            "rapids ai",
                        ],
                        dtype="object",
                    ),
                    "c": cudf.Series(
                        [0.1, None, 0.2, None, 3, 4, 1000, None],
                        dtype="float64",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [True, None, False, None, False, True, True, False],
                        dtype=pd.BooleanDtype(),
                    ),
                    "b": pd.Series(
                        [
                            "abc",
                            "a",
                            None,
                            "hello world",
                            "foo buzz",
                            "",
                            None,
                            "rapids ai",
                        ],
                        dtype=pd.StringDtype(),
                    ),
                    "c": pd.Series(
                        [0.1, None, 0.2, None, 3, 4, 1000, None],
                        dtype=pd.Float64Dtype(),
                    ),
                }
            ),
        ),
    ],
)
def test_dataframe_to_pandas_nullable_dtypes(df, expected_pdf):
    actual_pdf = df.to_pandas(nullable=True)

    assert_eq(actual_pdf, expected_pdf)


@pytest.mark.parametrize(
    "data",
    [
        [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}],
        [{"a": 1, "b": 2, "c": None}, {"a": None, "b": 5, "c": 6}],
        [{"a": 1, "b": 2}, {"a": 1, "b": 5, "c": 6}],
        [{"a": 1, "b": 2}, {"b": 5, "c": 6}],
        [{}, {"a": 1, "b": 5, "c": 6}],
        [{"a": 1, "b": 2, "c": 3}, {"a": 4.5, "b": 5.5, "c": 6.5}],
    ],
)
def test_dataframe_init_from_list_of_dicts(data):
    expect = pd.DataFrame(data)
    got = cudf.DataFrame(data)

    assert_eq(expect, got)


def test_dataframe_pipe():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    def add_int_col(df, column):
        df[column] = df._constructor_sliced([10, 20, 30, 40])
        return df

    def add_str_col(df, column):
        df[column] = df._constructor_sliced(["a", "b", "xyz", "ai"])
        return df

    expected = (
        pdf.pipe(add_int_col, "one")
        .pipe(add_int_col, column="two")
        .pipe(add_str_col, "three")
    )
    actual = (
        gdf.pipe(add_int_col, "one")
        .pipe(add_int_col, column="two")
        .pipe(add_str_col, "three")
    )

    assert_eq(expected, actual)

    expected = (
        pdf.pipe((add_str_col, "df"), column="one")
        .pipe(add_str_col, column="two")
        .pipe(add_int_col, "three")
    )
    actual = (
        gdf.pipe((add_str_col, "df"), column="one")
        .pipe(add_str_col, column="two")
        .pipe(add_int_col, "three")
    )

    assert_eq(expected, actual)


def test_dataframe_pipe_error():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    def custom_func(df, column):
        df[column] = df._constructor_sliced([10, 20, 30, 40])
        return df

    assert_exceptions_equal(
        lfunc=pdf.pipe,
        rfunc=gdf.pipe,
        lfunc_args_and_kwargs=([(custom_func, "columns")], {"columns": "d"}),
        rfunc_args_and_kwargs=([(custom_func, "columns")], {"columns": "d"}),
    )


@pytest.mark.parametrize(
    "op",
    ["count", "kurt", "kurtosis", "skew"],
)
def test_dataframe_axis1_unsupported_ops(op):
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [8, 9, 10]})

    with pytest.raises(
        NotImplementedError, match="Only axis=0 is currently supported."
    ):
        getattr(df, op)(axis=1)


def test_dataframe_from_pandas_duplicate_columns():
    pdf = pd.DataFrame(columns=["a", "b", "c", "a"])
    pdf["a"] = [1, 2, 3]

    with pytest.raises(
        ValueError, match="Duplicate column names are not allowed"
    ):
        cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {"a": [1, 2, 3], "b": [10, 11, 20], "c": ["a", "bcd", "xyz"]}
        ),
        pd.DataFrame(),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        None,
        ["a"],
        ["c", "a"],
        ["b", "a", "c"],
        [],
        pd.Index(["c", "a"]),
        cudf.Index(["c", "a"]),
        ["abc", "a"],
        ["column_not_exists1", "column_not_exists2"],
    ],
)
@pytest.mark.parametrize("index", [["abc", "def", "ghi"]])
def test_dataframe_constructor_columns(df, columns, index):
    def assert_local_eq(actual, df, expected, host_columns):
        check_index_type = not expected.empty
        if host_columns is not None and any(
            col not in df.columns for col in host_columns
        ):
            assert_eq(
                expected,
                actual,
                check_dtype=False,
                check_index_type=check_index_type,
            )
        else:
            assert_eq(expected, actual, check_index_type=check_index_type)

    gdf = cudf.from_pandas(df)
    host_columns = (
        columns.to_pandas() if isinstance(columns, cudf.BaseIndex) else columns
    )

    expected = pd.DataFrame(df, columns=host_columns, index=index)
    actual = cudf.DataFrame(gdf, columns=columns, index=index)

    assert_local_eq(actual, df, expected, host_columns)


def test_dataframe_constructor_column_index_only():
    columns = ["a", "b", "c"]
    index = ["r1", "r2", "r3"]

    gdf = cudf.DataFrame(index=index, columns=columns)
    assert not id(gdf["a"]._column) == id(gdf["b"]._column) and not id(
        gdf["b"]._column
    ) == id(gdf["c"]._column)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [3.0, 4.0, 5.0], "c": [True, True, False]},
        {"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0], "c": [True, True, False]},
        {"a": [1, 2, 3], "b": [3, 4, 5], "c": [True, True, False]},
        {"a": [1, 2, 3], "b": [True, True, False], "c": [False, True, False]},
        {
            "a": [1.0, 2.0, 3.0],
            "b": [True, True, False],
            "c": [False, True, False],
        },
        {"a": [1, 2, 3], "b": [3, 4, 5], "c": [2.0, 3.0, 4.0]},
        {"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": [5.0, 6.0, 4.0]},
    ],
)
@pytest.mark.parametrize(
    "aggs",
    [
        ["min", "sum", "max"],
        ("min", "sum", "max"),
        {"min", "sum", "max"},
        "sum",
        {"a": "sum", "b": "min", "c": "max"},
        {"a": ["sum"], "b": ["min"], "c": ["max"]},
        {"a": ("sum"), "b": ("min"), "c": ("max")},
        {"a": {"sum"}, "b": {"min"}, "c": {"max"}},
        {"a": ["sum", "min"], "b": ["sum", "max"], "c": ["min", "max"]},
        {"a": ("sum", "min"), "b": ("sum", "max"), "c": ("min", "max")},
        {"a": {"sum", "min"}, "b": {"sum", "max"}, "c": {"min", "max"}},
    ],
)
def test_agg_for_dataframes(data, aggs):
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    expect = pdf.agg(aggs).sort_index()
    got = gdf.agg(aggs).sort_index()
    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("aggs", [{"a": np.sum, "b": np.min, "c": np.max}])
def test_agg_for_unsupported_function(aggs):
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3], "b": [3.0, 4.0, 5.0], "c": [True, True, False]}
    )

    with pytest.raises(NotImplementedError):
        gdf.agg(aggs)


@pytest.mark.parametrize("aggs", ["asdf"])
def test_agg_for_dataframe_with_invalid_function(aggs):
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3], "b": [3.0, 4.0, 5.0], "c": [True, True, False]}
    )

    with pytest.raises(
        AttributeError,
        match=f"{aggs} is not a valid function for 'DataFrame' object",
    ):
        gdf.agg(aggs)


@pytest.mark.parametrize("aggs", [{"a": "asdf"}])
def test_agg_for_series_with_invalid_function(aggs):
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3], "b": [3.0, 4.0, 5.0], "c": [True, True, False]}
    )

    with pytest.raises(
        AttributeError,
        match=f"{aggs['a']} is not a valid function for 'Series' object",
    ):
        gdf.agg(aggs)


@pytest.mark.parametrize(
    "aggs",
    [
        "sum",
        ["min", "sum", "max"],
        {"a": {"sum", "min"}, "b": {"sum", "max"}, "c": {"min", "max"}},
    ],
)
def test_agg_for_dataframe_with_string_columns(aggs):
    gdf = cudf.DataFrame(
        {"a": ["m", "n", "o"], "b": ["t", "u", "v"], "c": ["x", "y", "z"]},
        index=["a", "b", "c"],
    )

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "DataFrame.agg() is not supported for "
            "frames containing string columns"
        ),
    ):
        gdf.agg(aggs)


@pytest.mark.parametrize(
    "join",
    ["left"],
)
@pytest.mark.parametrize(
    "overwrite",
    [True, False],
)
@pytest.mark.parametrize(
    "errors",
    ["ignore"],
)
@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [3, 4, 5]},
        {"e": [1.0, 2.0, 3.0], "d": [3.0, 4.0, 5.0]},
        {"c": [True, False, False], "d": [False, True, True]},
        {"g": [2.0, np.nan, 4.0], "n": [np.nan, np.nan, np.nan]},
        {"d": [np.nan, np.nan, np.nan], "e": [np.nan, np.nan, np.nan]},
        {"a": [1.0, 2, 3], "b": pd.Series([4.0, 8.0, 3.0], index=[1, 2, 3])},
        {
            "d": [1.0, 2.0, 3.0],
            "c": pd.Series([np.nan, np.nan, np.nan], index=[1, 2, 3]),
        },
        {
            "a": [False, True, False],
            "b": pd.Series([1.0, 2.0, np.nan], index=[1, 2, 3]),
        },
        {
            "a": [np.nan, np.nan, np.nan],
            "e": pd.Series([np.nan, np.nan, np.nan], index=[1, 2, 3]),
        },
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        {"b": [3, 5, 6], "e": [8, 2, 1]},
        {"c": [True, False, True], "d": [3.0, 4.0, 5.0]},
        {"e": [False, False, True], "g": [True, True, False]},
        {"g": [np.nan, np.nan, np.nan], "c": [np.nan, np.nan, np.nan]},
        {"a": [7, 5, 8], "b": pd.Series([2.0, 7.0, 9.0], index=[0, 1, 2])},
        {
            "b": [np.nan, 2.0, np.nan],
            "c": pd.Series([2, np.nan, 5.0], index=[2, 3, 4]),
        },
        {
            "a": [True, np.nan, True],
            "d": pd.Series([False, True, np.nan], index=[0, 1, 3]),
        },
    ],
)
def test_update_for_dataframes(data, data2, join, overwrite, errors):
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data, nan_as_null=False)

    other_pd = pd.DataFrame(data2)
    other_gd = cudf.DataFrame(data2, nan_as_null=False)

    pdf.update(other=other_pd, join=join, overwrite=overwrite, errors=errors)
    gdf.update(other=other_gd, join=join, overwrite=overwrite, errors=errors)

    assert_eq(pdf, gdf, check_dtype=False)


@pytest.mark.parametrize(
    "join",
    ["right"],
)
def test_update_for_right_join(join):
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})
    other_gd = cudf.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})

    with pytest.raises(
        NotImplementedError, match="Only left join is supported"
    ):
        gdf.update(other_gd, join)


@pytest.mark.parametrize(
    "errors",
    ["raise"],
)
def test_update_for_data_overlap(errors):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    other_pd = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})
    other_gd = cudf.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})

    assert_exceptions_equal(
        lfunc=pdf.update,
        rfunc=gdf.update,
        lfunc_args_and_kwargs=([other_pd, errors], {}),
        rfunc_args_and_kwargs=([other_gd, errors], {}),
    )


@pytest.mark.parametrize(
    "gdf",
    [
        cudf.DataFrame({"a": [[1], [2], [3]]}),
        cudf.DataFrame(
            {
                "left-a": [0, 1, 2],
                "a": [[1], None, [3]],
                "right-a": ["abc", "def", "ghi"],
            }
        ),
        cudf.DataFrame(
            {
                "left-a": [[], None, None],
                "a": [[1], None, [3]],
                "right-a": ["abc", "def", "ghi"],
            }
        ),
    ],
)
def test_dataframe_roundtrip_arrow_list_dtype(gdf):
    table = gdf.to_arrow()
    expected = cudf.DataFrame.from_arrow(table)

    assert_eq(gdf, expected)


@pytest.mark.parametrize(
    "gdf",
    [
        cudf.DataFrame({"a": [{"one": 3, "two": 4, "three": 10}]}),
        cudf.DataFrame(
            {
                "left-a": [0, 1, 2],
                "a": [{"x": 0.23, "y": 43}, None, {"x": 23.9, "y": 4.3}],
                "right-a": ["abc", "def", "ghi"],
            }
        ),
        cudf.DataFrame(
            {
                "left-a": [{"a": 1}, None, None],
                "a": [
                    {"one": 324, "two": 23432, "three": 324},
                    None,
                    {"one": 3.24, "two": 1, "three": 324},
                ],
                "right-a": ["abc", "def", "ghi"],
            }
        ),
    ],
)
def test_dataframe_roundtrip_arrow_struct_dtype(gdf):
    table = gdf.to_arrow()
    expected = cudf.DataFrame.from_arrow(table)

    assert_eq(gdf, expected)


def test_dataframe_setitem_cupy_array():
    np.random.seed(0)
    pdf = pd.DataFrame(np.random.randn(10, 2))
    gdf = cudf.from_pandas(pdf)

    gpu_array = cupy.array([True, False] * 5)
    pdf[gpu_array.get()] = 1.5
    gdf[gpu_array] = 1.5

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data", [{"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}]
)
@pytest.mark.parametrize(
    "index",
    [{0: 123, 1: 4, 2: 6}],
)
@pytest.mark.parametrize(
    "level",
    ["x", 0],
)
def test_rename_for_level_MultiIndex_dataframe(data, index, level):
    pdf = pd.DataFrame(
        data,
        index=pd.MultiIndex.from_tuples([(0, 1, 2), (1, 2, 3), (2, 3, 4)]),
    )
    pdf.index.names = ["x", "y", "z"]
    gdf = cudf.from_pandas(pdf)

    expect = pdf.rename(index=index, level=level)
    got = gdf.rename(index=index, level=level)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data", [{"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}]
)
@pytest.mark.parametrize(
    "columns",
    [{"a": "f", "b": "g"}, {1: 3, 2: 4}, lambda s: 2 * s],
)
@pytest.mark.parametrize(
    "level",
    [0, 1],
)
def test_rename_for_level_MultiColumn_dataframe(data, columns, level):
    gdf = cudf.DataFrame(data)
    gdf.columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])

    pdf = gdf.to_pandas()

    expect = pdf.rename(columns=columns, level=level)
    got = gdf.rename(columns=columns, level=level)

    assert_eq(expect, got)


def test_rename_for_level_RangeIndex_dataframe():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    pdf = gdf.to_pandas()

    expect = pdf.rename(columns={"a": "f"}, index={0: 3, 1: 4}, level=0)
    got = gdf.rename(columns={"a": "f"}, index={0: 3, 1: 4}, level=0)

    assert_eq(expect, got)


@pytest.mark.xfail(reason="level=None not implemented yet")
def test_rename_for_level_is_None_MC():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf.columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
    pdf = gdf.to_pandas()

    expect = pdf.rename(columns={"a": "f"}, level=None)
    got = gdf.rename(columns={"a": "f"}, level=None)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [
            [[1, 2, 3], 11, "a"],
            [None, 22, "e"],
            [[4], 33, "i"],
            [[], 44, "o"],
            [[5, 6], 55, "u"],
        ],  # nested
        [
            [1, 11, "a"],
            [2, 22, "e"],
            [3, 33, "i"],
            [4, 44, "o"],
            [5, 55, "u"],
        ],  # non-nested
    ],
)
@pytest.mark.parametrize(
    ("labels", "label_to_explode"),
    [
        (None, 0),
        (pd.Index(["a", "b", "c"]), "a"),
        (
            pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (1, "a")], names=["l0", "l1"]
            ),
            (0, "a"),
        ),
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize(
    "p_index",
    [
        None,
        ["ia", "ib", "ic", "id", "ie"],
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b")]
        ),
    ],
)
def test_explode(data, labels, ignore_index, p_index, label_to_explode):
    pdf = pd.DataFrame(data, index=p_index, columns=labels)
    gdf = cudf.from_pandas(pdf)

    if PANDAS_GE_134:
        expect = pdf.explode(label_to_explode, ignore_index)
    else:
        # https://github.com/pandas-dev/pandas/issues/43314
        if isinstance(label_to_explode, int):
            pdlabel_to_explode = [label_to_explode]
        else:
            pdlabel_to_explode = label_to_explode
        expect = pdf.explode(pdlabel_to_explode, ignore_index)

    got = gdf.explode(label_to_explode, ignore_index)

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "df,ascending,expected",
    [
        (
            cudf.DataFrame({"a": [10, 0, 2], "b": [-10, 10, 1]}),
            True,
            cupy.array([1, 2, 0], dtype="int32"),
        ),
        (
            cudf.DataFrame({"a": [10, 0, 2], "b": [-10, 10, 1]}),
            False,
            cupy.array([0, 2, 1], dtype="int32"),
        ),
    ],
)
def test_dataframe_argsort(df, ascending, expected):
    actual = df.argsort(ascending=ascending)

    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data,columns,index",
    [
        (pd.Series([1, 2, 3]), None, None),
        (pd.Series(["a", "b", None, "c"], name="abc"), None, None),
        (
            pd.Series(["a", "b", None, "c"], name="abc"),
            ["abc", "b"],
            [1, 2, 3],
        ),
    ],
)
def test_dataframe_init_from_series(data, columns, index):
    expected = pd.DataFrame(data, columns=columns, index=index)
    actual = cudf.DataFrame(data, columns=columns, index=index)

    assert_eq(
        expected,
        actual,
        check_index_type=len(expected) != 0,
    )


def test_frame_series_where():
    gdf = cudf.DataFrame(
        {"a": [1.0, 2.0, None, 3.0, None], "b": [None, 10.0, 11.0, None, 23.0]}
    )
    pdf = gdf.to_pandas()
    expected = gdf.where(gdf.notna(), gdf.mean())
    actual = pdf.where(pdf.notna(), pdf.mean(), axis=1)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [{"a": [1, 2, 3], "b": [1, 1, 0]}],
)
def test_frame_series_where_other(data):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = gdf.where(gdf["b"] == 1, cudf.NA)
    actual = pdf.where(pdf["b"] == 1, pd.NA)
    assert_eq(
        actual.fillna(-1).values,
        expected.fillna(-1).values,
        check_dtype=False,
    )

    expected = gdf.where(gdf["b"] == 1, 0)
    actual = pdf.where(pdf["b"] == 1, 0)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data, gkey",
    [
        (
            {
                "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
                "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
                "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
            },
            ["id", "val1", "val2"],
        ),
        (
            {
                "id": [0] * 4 + [1] * 3,
                "a": [10, 3, 4, 2, -3, 9, 10],
                "b": [10, 23, -4, 2, -3, 9, 19],
            },
            ["id", "a"],
        ),
        (
            {
                "id": ["a", "a", "b", "b", "c", "c"],
                "val": cudf.Series(
                    [None, None, None, None, None, None], dtype="float64"
                ),
            },
            ["id"],
        ),
        (
            {
                "id": ["a", "a", "b", "b", "c", "c"],
                "val1": [None, 4, 6, 8, None, 2],
                "val2": [4, 5, None, 2, 9, None],
            },
            ["id"],
        ),
        ({"id": [1.0], "val1": [2.0], "val2": [3.0]}, ["id"]),
    ],
)
@pytest.mark.parametrize(
    "min_per",
    [0, 1, 2, 3, 4],
)
def test_pearson_corr_passing(data, gkey, min_per):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.groupby(gkey).corr(method="pearson", min_periods=min_per)
    expected = pdf.groupby(gkey).corr(method="pearson", min_periods=min_per)

    assert_eq(expected, actual)


@pytest.mark.parametrize("method", ["kendall", "spearman"])
def test_pearson_corr_unsupported_methods(method):
    gdf = cudf.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
            "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
            "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        }
    )

    with pytest.raises(
        NotImplementedError,
        match="Only pearson correlation is currently supported",
    ):
        gdf.groupby("id").corr(method)


def test_pearson_corr_empty_columns():
    gdf = cudf.DataFrame(columns=["id", "val1", "val2"])
    pdf = gdf.to_pandas()

    actual = gdf.groupby("id").corr("pearson")
    expected = pdf.groupby("id").corr("pearson")

    assert_eq(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "data",
    [
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": ["v", "n", "k", "l", "m", "i", "y", "r", "w"],
            "val2": ["d", "d", "d", "e", "e", "e", "f", "f", "f"],
        },
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "val2": ["d", "d", "d", "e", "e", "e", "f", "f", "f"],
        },
    ],
)
@pytest.mark.parametrize("gkey", ["id", "val1", "val2"])
def test_pearson_corr_invalid_column_types(data, gkey):
    with pytest.raises(
        TypeError,
        match="Correlation accepts only numerical column-pairs",
    ):
        cudf.DataFrame(data).groupby(gkey).corr("pearson")


def test_pearson_corr_multiindex_dataframe():
    gdf = cudf.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 1, 2, 3], "c": [2, 3, 4, 5]}
    ).set_index(["a", "b"])

    actual = gdf.groupby(level="a").corr("pearson")
    expected = gdf.to_pandas().groupby(level="a").corr("pearson")

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [np.nan, 1, 2], "b": [None, None, None]},
        {"a": [1, 2, np.nan, 2], "b": [np.nan, np.nan, np.nan, np.nan]},
        {
            "a": [1, 2, np.nan, 2, None],
            "b": [np.nan, np.nan, None, np.nan, np.nan],
        },
        {"a": [1, 2, 2, None, 1.1], "b": [1, 2.2, 3, None, 5]},
    ],
)
@pytest.mark.parametrize("nan_as_null", [True, False])
def test_dataframe_constructor_nan_as_null(data, nan_as_null):
    actual = cudf.DataFrame(data, nan_as_null=nan_as_null)

    if nan_as_null:
        assert (
            not (
                actual.astype("float").replace(
                    cudf.Series([np.nan], nan_as_null=False), cudf.Series([-1])
                )
                == -1
            )
            .any()
            .any()
        )
    else:
        actual = actual.select_dtypes(exclude=["object"])
        assert (actual.replace(np.nan, -1) == -1).any().any()


def test_dataframe_add_prefix():
    cdf = cudf.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    pdf = cdf.to_pandas()

    got = cdf.add_prefix("item_")
    expected = pdf.add_prefix("item_")

    assert_eq(got, expected)


def test_dataframe_add_suffix():
    cdf = cudf.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    pdf = cdf.to_pandas()

    got = cdf.add_suffix("_item")
    expected = pdf.add_suffix("_item")

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data, gkey",
    [
        (
            {
                "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
                "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
                "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
            },
            ["id"],
        ),
        (
            {
                "id": [0, 0, 0, 0, 1, 1, 1],
                "a": [10.0, 3, 4, 2.0, -3.0, 9.0, 10.0],
                "b": [10.0, 23, -4.0, 2, -3.0, 9, 19.0],
            },
            ["id", "a"],
        ),
    ],
)
@pytest.mark.parametrize(
    "min_periods",
    [0, 3],
)
@pytest.mark.parametrize(
    "ddof",
    [1, 2],
)
def test_groupby_covariance(data, gkey, min_periods, ddof):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.groupby(gkey).cov(min_periods=min_periods, ddof=ddof)
    expected = pdf.groupby(gkey).cov(min_periods=min_periods, ddof=ddof)

    assert_eq(expected, actual)


def test_groupby_covariance_multiindex_dataframe():
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 1, 2, 3],
            "c": [2, 3, 4, 5],
            "d": [6, 8, 9, 1],
        }
    ).set_index(["a", "b"])

    actual = gdf.groupby(level=["a", "b"]).cov()
    expected = gdf.to_pandas().groupby(level=["a", "b"]).cov()

    assert_eq(expected, actual)


def test_groupby_covariance_empty_columns():
    gdf = cudf.DataFrame(columns=["id", "val1", "val2"])
    pdf = gdf.to_pandas()

    actual = gdf.groupby("id").cov()
    expected = pdf.groupby("id").cov()

    assert_eq(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False,
    )


def test_groupby_cov_invalid_column_types():
    gdf = cudf.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": ["v", "n", "k", "l", "m", "i", "y", "r", "w"],
            "val2": ["d", "d", "d", "e", "e", "e", "f", "f", "f"],
        },
    )
    with pytest.raises(
        TypeError,
        match="Covariance accepts only numerical column-pairs",
    ):
        gdf.groupby("id").cov()


def test_groupby_cov_positive_semidefinite_matrix():
    # Refer to discussions in PR #9889 re "pair-wise deletion" strategy
    # being used in pandas to compute the covariance of a dataframe with
    # rows containing missing values.
    # Note: cuDF currently matches pandas behavior in that the covariance
    # matrices are not guaranteed PSD (positive semi definite).
    # https://github.com/rapidsai/cudf/pull/9889#discussion_r794158358
    gdf = cudf.DataFrame(
        [[1, 2], [None, 4], [5, None], [7, 8]], columns=["v0", "v1"]
    )
    actual = gdf.groupby(by=cudf.Series([1, 1, 1, 1])).cov()
    actual.reset_index(drop=True, inplace=True)

    pdf = gdf.to_pandas()
    expected = pdf.groupby(by=pd.Series([1, 1, 1, 1])).cov()
    expected.reset_index(drop=True, inplace=True)

    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


@pytest.mark.xfail
def test_groupby_cov_for_pandas_bug_case():
    # Handles case: pandas bug using ddof with missing data.
    # Filed an issue in Pandas on GH, link below:
    # https://github.com/pandas-dev/pandas/issues/45814
    pdf = pd.DataFrame(
        {"id": ["a", "a"], "val1": [1.0, 2.0], "val2": [np.nan, np.nan]}
    )
    expected = pdf.groupby("id").cov(ddof=2)

    gdf = cudf.from_pandas(pdf)
    actual = gdf.groupby("id").cov(ddof=2)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        np.random.RandomState(seed=10).randint(-50, 50, (25, 30)),
        np.random.RandomState(seed=10).random_sample((4, 4)),
        np.array([1.123, 2.343, 5.890, 0.0]),
        [True, False, True, False, False],
        {"a": [1.123, 2.343, np.nan, np.nan], "b": [None, 3, 9.08, None]},
    ],
)
@pytest.mark.parametrize("periods", (-5, -1, 0, 1, 5))
def test_diff_numeric_dtypes(data, periods):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.diff(periods=periods, axis=0)
    expected = pdf.diff(periods=periods, axis=0)

    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("precision", "scale"),
    [(5, 2), (8, 5)],
)
@pytest.mark.parametrize(
    "dtype",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype],
)
def test_diff_decimal_dtypes(precision, scale, dtype):
    gdf = cudf.DataFrame(
        np.random.default_rng(seed=42).uniform(10.5, 75.5, (10, 6)),
        dtype=dtype(precision=precision, scale=scale),
    )
    pdf = gdf.to_pandas()

    actual = gdf.diff()
    expected = pdf.diff()

    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


def test_diff_invalid_axis():
    gdf = cudf.DataFrame(np.array([1.123, 2.343, 5.890, 0.0]))
    with pytest.raises(NotImplementedError, match="Only axis=0 is supported."):
        gdf.diff(periods=1, axis=1)


@pytest.mark.parametrize(
    "data",
    [
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "string_col": ["a", "b", "c", "d", "e"],
        },
        ["a", "b", "c", "d", "e"],
    ],
)
def test_diff_unsupported_dtypes(data):
    gdf = cudf.DataFrame(data)
    with pytest.raises(
        TypeError,
        match=r"unsupported operand type\(s\)",
    ):
        gdf.diff()


def test_diff_many_dtypes():
    pdf = pd.DataFrame(
        {
            "dates": pd.date_range("2020-01-01", "2020-01-06", freq="D"),
            "bools": [True, True, True, False, True, True],
            "floats": [1.0, 2.0, 3.5, np.nan, 5.0, -1.7],
            "ints": [1, 2, 3, 3, 4, 5],
            "nans_nulls": [np.nan, None, None, np.nan, np.nan, None],
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.diff(), gdf.diff())
    assert_eq(pdf.diff(periods=2), gdf.diff(periods=2))


def test_dataframe_assign_cp_np_array():
    m, n = 5, 3
    cp_ndarray = cupy.random.randn(m, n)
    pdf = pd.DataFrame({f"f_{i}": range(m) for i in range(n)})
    gdf = cudf.DataFrame({f"f_{i}": range(m) for i in range(n)})
    pdf[[f"f_{i}" for i in range(n)]] = cupy.asnumpy(cp_ndarray)
    gdf[[f"f_{i}" for i in range(n)]] = cp_ndarray

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [{"a": [1, 2, 3], "b": [1, 1, 0]}],
)
def test_dataframe_nunique(data):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.nunique()
    expected = pdf.nunique()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [{"key": [0, 1, 1, 0, 0, 1], "val": [1, 8, 3, 9, -3, 8]}],
)
def test_dataframe_nunique_index(data):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.index.nunique()
    expected = pdf.index.nunique()

    assert_eq(expected, actual)


def test_dataframe_rename_duplicate_column():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    with pytest.raises(
        ValueError, match="Duplicate column names are not allowed"
    ):
        gdf.rename(columns={"a": "b"}, inplace=True)


@pytest.mark.parametrize(
    "data",
    [
        np.random.RandomState(seed=10).randint(-50, 50, (10, 10)),
        np.random.RandomState(seed=10).random_sample((4, 4)),
        np.array([1.123, 2.343, 5.890, 0.0]),
        {"a": [1.123, 2.343, np.nan, np.nan], "b": [None, 3, 9.08, None]},
    ],
)
@pytest.mark.parametrize("periods", [-5, -2, 0, 2, 5])
@pytest.mark.parametrize("fill_method", ["ffill", "bfill", "pad", "backfill"])
def test_dataframe_pct_change(data, periods, fill_method):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.pct_change(periods=periods, fill_method=fill_method)
    expected = pdf.pct_change(periods=periods, fill_method=fill_method)

    assert_eq(expected, actual)


def test_mean_timeseries():
    gdf = cudf.datasets.timeseries()
    pdf = gdf.to_pandas()

    expected = pdf.mean(numeric_only=True)
    actual = gdf.mean(numeric_only=True)

    assert_eq(expected, actual)

    with pytest.raises(TypeError):
        gdf.mean()


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
def test_std_different_dtypes(data):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = pdf.std(numeric_only=True)
    actual = gdf.std(numeric_only=True)

    assert_eq(expected, actual)

    with pytest.raises(TypeError):
        gdf.std()


@pytest.mark.parametrize(
    "data",
    [
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": ["v", "n", "k", "l", "m", "i", "y", "r", "w"],
            "val2": ["d", "d", "d", "e", "e", "e", "f", "f", "f"],
        }
    ],
)
def test_empty_numeric_only(data):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()
    expected = pdf.prod(numeric_only=True)
    actual = gdf.prod(numeric_only=True)
    assert_eq(expected, actual)


@pytest.fixture
def df_eval():
    N = 10
    int_max = 10
    rng = cupy.random.default_rng(0)
    return cudf.DataFrame(
        {
            "a": rng.integers(N, size=int_max),
            "b": rng.integers(N, size=int_max),
            "c": rng.integers(N, size=int_max),
            "d": rng.integers(N, size=int_max),
        }
    )


# Note that for now expressions do not automatically handle casting, so inputs
# need to be casted appropriately
@pytest.mark.parametrize(
    "expr, dtype",
    [
        ("a", int),
        ("+a", int),
        ("a + b", int),
        ("a == b", int),
        ("a / b", float),
        ("a * b", int),
        ("a > b", int),
        ("a > b > c", int),
        ("a > b < c", int),
        ("a & b", int),
        ("a & b | c", int),
        ("sin(a)", float),
        ("exp(sin(abs(a)))", float),
        ("sqrt(floor(a))", float),
        ("ceil(arctanh(a))", float),
        ("(a + b) - (c * d)", int),
        ("~a", int),
        ("(a > b) and (c > d)", int),
        ("(a > b) or (c > d)", int),
        ("not (a > b)", int),
        ("a + 1", int),
        ("a + 1.0", float),
        ("-a + 1", int),
        ("+a + 1", int),
        ("e = a + 1", int),
        (
            """
            e = log(cos(a)) + 1.0
            f = abs(c) - exp(d)
            """,
            float,
        ),
        ("a_b_are_equal = (a == b)", int),
    ],
)
def test_dataframe_eval(df_eval, expr, dtype):
    df_eval = df_eval.astype(dtype)
    expect = df_eval.to_pandas().eval(expr)
    got = df_eval.eval(expr)
    # In the specific case where the evaluated expression is a unary function
    # of a single column with no nesting, pandas will retain the name. This
    # level of compatibility is out of scope for now.
    assert_eq(expect, got, check_names=False)

    # Test inplace
    if re.search("[^=]=[^=]", expr) is not None:
        pdf_eval = df_eval.to_pandas()
        pdf_eval.eval(expr, inplace=True)
        df_eval.eval(expr, inplace=True)
        assert_eq(pdf_eval, df_eval)


@pytest.mark.parametrize(
    "expr",
    [
        """
        e = a + b
        a == b
        """,
        "a_b_are_equal = (a == b) = c",
    ],
)
def test_dataframe_eval_errors(df_eval, expr):
    with pytest.raises(ValueError):
        df_eval.eval(expr)


@pytest.mark.parametrize(
    "gdf,subset",
    [
        (
            cudf.DataFrame(
                {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
                index=["falcon", "dog", "cat", "ant"],
            ),
            ["num_legs"],
        ),
        (
            cudf.DataFrame(
                {
                    "first_name": ["John", "Anne", "John", "Beth"],
                    "middle_name": ["Smith", None, None, "Louise"],
                }
            ),
            ["first_name"],
        ),
    ],
)
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("use_subset", [True, False])
def test_value_counts(
    gdf,
    subset,
    sort,
    ascending,
    normalize,
    dropna,
    use_subset,
):
    pdf = gdf.to_pandas()

    got = gdf.value_counts(
        subset=subset if (use_subset) else None,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        dropna=dropna,
    )
    expected = pdf.value_counts(
        subset=subset if (use_subset) else None,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        dropna=dropna,
    )

    if not dropna:
        # Convert the Pandas series to a cuDF one due to difference
        # in the handling of NaNs between the two (<NA> in cuDF and
        # NaN in Pandas) when dropna=False.
        assert_eq(got.sort_index(), cudf.from_pandas(expected).sort_index())
    else:
        assert_eq(got.sort_index(), expected.sort_index())

    with pytest.raises(KeyError):
        gdf.value_counts(subset=["not_a_column_name"])


@pytest.fixture
def wildcard_df():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    return df


def test_multiindex_wildcard_selection_all(wildcard_df):
    expect = wildcard_df.to_pandas().loc[:, (slice(None), "b")]
    got = wildcard_df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


@pytest.mark.xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_partial(wildcard_df):
    expect = wildcard_df.to_pandas().loc[:, (slice("a", "b"), "b")]
    got = wildcard_df.loc[:, (slice("a", "b"), "b")]
    assert_eq(expect, got)


@pytest.mark.xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_three_level_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2, c3) for c1 in "abcd" for c2 in "abc" for c3 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(24)})
    df.columns = midx

    expect = df.to_pandas().loc[:, (slice("a", "c"), slice("a", "b"), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


def test_dataframe_assign_scalar_to_empty_series():
    expected = pd.DataFrame({"a": []})
    actual = cudf.DataFrame({"a": []})
    expected.a = 0
    actual.a = 0
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        {0: [1, 2, 3], 2: [10, 11, 23]},
        {("a", "b"): [1, 2, 3], ("2",): [10, 11, 23]},
    ],
)
def test_non_string_column_name_to_arrow(data):
    df = cudf.DataFrame(data)

    expected = df.to_arrow()
    actual = pa.Table.from_pandas(df.to_pandas())

    assert expected.equals(actual)


def test_complex_types_from_arrow():

    expected = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3]),
            pa.array([10, 20, 30]),
            pa.array([{"a": 9}, {"b": 10}, {"c": 11}]),
            pa.array([[{"a": 1}], [{"b": 2}], [{"c": 3}]]),
            pa.array([10, 11, 12]).cast(pa.decimal128(21, 2)),
            pa.array([{"a": 9}, {"b": 10, "c": {"g": 43}}, {"c": {"a": 10}}]),
        ],
        names=["a", "b", "c", "d", "e", "f"],
    )

    df = cudf.DataFrame.from_arrow(expected)
    actual = df.to_arrow()

    assert expected.equals(actual)
