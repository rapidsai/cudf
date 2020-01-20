# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import array as arr
import operator

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import rmm

import cudf as gd
from cudf.core.dataframe import DataFrame, Series
from cudf.tests import utils
from cudf.tests.utils import assert_eq, does_not_raise, gen_rand


def test_init_via_list_of_tuples():
    data = [
        (5, "cats", "jump", np.nan),
        (2, "dogs", "dig", 7.5),
        (3, "cows", "moo", -2.1, "occasionally"),
    ]

    pdf = pd.DataFrame(data)
    gdf = DataFrame(data)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("rows", [0, 1, 2, 100])
def test_init_via_list_of_empty_tuples(rows):
    data = [()] * rows

    pdf = pd.DataFrame(data)
    gdf = DataFrame(data)

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
    series = Series(a1)
    assert len(series) == 10
    np.testing.assert_equal(series.to_array(), np.hstack([a1]))


@pytest.mark.parametrize("a", [[1, 2, 3], [1, 10, 30]])
@pytest.mark.parametrize("b", [[4, 5, 6], [-11, -100, 30]])
def test_append_index(a, b):

    df = pd.DataFrame()
    df["a"] = a
    df["b"] = b

    gdf = DataFrame()
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


def test_series_append():
    a1 = np.arange(10, dtype=np.float64)
    series = Series(a1)
    # Add new buffer
    a2 = np.arange(5)
    series = series.append(a2)
    assert len(series) == 15
    np.testing.assert_equal(series.to_array(), np.hstack([a1, a2]))

    # Ensure appending to previous buffer
    a3 = np.arange(3)
    series = series.append(a3)
    assert len(series) == 18
    a4 = np.hstack([a1, a2, a3])
    np.testing.assert_equal(series.to_array(), a4)

    # Appending different dtype
    a5 = np.array([1, 2, 3], dtype=np.int32)
    a6 = np.array([4.5, 5.5, 6.5], dtype=np.float64)
    series = Series(a5).append(a6)
    np.testing.assert_equal(series.to_array(), np.hstack([a5, a6]))
    series = Series(a6).append(a5)
    np.testing.assert_equal(series.to_array(), np.hstack([a6, a5]))


def test_series_init_none():

    # test for creating empty series
    # 1: without initializing
    sr1 = Series()
    got = sr1.to_string()
    print(got)
    expect = "Series([], dtype: float64)"
    # values should match despite whitespace difference
    assert got.split() == expect.split()

    # 2: Using `None` as a initializer
    sr2 = Series(None)
    got = sr2.to_string()
    print(got)
    expect = "Series([], dtype: float64)"
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_basic():
    np.random.seed(0)
    df = DataFrame()

    # Populate with cuda memory
    df["keys"] = rmm.to_device(np.arange(10, dtype=np.float64))
    np.testing.assert_equal(df["keys"].to_array(), np.arange(10))
    assert len(df) == 10

    # Populate with numpy array
    rnd_vals = np.random.random(10)
    df["vals"] = rnd_vals
    np.testing.assert_equal(df["vals"].to_array(), rnd_vals)
    assert len(df) == 10
    assert tuple(df.columns) == ("keys", "vals")

    # Make another dataframe
    df2 = DataFrame()
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
    df_tup = DataFrame()
    data = np.arange(10)
    df_tup[(1, "foobar")] = data
    np.testing.assert_equal(data, df_tup[(1, "foobar")].to_array())


def test_dataframe_drop_method():
    df = DataFrame()
    data = np.asarray(range(10))
    df["a"] = data
    df["b"] = data
    df["c"] = data

    assert tuple(df.columns) == ("a", "b", "c")
    assert tuple(df.drop("a").columns) == ("b", "c")
    assert tuple(df.drop("a", axis=1).columns) == ("b", "c")
    assert tuple(df.columns) == ("a", "b", "c")
    assert tuple(df.drop(["a", "b"]).columns) == ("c",)
    assert tuple(df.columns) == ("a", "b", "c")
    assert tuple(df.drop(["a", "b"]).columns) == ("c",)
    assert tuple(df.columns) == ("a", "b", "c")
    assert tuple(df.drop(columns=["a", "b"]).columns) == ("c",)
    assert tuple(df.columns) == ("a", "b", "c")
    assert tuple(df.drop(columns="a").columns) == ("b", "c")
    assert tuple(df.columns) == ("a", "b", "c")
    assert tuple(df.drop(columns=["a"]).columns) == ("b", "c")
    assert tuple(df.columns) == ("a", "b", "c")
    assert tuple(df.drop(columns=["a", "b", "c"]).columns) == tuple()
    assert tuple(df.columns) == ("a", "b", "c")

    # Test drop error
    with pytest.raises(NameError) as raises:
        df.drop("d")
    raises.match("column 'd' does not exist")
    with pytest.raises(NameError) as raises:
        df.drop(["a", "d", "b"])
    raises.match("column 'd' does not exist")
    with pytest.raises(ValueError) as raises:
        df.drop("a", axis=1, columns="a")
    raises.match("Cannot specify both")
    with pytest.raises(ValueError) as raises:
        df.drop(axis=1)
    raises.match("Need to specify at least")


def test_dataframe_column_add_drop_via_setitem():
    df = DataFrame()
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
    df = DataFrame({"a": data_0, "b": data_1, "c": data_2})

    for i in range(10):
        df.c = df.a
        assert assert_eq(df.c, df.a, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")

        df.c = df.b
        assert assert_eq(df.c, df.b, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")


def test_dataframe_column_drop_via_attr():
    df = DataFrame({"a": []})

    with pytest.raises(AttributeError):
        del df.a

    assert tuple(df.columns) == tuple("a")


def test_dataframe_attribute_add_drop():
    df = DataFrame()

    with pytest.warns(UserWarning) as record:
        df.some_new_attr = 5

    assert len(record) == 1
    assert "A new attribute will be created" in str(record[0].message)

    assert hasattr(df, "some_new_attr")
    assert isinstance(df.some_new_attr, int)
    assert df.some_new_attr == 5

    del df.some_new_attr

    assert not hasattr(df, "some_new_attr")


def test_dataframe_pop():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [7.0, 8.0, 9.0]}
    )
    gdf = DataFrame.from_pandas(pdf)

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
    empty_gdf = DataFrame(columns=["a", "b"])
    pb = empty_pdf.pop("b")
    gb = empty_gdf.pop("b")
    assert len(pb) == len(gb)
    assert empty_pdf.empty and empty_gdf.empty


@pytest.mark.parametrize("nelem", [0, 3, 100, 1000])
def test_dataframe_astype(nelem):
    df = DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df["a"].dtype is np.dtype(np.int32)
    df["b"] = df["a"].astype(np.float32)
    assert df["b"].dtype is np.dtype(np.float32)
    np.testing.assert_equal(df["a"].to_array(), df["b"].to_array())


@pytest.mark.parametrize("nelem", [0, 100])
def test_index_astype(nelem):
    df = DataFrame()
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
    df = DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]})
    string = str(df)
    print(string)
    assert string.splitlines()[-1] == "[6 rows x 2 columns]"

    # Test skipped columns
    df = DataFrame(
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
    df = DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]})

    data = np.arange(6)
    mask = np.zeros(1, dtype=np.uint8)
    mask[0] = 0b00101101

    masked = Series.from_masked_array(data, mask)
    assert masked.null_count == 2
    df["c"] = masked

    # check data
    values = list(masked)
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
1 2 12 null
2 3 13 2
3 4 14 3
4 5 15 null
5 6 16 5
"""
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_to_string_wide():
    # Test basic
    df = DataFrame()
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
    df = DataFrame()
    got = df.to_string()
    print(got)
    expect = "Empty DataFrame\nColumns: []\nIndex: []\n"
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_emptycolumns_to_string():
    # Test for printing dataframe having empty columns
    df = DataFrame()
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

    df = DataFrame()
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
    df = DataFrame()
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
    df = DataFrame({k: np.ones(10, dtype=v) for k, v in dtypes.iteritems()})
    assert df.dtypes.equals(dtypes)


def test_dataframe_add_col_to_object_dataframe():
    # Test for adding column to an empty object dataframe
    cols = ["a", "b", "c"]
    df = pd.DataFrame(columns=cols, dtype="str")

    data = {k: v for (k, v) in zip(cols, [["a"] for _ in cols])}

    gdf = DataFrame(data)
    gdf = gdf[:0]

    assert gdf.dtypes.equals(df.dtypes)
    gdf["a"] = [1]
    df["a"] = [10]
    assert gdf.dtypes.equals(df.dtypes)
    gdf["b"] = [1.0]
    df["b"] = [10.0]
    assert gdf.dtypes.equals(df.dtypes)


def test_dataframe_dir_and_getattr():
    df = DataFrame(
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
def test_dataframe_as_gpu_matrix(order):
    df = DataFrame()

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
    df = DataFrame()

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
    gdf = DataFrame.from_pandas(pdf)

    gdf["newcol"] = 100
    pdf["newcol"] = 100

    assert len(gdf["newcol"]) == len(pdf)
    assert len(pdf["newcol"]) == len(pdf)
    pd.testing.assert_frame_equal(gdf.to_pandas(), pdf)


def test_dataframe_setitem_from_masked_object():
    ary = np.random.randn(100)
    mask = np.zeros(100, dtype=bool)
    mask[:20] = True
    np.random.shuffle(mask)
    ary[mask] = np.nan

    test1_null = Series(ary, nan_as_null=True)
    assert test1_null.nullable
    assert test1_null.null_count == 20
    test1_nan = Series(ary, nan_as_null=False)
    assert test1_nan.null_count == 0

    test2_null = DataFrame.from_pandas(
        pd.DataFrame({"a": ary}), nan_as_null=True
    )
    assert test2_null["a"].nullable
    assert test2_null["a"].null_count == 20
    test2_nan = DataFrame.from_pandas(
        pd.DataFrame({"a": ary}), nan_as_null=False
    )
    assert test2_nan["a"].null_count == 0

    gpu_ary = rmm.to_device(ary)
    test3_null = Series(gpu_ary, nan_as_null=True)
    assert test3_null.nullable
    assert test3_null.null_count == 20
    test3_nan = Series(gpu_ary, nan_as_null=False)
    assert test3_nan.null_count == 0

    test4 = DataFrame()
    lst = [1, 2, None, 4, 5, 6, None, 8, 9]
    test4["lst"] = lst
    assert test4["lst"].nullable
    assert test4["lst"].null_count == 2


def test_dataframe_append_to_empty():
    pdf = pd.DataFrame()
    pdf["a"] = []
    pdf["b"] = [1, 2, 3]

    gdf = DataFrame()
    gdf["a"] = []
    gdf["b"] = [1, 2, 3]

    pd.testing.assert_frame_equal(gdf.to_pandas(), pdf)


def test_dataframe_setitem_index_len1():
    gdf = DataFrame()
    gdf["a"] = [1]
    gdf["b"] = gdf.index.as_column()

    np.testing.assert_equal(gdf.b.to_array(), [0])


def test_assign():
    gdf = DataFrame({"x": [1, 2, 3]})
    gdf2 = gdf.assign(y=gdf.x + 1)
    assert list(gdf.columns) == ["x"]
    assert list(gdf2.columns) == ["x", "y"]

    np.testing.assert_equal(gdf2.y.to_array(), [2, 3, 4])


@pytest.mark.parametrize("nrows", [1, 8, 100, 1000])
def test_dataframe_hash_columns(nrows):
    gdf = DataFrame()
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
    # Equivalent to the Series.hash_values()
    np.testing.assert_array_equal(cupy.asnumpy(gdf.a.hash_values()), out_one)


@pytest.mark.parametrize("nrows", [3, 10, 100, 1000])
@pytest.mark.parametrize("nparts", [1, 2, 8, 13])
@pytest.mark.parametrize("nkeys", [1, 2])
def test_dataframe_hash_partition(nrows, nparts, nkeys):
    np.random.seed(123)
    gdf = DataFrame()
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
    assert all(isinstance(p, DataFrame) for p in got)
    # Check that all partitions have unique keys
    part_unique_keys = set()
    for p in got:
        if len(p):
            # Take rows of the keycolums and build a set of the key-values
            unique_keys = set(map(tuple, p.as_matrix(columns=keycols)))
            # Ensure that none of the key-values have occurred in other groups
            assert not (unique_keys & part_unique_keys)
            part_unique_keys |= unique_keys
    assert len(part_unique_keys)


@pytest.mark.parametrize("nrows", [3, 10, 50])
def test_dataframe_hash_partition_masked_value(nrows):
    gdf = DataFrame()
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
            expected_value = row.key + 100 if valid else -1
            got_value = row.val
            assert expected_value == got_value


@pytest.mark.parametrize("nrows", [3, 10, 50])
def test_dataframe_hash_partition_masked_keys(nrows):
    gdf = DataFrame()
    gdf["key"] = np.arange(nrows)
    gdf["val"] = np.arange(nrows) + 100
    bitmask = utils.random_bitmask(nrows)
    bytemask = utils.expand_bits_to_bytes(bitmask)
    gdf["key"] = gdf["key"].set_mask(bitmask)
    parted = gdf.partition_by_hash(["key"], nparts=3)
    # Verify that the valid mask is correct
    for p in parted:
        df = p.to_pandas()
        for row in df.itertuples():
            valid = bool(bytemask[row.val - 100])
            # val is key + 100
            expected_value = row.val - 100 if valid else -1
            got_value = row.key
            assert expected_value == got_value


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
    with pytest.raises(ValueError):
        gd.concat([df1, df2])


@pytest.mark.parametrize(
    "df_1", [DataFrame({"a": [1, 2], "b": [1, 3]}), DataFrame({})]
)
@pytest.mark.parametrize(
    "df_2", [DataFrame({"a": [], "b": []}), DataFrame({})]
)
def test_concat_empty_dataframe(df_1, df_2):

    got = gd.concat([df_1, df_2])
    expect = pd.concat([df_1.to_pandas(), df_2.to_pandas()], sort=False)

    # ignoring dtypes as pandas upcasts int to float
    # on concatenation with empty dataframes

    pd.testing.assert_frame_equal(got.to_pandas(), expect, check_dtype=False)


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
        [DataFrame(df1_d), DataFrame(df2_d), DataFrame(df1_d)], sort=False
    )

    expect = pd.concat(
        [pd.DataFrame(df1_d), pd.DataFrame(df2_d), pd.DataFrame(df1_d)],
        sort=False,
    )

    # numerical columns are upcasted to float in cudf.DataFrame.to_pandas()
    # casts nan to -1 in non-float numerical columns

    numeric_cols = got.dtypes[got.dtypes != "object"].index
    for col in numeric_cols:
        got[col] = got[col].astype(np.float64).fillna(np.nan)

    pd.testing.assert_frame_equal(got.to_pandas(), expect, check_dtype=False)


@pytest.mark.parametrize("ser_1", [pd.Series([1, 2, 3]), pd.Series([])])
@pytest.mark.parametrize("ser_2", [pd.Series([])])
def test_concat_empty_series(ser_1, ser_2):
    got = gd.concat([Series(ser_1), Series(ser_2)])
    expect = pd.concat([ser_1, ser_2])

    pd.testing.assert_series_equal(got.to_pandas(), expect)


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

    gdf = DataFrame()
    gdf["a"] = np.random.randint(2147483647, size=nrows)
    gdf["b"] = np.random.randint(2147483647, size=nrows)
    gdf = gdf.set_index("b")

    test_values = np.random.randint(2147483647, size=nrows)
    gdf["c"] = test_values
    assert len(test_values) == len(gdf["c"])
    assert (
        gdf["c"]
        .to_pandas()
        .equals(Series(test_values).set_index(gdf._index).to_pandas())
    )


def test_from_pandas():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])
    gdf = gd.DataFrame.from_pandas(df)
    assert isinstance(gdf, gd.DataFrame)

    pd.testing.assert_frame_equal(df, gdf.to_pandas())

    s = df.x
    gs = gd.Series.from_pandas(s)
    assert isinstance(gs, gd.Series)

    pd.testing.assert_series_equal(s, gs.to_pandas())


@pytest.mark.parametrize("dtypes", [int, float])
def test_from_records(dtypes):
    h_ary = np.ndarray(shape=(10, 4), dtype=dtypes)
    rec_ary = h_ary.view(np.recarray)

    gdf = gd.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    df = pd.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    assert isinstance(gdf, gd.DataFrame)

    pd.testing.assert_frame_equal(df, gdf.to_pandas())

    gdf = gd.DataFrame.from_records(rec_ary)
    df = pd.DataFrame.from_records(rec_ary)
    assert isinstance(gdf, gd.DataFrame)

    pd.testing.assert_frame_equal(df, gdf.to_pandas())


def test_from_gpu_matrix():
    h_ary = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    d_ary = rmm.to_device(h_ary)

    gdf = gd.DataFrame.from_gpu_matrix(d_ary, columns=["a", "b", "c"])
    df = pd.DataFrame(h_ary, columns=["a", "b", "c"])
    assert isinstance(gdf, gd.DataFrame)

    pd.testing.assert_frame_equal(df, gdf.to_pandas())

    gdf = gd.DataFrame.from_gpu_matrix(d_ary)
    df = pd.DataFrame(h_ary)
    assert isinstance(gdf, gd.DataFrame)

    pd.testing.assert_frame_equal(df, gdf.to_pandas())


@pytest.mark.xfail(reason="matrix dimension is not 2")
def test_from_gpu_matrix_wrong_dimensions():
    d_ary = rmm.device_array((2, 3, 4), dtype=np.int32)
    gdf = gd.DataFrame.from_gpu_matrix(d_ary)
    assert gdf is not None


@pytest.mark.xfail(reason="constructor does not coerce index inputs")
def test_index_in_dataframe_constructor():
    a = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])
    b = gd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])

    pd.testing.assert_frame_equal(a, b.to_pandas())
    assert pd.testing.assert_frame_equal(a.loc[4:], b.loc[4:].to_pandas())


@pytest.mark.parametrize("nelem", [0, 2, 3, 100, 1000])
@pytest.mark.parametrize(
    "data_type",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
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

    pd.testing.assert_frame_equal(df, gdf.to_pandas())

    s = pa.Array.from_pandas(df.a)
    gs = gd.Series.from_arrow(s)
    assert isinstance(gs, gd.Series)

    # For some reason PyArrow to_pandas() converts to numpy array and has
    # better type compatibility
    np.testing.assert_array_equal(s.to_pandas(), gs.to_array())


@pytest.mark.parametrize("nelem", [0, 2, 3, 100, 1000])
@pytest.mark.parametrize(
    "data_type",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
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


@pytest.mark.parametrize(
    "data_type",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_to_from_arrow_nulls(data_type):
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
        np.array(s1.buffers()[0])[0],
        gs1._column.mask_array_view.copy_to_host()[0],
    )
    assert pa.Array.equals(s1, gs1.to_arrow())

    s2 = pa.array([None, None, None, None, None], type=data_type)
    gs2 = gd.Series.from_arrow(s2)
    assert isinstance(gs2, gd.Series)
    # We have 64B padded buffers for nulls whereas Arrow returns a minimal
    # number of bytes, so only check the first byte in this case
    np.testing.assert_array_equal(
        np.array(s2.buffers()[0])[0],
        gs2._column.mask_array_view.copy_to_host()[0],
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
    pd.testing.assert_series_equal(
        pd.Series(pa_cat.to_pandas()),  # PyArrow returns a pd.Categorical
        gd_cat.to_pandas(),
    )


def test_to_arrow_missing_categorical():
    pd_cat = pd.Categorical(["a", "b", "c"], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = gd.Series(pa_cat)

    assert isinstance(gd_cat, gd.Series)
    assert pa.Array.equals(pa_cat, gd_cat.to_arrow())


@pytest.mark.parametrize(
    "data_type",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "longlong",
        "float32",
        "float64",
        "datetime64[ms]",
        "datetime64",
        "datetime64[ns]",
        "datetime64[D]",
        "datetime64[s]",
        "datetime64[M]",
    ],
)
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


@pytest.mark.parametrize(
    "data_type",
    ["int8", "int16", "int32", "int64", "float32", "float64", "longlong"],
)
def test_from_python_array(data_type):
    np_arr = np.random.randint(0, 100, 10).astype(data_type)
    data = memoryview(np_arr)
    data = arr.array(data.format, data)

    gs = gd.Series(data)

    np.testing.assert_equal(gs.to_array(), np_arr)


def test_series_shape():
    ps = pd.Series([1, 2, 3, 4])
    cs = Series([1, 2, 3, 4])

    assert ps.shape == cs.shape


def test_series_shape_empty():
    ps = pd.Series()
    cs = Series([])

    assert ps.shape == cs.shape


def test_dataframe_shape():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    gdf = DataFrame.from_pandas(pdf)

    assert pdf.shape == gdf.shape


def test_dataframe_shape_empty():
    pdf = pd.DataFrame()
    gdf = DataFrame()

    assert pdf.shape == gdf.shape


@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 20])
@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_dataframe_transpose(nulls, num_cols, num_rows, dtype):
    if dtype not in ["float32", "float64"] and nulls in ["some", "all"]:
        pytest.skip(msg="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame()
    from string import ascii_lowercase

    for i in range(num_cols):
        colname = ascii_lowercase[i]
        data = np.random.randint(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = np.random.choice(
                num_rows, size=int(num_rows / 2), replace=False
            )
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    got_function = gdf.transpose()
    got_property = gdf.T

    expect = pdf.transpose()

    assert_eq(expect, got_function)
    assert_eq(expect, got_property)


@pytest.mark.parametrize("num_cols", [0, 1, 2, 10])
@pytest.mark.parametrize("num_rows", [0, 1, 2, 1000])
def test_dataframe_tranpose_category(num_cols, num_rows):
    pytest.xfail("category dtype not yet supported for transpose")
    pdf = pd.DataFrame()
    from string import ascii_lowercase

    for i in range(num_cols):
        colname = ascii_lowercase[i]
        data = pd.Series(list(ascii_lowercase), dtype="category")
        data = data.sample(num_rows, replace=True).reset_index(drop=True)
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    got_function = gdf.transpose()
    got_property = gdf.T

    expect = pdf.transpose()

    pd.testing.assert_frame_equal(expect, got_function.to_pandas())
    pd.testing.assert_frame_equal(expect, got_property.to_pandas())


def test_generated_column():
    gdf = DataFrame({"a": (i for i in range(5))})
    assert len(gdf) == 5


@pytest.fixture
def pdf():
    return pd.DataFrame({"x": range(10), "y": range(10)})


@pytest.fixture
def gdf(pdf):
    return gd.DataFrame.from_pandas(pdf)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.count(),
        lambda df: df.min(),
        lambda df: df.max(),
        lambda df: df.sum(),
        lambda df: df.product(),
        lambda df: df.cummin(),
        lambda df: df.cummax(),
        lambda df: df.cumsum(),
        lambda df: df.cumprod(),
        lambda df: df.mean(),
        lambda df: df.sum(),
        lambda df: df.max(),
        lambda df: df.std(ddof=1),
        lambda df: df.var(ddof=1),
        lambda df: df.std(ddof=2),
        lambda df: df.var(ddof=2),
    ],
)
def test_dataframe_reductions(func):
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    print(func(pdf))
    gdf = DataFrame.from_pandas(pdf)
    print(func(gdf))
    assert_eq(func(pdf), func(gdf))


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

    gdf2 = DataFrame.from_arrow(pdf_arrow_table)
    pdf2 = pdf_arrow_table.to_pandas()

    assert_eq(pdf2, gdf2)


@pytest.mark.parametrize("nrows", [1, 8, 100, 1000, 100000])
def test_series_hash_encode(nrows):
    data = np.asarray(range(nrows))
    s = Series(data, name="x1")
    num_features = 1000

    encoded_series = s.hash_encode(num_features)
    assert isinstance(encoded_series, gd.Series)
    enc_arr = encoded_series.to_array()
    assert np.all(enc_arr >= 0)
    assert np.max(enc_arr) < num_features

    enc_with_name_arr = s.hash_encode(num_features, use_name=True).to_array()
    assert enc_with_name_arr[0] != enc_arr[0]


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
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
@pytest.mark.parametrize(
    "data_type",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
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
    from numba import cuda
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
    gdf = DataFrame.from_pandas(pdf)
    gdf_mask = DataFrame.from_pandas(pdf_mask)

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
            Series([True, False, True]),
            marks=pytest.mark.xfail(
                reason="Pandas can't index a multiindex with a Series"
            ),
        ),
    ],
)
def test_dataframe_multiindex_boolmask(mask):
    gdf = DataFrame(
        {"w": [3, 2, 1], "x": [1, 2, 3], "y": [0, 1, 0], "z": [1, 1, 1]}
    )
    gdg = gdf.groupby(["w", "x"]).count()
    pdg = gdg.to_pandas()
    assert_eq(gdg[mask], pdg[mask])


def test_dataframe_assignment():
    pdf = pd.DataFrame()
    for col in "abc":
        pdf[col] = np.array([0, 1, 1, -2, 10])
    gdf = DataFrame.from_pandas(pdf)
    gdf[gdf < 0] = 999
    pdf[pdf < 0] = 999
    assert_eq(gdf, pdf)


def test_1row_arrow_table():
    data = [pa.array([0]), pa.array([1])]
    batch = pa.RecordBatch.from_arrays(data, ["f0", "f1"])
    table = pa.Table.from_batches([batch])

    expect = table.to_pandas()
    got = DataFrame.from_arrow(table)
    assert_eq(expect, got)


def test_arrow_handle_no_index_name(pdf, gdf):
    gdf_arrow = gdf.to_arrow()
    pdf_arrow = pa.Table.from_pandas(pdf)
    assert pa.Table.equals(pdf_arrow, gdf_arrow)

    got = DataFrame.from_arrow(gdf_arrow)
    expect = pdf_arrow.to_pandas()
    assert_eq(expect, got)


@pytest.mark.parametrize("num_rows", [1, 3, 10, 100])
@pytest.mark.parametrize("num_bins", [1, 2, 4, 20])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
def test_series_digitize(num_rows, num_bins, right, dtype):
    data = np.random.randint(0, 100, num_rows).astype(dtype)
    bins = np.unique(np.sort(np.random.randint(2, 95, num_bins).astype(dtype)))
    s = Series(data)
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
    got = Series(data)

    assert_eq(expect, got)


@pytest.mark.parametrize("num_elements", [0, 2, 10, 100])
def test_series_all_valid_nan(num_elements):
    data = [np.nan] * num_elements
    sr = Series(data, nan_as_null=False)
    np.testing.assert_equal(sr.null_count, 0)


def test_dataframe_rename():
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf = DataFrame.from_pandas(pdf)

    expect = pdf.rename(columns=lambda name: 2 * name)
    got = gdf.rename(columns=lambda name: 2 * name)

    assert_eq(expect, got)

    rename_mapper = {"a": "z", "b": "y", "c": "x"}
    expect = pdf.rename(columns=rename_mapper)
    got = gdf.rename(columns=rename_mapper)

    assert_eq(expect, got)


def test_series_rename():
    pds = pd.Series([1, 2, 3], name="asdf")
    gds = Series([1, 2, 3], name="asdf")

    expect = pds.rename("new_name")
    got = gds.rename("new_name")

    assert_eq(expect, got)

    pds = pd.Series(expect)
    gds = Series(got)

    assert_eq(pds, gds)

    pds = pd.Series(expect, name="name name")
    gds = Series(got, name="name name")

    assert_eq(pds, gds)


@pytest.mark.parametrize(
    "data_type",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
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
    gdf = DataFrame()
    gdf["id"] = Series(["a", "b"], dtype=np.object)
    gdf["v"] = Series([1, 2])
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


@pytest.mark.parametrize("drop", [True, False])
def test_set_index(pdf, gdf, drop):
    for col in pdf.columns:
        assert_eq(pdf.set_index(col, drop=drop), gdf.set_index(col, drop=drop))


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
    gdf = DataFrame.from_pandas(df)

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
    gdf = DataFrame.from_pandas(pdf)

    expect = pdf.sort_index()
    got = gdf.sort_index()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_dataframe_0_row_dtype(dtype):
    if dtype == "category":
        data = pd.Series(["a", "b", "c", "d", "e"], dtype="category")
    else:
        data = np.array([1, 2, 3, 4, 5], dtype=dtype)

    expect = DataFrame()
    expect["x"] = data
    expect["y"] = data
    got = expect.head(0)

    for col_name in got.columns:
        assert expect[col_name].dtype == got[col_name].dtype

    expect = Series(data)
    got = expect.head(0)

    assert expect.dtype == got.dtype


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_series_list_nanasnull(nan_as_null):
    data = [1.0, 2.0, 3.0, np.nan, None]

    expect = pa.array(data, from_pandas=nan_as_null)
    got = Series(data, nan_as_null=nan_as_null).to_arrow()

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

    gdf = DataFrame({"A": [3, 4, 5], "C": [1, 2, 3], "D": ["a", "b", "c"]})
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

    assert_eq(gdf[["timestamp"]], gdf.select_dtypes("datetime64"))
    assert_eq(gdf[["timestamp"]], gdf.select_dtypes(np.dtype("datetime64")))
    assert_eq(gdf[["timestamp"]], gdf.select_dtypes(include="datetime64"))
    assert_eq(gdf[["timestamp"]], gdf.select_dtypes("datetime64[ms]"))
    assert_eq(
        gdf[["timestamp"]], gdf.select_dtypes(np.dtype("datetime64[ms]"))
    )
    assert_eq(gdf[["timestamp"]], gdf.select_dtypes(include="datetime64[ms]"))


def test_array_ufunc():
    gdf = gd.DataFrame({"x": [2, 3, 4.0], "y": [9.0, 2.5, 1.1]})
    pdf = gdf.to_pandas()

    assert_eq(np.sqrt(gdf), np.sqrt(pdf))
    assert_eq(np.sqrt(gdf.x), np.sqrt(pdf.x))


@pytest.mark.parametrize("nan_value", [-5, -5.0, 0, 5, 5.0, None, "pandas"])
def test_series_to_gpu_array(nan_value):

    s = Series([0, 1, None, 3])
    np.testing.assert_array_equal(
        s.to_array(nan_value), s.to_gpu_array(nan_value).copy_to_host()
    )


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
def test_series_describe_numeric(dtype):
    pdf = pd.Series([0, 1, 2, 3])
    gdf = Series.from_pandas(pdf).astype(dtype)
    gdf_results = gdf.describe().to_pandas()
    pdf_results = gdf.to_pandas().describe()

    np.testing.assert_array_almost_equal(
        gdf_results.values, pdf_results.values, decimal=4
    )


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Describing non-numeric columns is not yet supported.",
)
def test_series_describe_datetime():
    pdf = pd.Series([0, 1, 2, 3]).astype("datetime64[ms]")
    gdf = Series.from_pandas(pdf)
    gdf_results = gdf.describe()
    pdf_results = pdf.describe()

    np.testing.assert_array_almost_equal(
        gdf_results.values, pdf_results.values, decimal=4
    )


def test_dataframe_describe_exclude():
    np.random.seed(12)
    data_length = 10000

    df = DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["x"] = df.x.astype("int64")
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe(exclude=["float"]).to_pandas()
    pdf_results = pdf.describe(exclude=["float"])

    np.testing.assert_array_almost_equal(
        gdf_results.values, pdf_results.values, decimal=4
    )


def test_dataframe_describe_include():
    np.random.seed(12)
    data_length = 10000

    df = DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["x"] = df.x.astype("int64")
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe(include=["int"]).to_pandas()
    pdf_results = pdf.describe(include=["int"])

    np.testing.assert_array_almost_equal(
        gdf_results.values, pdf_results.values, decimal=4
    )


def test_dataframe_describe_default():
    np.random.seed(12)
    data_length = 10000

    df = DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe().to_pandas()
    pdf_results = pdf.describe()

    assert_eq(pdf_results, gdf_results)


@pytest.mark.xfail(
    raises=AssertionError,
    reason="Describing non-numeric columns is not yet supported.",
)
def test_series_describe_include_all():
    np.random.seed(12)
    data_length = 10000

    df = DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["x"] = df.x.astype("int64")
    df["y"] = np.random.normal(10, 1, data_length)
    df["animal"] = np.random.choice(["dog", "cat", "bird"], data_length)

    pdf = df.to_pandas()
    gdf_results = df.describe(include="all").to_pandas()
    pdf_results = pdf.describe(include="all")

    np.testing.assert_array_almost_equal(
        gdf_results.values, pdf_results.values, decimal=4
    )


def test_dataframe_describe_percentiles():
    np.random.seed(12)
    data_length = 10000
    sample_percentiles = [0.0, 0.1, 0.33, 0.84, 0.4, 0.99]

    df = DataFrame()
    df["x"] = np.random.normal(10, 1, data_length)
    df["y"] = np.random.normal(10, 1, data_length)
    pdf = df.to_pandas()
    gdf_results = df.describe(percentiles=sample_percentiles).to_pandas()
    pdf_results = pdf.describe(percentiles=sample_percentiles)

    assert_eq(pdf_results, gdf_results)


def test_get_numeric_data():
    pdf = pd.DataFrame(
        {"x": [1, 2, 3], "y": [1.0, 2.0, 3.0], "z": ["a", "b", "c"]}
    )
    gdf = gd.from_pandas(pdf)

    assert_eq(pdf._get_numeric_data(), gdf._get_numeric_data())


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
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

    gdf = DataFrame({"a": Series(data, dtype=dtype)})
    pdf = pd.DataFrame({"a": pd.Series(data, dtype=dtype)})

    shifted_outcome = gdf.a.shift(period)
    expected_outcome = pdf.a.shift(period).fillna(-1).astype(dtype)

    if data_empty:
        assert_eq(shifted_outcome, expected_outcome, check_index_type=False)
    else:
        assert_eq(shifted_outcome, expected_outcome)


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
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

    gdf = DataFrame({"a": Series(data, dtype=dtype)})
    pdf = pd.DataFrame({"a": pd.Series(data, dtype=dtype)})

    diffed_outcome = gdf.a.diff(period)
    expected_outcome = pdf.a.diff(period).fillna(-1).astype(dtype)

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
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # integer & string none missing
    ps = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": ["a", "b", "u", "h", "d"]})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # all missing
    ps = pd.DataFrame(
        {"a": [None, None, np.nan, None], "b": [np.nan, None, np.nan, None]}
    )
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # empty
    ps = pd.DataFrame({"a": []})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # one missing
    ps = pd.DataFrame({"a": [np.nan], "b": [None]})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # strings missing
    ps = pd.DataFrame({"a": ["a", "b", "c", None, "e"]})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # strings none missing
    ps = pd.DataFrame({"a": ["a", "b", "c", "d", "e"]})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.a.isnull(), gs.a.isnull())
    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.a.isna(), gs.a.isna())
    assert_eq(ps.isna(), gs.isna())

    # unnamed series
    ps = pd.Series([0, 1, 2, np.nan, 4, None, 6])
    gs = Series.from_pandas(ps)
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
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # integer & string none missing
    ps = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": ["a", "b", "u", "h", "d"]})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # all missing
    ps = pd.DataFrame(
        {"a": [None, None, np.nan, None], "b": [np.nan, None, np.nan, None]}
    )
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # empty
    ps = pd.DataFrame({"a": []})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # one missing
    ps = pd.DataFrame({"a": [np.nan], "b": [None]})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # strings missing
    ps = pd.DataFrame({"a": ["a", "b", "c", None, "e"]})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # strings none missing
    ps = pd.DataFrame({"a": ["a", "b", "c", "d", "e"]})
    gs = DataFrame.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.a.notna(), gs.a.notna())
    assert_eq(ps.notnull(), gs.notnull())
    assert_eq(ps.a.notnull(), gs.a.notnull())

    # unnamed series
    ps = pd.Series([0, 1, 2, np.nan, 4, None, 6])
    gs = Series.from_pandas(ps)
    assert_eq(ps.notna(), gs.notna())
    assert_eq(ps.notnull(), gs.notnull())


def test_ndim():
    pdf = pd.DataFrame({"x": range(5), "y": range(5, 10)})
    gdf = DataFrame.from_pandas(pdf)
    assert pdf.ndim == gdf.ndim
    assert pdf.x.ndim == gdf.x.ndim

    s = pd.Series()
    gs = Series()
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
    ser = Series(arr)
    result = ser.round(decimal)
    expected = pser.round(decimal)

    assert_eq(result, expected)

    # with nulls, maintaining existing null mask
    arr = arr.astype("float64")  # for pandas nulls
    mask = np.random.randint(0, 2, arr.shape[0])
    arr[mask == 1] = np.nan

    pser = pd.Series(arr)
    ser = Series(arr)
    result = ser.round(decimal)
    expected = pser.round(decimal)

    assert_eq(result, expected)
    np.array_equal(ser.nullmask.to_array(), result.to_array())


@pytest.mark.parametrize(
    "series",
    [
        Series([1.0, None, np.nan, 4.0], nan_as_null=False),
        Series([1.24430, None, np.nan, 4.423530], nan_as_null=False),
        Series([1.24430, np.nan, 4.423530], nan_as_null=False),
        Series([-1.24430, np.nan, -4.423530], nan_as_null=False),
        Series(np.repeat(np.nan, 100)),
    ],
)
@pytest.mark.parametrize("decimal", [0, 1, 2, 3])
def test_round_nan_as_null_false(series, decimal):
    pser = series.to_pandas()
    ser = Series(series)
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
    if np.array(data).ndim <= 1:
        pdata = pd.Series(data)
        gdata = Series.from_pandas(pdata)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"])
        gdata = DataFrame.from_pandas(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.all(bool_only=True)
            expected = pdata.all(bool_only=True)
            assert_eq(got, expected)

    got = gdata.all()
    expected = pdata.all()
    assert_eq(got, expected)


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
                    reason="NotImplementedError: any does not "
                    "support columns of object dtype."
                )
            ],
        ),
    ],
)
def test_any(data):
    if np.array(data).ndim <= 1:
        pdata = pd.Series(data)
        gdata = Series.from_pandas(pdata)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"])
        gdata = DataFrame.from_pandas(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.all(bool_only=True)
            expected = pdata.all(bool_only=True)
            assert_eq(got, expected)

    got = gdata.any()
    expected = pdata.any()
    assert_eq(got, expected)


@pytest.mark.parametrize("indexed", [False, True])
def test_dataframe_sizeof(indexed):
    rows = int(1e6)
    index = list(i for i in range(rows)) if indexed else None

    gdf = gd.DataFrame({"A": [8] * rows, "B": [32] * rows}, index=index)

    for c in gdf._data.values():
        assert gdf._index.__sizeof__() == gdf._index.__sizeof__()
    cols_sizeof = sum(c.__sizeof__() for c in gdf._data.values())
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
    actual = DataFrame.from_pandas(expected)
    expected["b"] = b
    actual["b"] = b
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": []})
    actual = DataFrame.from_pandas(expected)
    expected["b"] = misc_data
    actual["b"] = misc_data
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": a})
    actual = DataFrame.from_pandas(expected)
    expected["b"] = non_list_data
    actual["b"] = non_list_data
    assert_eq(actual, expected)


def test_empty_dataframe_describe():
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = DataFrame.from_pandas(pdf)

    expected = pdf.describe()
    actual = gdf.describe()

    assert_eq(expected, actual)


def test_as_column_types():
    from cudf.core.column import column

    col = column.as_column(Series([]))
    assert_eq(col.dtype, np.dtype("float64"))
    gds = Series(col)
    pds = pd.Series(pd.Series([]))

    assert_eq(pds, gds)

    col = column.as_column(Series([]), dtype="float32")
    assert_eq(col.dtype, np.dtype("float32"))
    gds = Series(col)
    pds = pd.Series(pd.Series([], dtype="float32"))

    assert_eq(pds, gds)

    col = column.as_column(Series([]), dtype="str")
    assert_eq(col.dtype, np.dtype("object"))
    gds = Series(col)
    pds = pd.Series(pd.Series([], dtype="str"))

    assert_eq(pds, gds)

    col = column.as_column(Series([]), dtype="object")
    assert_eq(col.dtype, np.dtype("object"))
    gds = Series(col)
    pds = pd.Series(pd.Series([], dtype="object"))

    assert_eq(pds, gds)

    pds = pd.Series(np.array([1, 2, 3]), dtype="float32")
    gds = Series(column.as_column(np.array([1, 2, 3]), dtype="float32"))

    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 3], dtype="float32")
    gds = Series([1, 2, 3], dtype="float32")

    assert_eq(pds, gds)

    pds = pd.Series([])
    gds = Series(column.as_column(pds))
    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 4], dtype="int64")
    gds = Series(column.as_column(Series([1, 2, 4]), dtype="int64"))

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="float32")
    gds = Series(column.as_column(Series([1.2, 18.0, 9.0]), dtype="float32"))

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="str")
    gds = Series(column.as_column(Series([1.2, 18.0, 9.0]), dtype="str"))

    assert_eq(pds, gds)

    pds = pd.Series(pd.Index(["1", "18", "9"]), dtype="int")
    gds = Series(gd.core.index.StringIndex(["1", "18", "9"]), dtype="int")

    assert_eq(pds, gds)


def test_one_row_head():
    gdf = DataFrame({"name": ["carl"], "score": [100]}, index=[123])
    pdf = gdf.to_pandas()

    head_gdf = gdf.head()
    head_pdf = pdf.head()

    assert_eq(head_pdf, head_gdf)


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize(
    "as_dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
def test_series_astype_numeric_to_numeric(dtype, as_dtype):
    psr = pd.Series([1, 2, 4, 3], dtype=dtype)
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize(
    "as_dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
def test_series_astype_numeric_to_numeric_nulls(dtype, as_dtype):
    data = [1, 2, None, 3]
    sr = gd.Series(data, dtype=dtype)
    got = sr.astype(as_dtype)
    expect = gd.Series([1, 2, None, 3], dtype=as_dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
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
        kwargs = {"format": "%Y-%m-%d"}
    else:
        data = ["1", "2", "3"]
        kwargs = {}
    psr = pd.Series(data)
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype, **kwargs))


@pytest.mark.parametrize(
    "as_dtype",
    [
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
    ],
)
def test_series_astype_datetime_to_other(as_dtype):
    data = ["2001-01-01", "2002-02-02", "2001-01-05"]
    psr = pd.Series(data)
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype, format="%Y-%m-%d"))


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
def test_series_astype_categorical_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05", "2001-01-01"]
        kwargs = {"format": "%Y-%m-%d"}
    else:
        data = [1, 2, 3, 1]
        kwargs = {}
    psr = pd.Series(data, dtype="category")
    gsr = gd.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype, **kwargs))


@pytest.mark.parametrize("ordered", [True, False])
def test_series_astype_to_categorical_ordered(ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    gsr = gd.from_pandas(psr)
    assert_eq(
        psr.astype("int32", ordered=ordered),
        gsr.astype("int32", ordered=ordered),
    )


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
        gd.Series(data, dtype="datetime64[ms]"),
        gd.Series(data).astype("datetime64[ms]", format="%Y-%m-%d"),
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
        gd.Series(data, dtype="category").astype(
            "datetime64[ms]", format="%Y-%m-%d"
        ),
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
            "datetime64[ms]", format="%Y-%m-%d"
        ),
    )

    assert_eq(
        gd.Series(["a", "b", "c", None], dtype="category").to_pandas(),
        gd.Series(["a", "b", "c", None]).astype("category").to_pandas(),
    )

    # datetime to other
    data = ["2001-01-01", "2001-02-01", None, "2001-03-01"]

    assert_eq(
        gd.from_pandas(pd.Series(data)),
        gd.from_pandas(pd.Series(data, dtype="datetime64[ns]")).astype(
            "str", format="%Y-%m-%d"
        ),
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
    gdf = DataFrame(data, index=["count", "mean", "std", "min"])

    assert_eq(pdf, gdf)

    pdf = pd.DataFrame(data)
    gdf = DataFrame(data)

    assert_eq(pdf, gdf)


def test_create_dataframe_column():
    pdf = pd.DataFrame(columns=["a", "b", "c"], index=["A", "Z", "X"])
    gdf = DataFrame(columns=["a", "b", "c"], index=["A", "Z", "X"])

    assert_eq(pdf, gdf)

    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [2, 3, 5]},
        columns=["a", "b", "c"],
        index=["A", "Z", "X"],
    )
    gdf = DataFrame(
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
    gds = Series(data)

    np.testing.assert_array_equal(pds.values, gds.values_host)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4],
        [],
        [5.0, 7.0, 8.0],
        pytest.param(
            pd.Categorical(["a", "b", "c"]),
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        pytest.param(
            ["m", "a", "d", "v"], marks=pytest.mark.xfail(raises=TypeError)
        ),
    ],
)
def test_series_values_property(data):
    pds = pd.Series(data)
    gds = Series(data)
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
    gdf = DataFrame.from_pandas(pdf)

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

    gdf = DataFrame(
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
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_value_counts(data, nulls):
    psr = pd.Series(data)

    if len(data) > 0:
        if nulls == "one":
            p = np.random.randint(0, len(data))
            psr[p] = None
        elif nulls == "some":
            p = np.random.randint(0, len(data), 2)
            psr[p] = None

    gsr = Series.from_pandas(psr)
    expected = psr.value_counts()
    got = gsr.value_counts()

    pandas_dict = expected.to_dict()
    gdf_dict = got.to_pandas().to_dict()

    assert pandas_dict == gdf_dict


@pytest.mark.parametrize("num_elements", [10, 100, 1000])
def test_categorical_value_counts(num_elements):
    from string import ascii_letters, digits

    # create categorical series
    np.random.seed(12)
    pd_cat = pd.Categorical(
        pd.Series(
            np.random.choice(list(ascii_letters + digits), num_elements),
            dtype="category",
        )
    )

    # gdf
    gdf = DataFrame()
    gdf["a"] = Series.from_categorical(pd_cat)
    gdf_value_counts = gdf["a"].value_counts()

    # pandas
    pdf = pd.DataFrame()
    pdf["a"] = pd_cat
    pdf_value_counts = pdf["a"].value_counts()

    # verify
    pandas_dict = pdf_value_counts.to_dict()
    gdf_dict = gdf_value_counts.to_pandas().to_dict()

    assert pandas_dict == gdf_dict


def test_series_value_counts():
    for size in [10 ** x for x in range(5)]:
        arr = np.random.randint(low=-1, high=10, size=size)
        mask = arr != -1
        sr = Series.from_masked_array(arr, Series(mask).as_mask())
        sr.name = "col"
        df = pd.DataFrame(data=arr[mask], columns=["col"])
        expect = df.col.value_counts().sort_index()
        got = sr.value_counts().sort_index()

        assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [0, 12, 14],
        np.random.randint(-100, 100, 200),
        pd.Series([0.0, 1.0, None, 10.0]),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.random.randint(-100, 100, 10),
        [],
        pytest.param([1.0, 12.0, None, None, 120], marks=pytest.mark.xfail),
        pytest.param([None, None, None], marks=pytest.mark.xfail),
        ["0", "12", "14"],
        pytest.param(
            ["0", "12", "14", "a"],
            marks=[
                pytest.mark.xfail(
                    reason="We don't gracefully handle typecasting errors."
                )
            ],
        ),
    ],
)
def test_isin_numeric(data, values):
    psr = pd.Series(data)
    gsr = Series.from_pandas(psr)

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
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        pytest.param(
            [1514764800000000000, 1577664000000000000],
            marks=[
                pytest.mark.xfail(reason="We don't yet support nanoseconds.")
            ],
        ),
        ["2019-04-03", "2019-12-30", "2012-01-01"],
    ],
)
def test_isin_datetime(data, values):
    psr = pd.Series(data)
    gsr = Series.from_pandas(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(["this", "is", None, "a", "test"]),
        pd.Series(["0", "12", "14"]),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        pytest.param([None, None, None], marks=pytest.mark.xfail),
        pytest.param(
            [12, 14, 19],
            marks=[
                pytest.mark.xfail(
                    reason="pandas's failure here seems like a bug "
                    "given the reverse succeeds"
                )
            ],
        ),
    ],
)
def test_isin_string(data, values):
    psr = pd.Series(data)
    gsr = Series.from_pandas(psr)

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
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["a", "b", None, "f", "words"],
        ["0", "12", None, "14"],
        [0, 10, 12, None, 39, 40, 1000],
    ],
)
def test_isin_categorical(data, values):
    psr = pd.Series(data)
    gsr = Series.from_pandas(psr)

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
    gsr = Series.from_pandas(psr)

    got = gsr.index.isin(values)
    expected = psr.index.isin(values)

    assert_eq(got._column.data_array_view.copy_to_host(), expected)


def test_constructor_properties():
    df = DataFrame()
    key1 = "a"
    key2 = "b"
    val1 = np.array([123], dtype=np.float64)
    val2 = np.array([321], dtype=np.float64)
    df[key1] = val1
    df[key2] = val2

    # Correct use of _constructor (for DataFrame)
    assert_eq(df, df._constructor({key1: val1, key2: val2}))

    # Correct use of _constructor (for Series)
    assert_eq(df[key1], df[key2]._constructor(val1, name=key1))

    # Correct use of _constructor_sliced (for DataFrame)
    assert_eq(df[key1], df._constructor_sliced(val1, name=key1))

    # Correct use of _constructor_expanddim (for Series)
    assert_eq(df, df[key2]._constructor_expanddim({key1: val1, key2: val2}))

    # Inorrect use of _constructor_sliced (Raises for Series)
    with pytest.raises(NotImplementedError):
        df[key1]._constructor_sliced

    # Inorrect use of _constructor_expanddim (Raises for DataFrame)
    with pytest.raises(NotImplementedError):
        df._constructor_expanddim


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5],
        [1, 2, None, 4, 5],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, None, 4.0, 5.0],
        ["a", "b", "c", "d", "e"],
        ["a", "b", None, "d", "e"],
        [None, None, None, None, None],
        np.array(["1991-11-20", "2004-12-04"], dtype=np.datetime64),
        np.array(["1991-11-20", None], dtype=np.datetime64),
        np.array(
            ["1991-11-20 05:15:00", "2004-12-04 10:00:00"], dtype=np.datetime64
        ),
        np.array(["1991-11-20 05:15:00", None], dtype=np.datetime64),
    ],
)
def test_tolist(data):
    psr = pd.Series(data)
    gsr = Series.from_pandas(psr)

    got = gsr.tolist()
    expected = [x if not pd.isnull(x) else None for x in psr.tolist()]

    np.testing.assert_array_equal(got, expected)


def test_tolist_mixed_nulls():
    num_data = pa.array([1.0, None, np.float64("nan")])
    num_data_expect = [1.0, None, np.float64("nan")]

    time_data = pa.array(
        [1, None, -9223372036854775808], type=pa.timestamp("ns")
    )
    time_data_expect = [
        pd.Timestamp("1970-01-01T00:00:00.000000001"),
        None,
        pd.NaT,
    ]

    df = DataFrame()
    df["num_data"] = num_data
    df["time_data"] = time_data

    num_data_got = df["num_data"].tolist()
    time_data_got = df["time_data"].tolist()

    np.testing.assert_equal(num_data_got, num_data_expect)
    for got, exp in zip(time_data_got, time_data_expect):  # deal with NaT
        assert (got == exp) or (pd.isnull(got) and pd.isnull(exp))


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize(
    "as_dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "str",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_df_astype_numeric_to_all(dtype, as_dtype):
    if "int" in dtype:
        data = [1, 2, None, 4, -7]
    elif "float" in dtype:
        data = [1.0, 2.0, None, 4.0, np.nan, -7.0]

    gdf = DataFrame()
    gdf["foo"] = Series(data, dtype=dtype)
    gdf["bar"] = Series(data, dtype=dtype)

    insert_data = Series(data, dtype=as_dtype)

    expect = DataFrame()
    expect["foo"] = insert_data
    expect["bar"] = insert_data

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
        data = ["2001-01-01", "2002-02-02", "2000-01-05", None]
        kwargs = {"format": "%Y-%m-%d"}
    elif as_dtype == "int32":
        data = [1, 2, 3, None]
        kwargs = {}
    elif as_dtype == "category":
        data = ["1", "2", "3", None]
        kwargs = {}
    elif "float" in as_dtype:
        data = [1.0, 2.0, 3.0, None]
        kwargs = {}

    insert_data = Series.from_pandas(pd.Series(data, dtype="str"))
    expect_data = Series(data, dtype=as_dtype)

    gdf = DataFrame()
    expect = DataFrame()

    gdf["foo"] = insert_data
    gdf["bar"] = insert_data

    expect["foo"] = expect_data
    expect["bar"] = expect_data

    got = gdf.astype(as_dtype, **kwargs)
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
    data = ["1991-11-20", "2004-12-04", "2016-09-13", None]

    gdf = DataFrame()
    gdf["foo"] = Series(data, dtype="datetime64[ms]")
    gdf["bar"] = Series(data, dtype="datetime64[ms]")

    expect = DataFrame()

    if as_dtype == "int64":
        expect["foo"] = Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
        expect["bar"] = Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
    elif as_dtype == "str":
        expect["foo"] = Series(data, dtype="str")
        expect["bar"] = Series(data, dtype="str")
    elif as_dtype == "category":
        expect["foo"] = Series(gdf["foo"], dtype="category")
        expect["bar"] = Series(gdf["bar"], dtype="category")
    else:
        expect["foo"] = Series(data, dtype=as_dtype)
        expect["bar"] = Series(data, dtype=as_dtype)

    got = gdf.astype(as_dtype, format="%Y-%m-%d")

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
        kwargs = {"format": "%Y-%m-%d"}
    else:
        data = [1, 2, 3, 1]
        kwargs = {}
    psr = pd.Series(data, dtype="category")
    pdf = pd.DataFrame()
    pdf["foo"] = psr
    pdf["bar"] = psr
    gdf = DataFrame.from_pandas(pdf)
    assert_eq(pdf.astype(as_dtype), gdf.astype(as_dtype, **kwargs))


@pytest.mark.parametrize("ordered", [True, False])
def test_df_astype_to_categorical_ordered(ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    pdf = pd.DataFrame()
    pdf["foo"] = psr
    pdf["bar"] = psr
    gdf = DataFrame.from_pandas(pdf)

    assert_eq(
        gdf.astype("int32", ordered=ordered),
        gdf.astype("int32", ordered=ordered),
    )


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
    sr = Series(["random", "words"])
    got = sr.astype("datetime64", errors=errors)
    assert_eq(sr, got)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "str",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_df_constructor_dtype(dtype):
    if "datetime" in dtype:
        data = ["1991-11-20", "2004-12-04", "2016-09-13", None]
    elif dtype == "str" or "float" in dtype:
        data = [1, 2, 3, None]
    else:
        data = [1.0, 0.5, -1.1, np.nan, None]

    sr = Series(data, dtype=dtype)

    expect = DataFrame()
    expect["foo"] = sr
    expect["bar"] = sr
    got = DataFrame({"foo": data, "bar": data}, dtype=dtype)

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
        pytest.param(
            gd.DataFrame(),
            marks=[
                pytest.mark.xfail(
                    reason="_apply_support_method fails on empty dataframes."
                )
            ],
        ),
        pytest.param(
            gd.DataFrame({"a": [0, 1, 2], "b": [1, None, 3]}),
            marks=[
                pytest.mark.xfail(
                    reason="Rowwise ops do not currently support nulls."
                )
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "product", "mean", "var", "std"]
)
def test_rowwise_ops(data, op):
    gdf = data
    pdf = gdf.to_pandas()

    if op in ("var", "std"):
        expected = getattr(pdf, op)(axis=1, ddof=0)
        got = getattr(gdf, op)(axis=1, ddof=0)
    else:
        expected = getattr(pdf, op)(axis=1)
        got = getattr(gdf, op)(axis=1)

    assert_eq(expected, got, check_less_precise=7)


@pytest.mark.parametrize("data", [[5.0, 6.0, 7.0], "single value"])
def test_insert(data):
    pdf = pd.DataFrame.from_dict({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    gdf = DataFrame.from_pandas(pdf)

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

    # pandas insert doesnt support negative indexing
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
        Series([1, 2, 3]),
        Series([1, 2, 3], index=["a", "b", "c"]),
        Series([1, 2, 3], index=["a", "b", "d"]),
        Series([1, 2], index=["a", "b"]),
        Series([1, 2, 3], index=gd.core.index.RangeIndex(0, 3)),
        pytest.param(
            Series([1, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"]),
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

    gdf = DataFrame(data)
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
    "gsr", [Series([1, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"])]
)
def test_df_sr_binop_col_order(gsr, op):
    colnames = [0, 1, 2]
    data = [[0, 2, 5], [3, None, 5], [6, 7, np.nan]]
    data = dict(zip(colnames, data))

    gdf = DataFrame(data)
    pdf = pd.DataFrame.from_dict(data)

    psr = gsr.to_pandas()

    expect = op(pdf, psr).astype("float")
    out = op(gdf, gsr).astype("float")
    got = out[expect.columns]

    assert_eq(expect, got)


@pytest.mark.xfail
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
    assert (
        gdf.B.memory_usage(deep=True, index=False)
        == gdf.B._column.nvstrings.device_memory()
    )

    # Check string index
    assert (
        gdf.set_index("B").index.memory_usage(deep=True)
        == gdf.B._column.nvstrings.device_memory()
    )


@pytest.mark.xfail
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
        ValueError,
        match=(
            "Cannot insert Column of different length into OrderedColumnDict"
        ),
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
