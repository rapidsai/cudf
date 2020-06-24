# Copyright (c) 2018, NVIDIA CORPORATION.

"""
Test related to Index
"""
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import DataFrame
from cudf.core.index import (
    CategoricalIndex,
    DatetimeIndex,
    GenericIndex,
    Int64Index,
    RangeIndex,
    as_index,
)
from cudf.tests.utils import (
    FLOAT_TYPES,
    INTEGER_TYPES,
    NUMERIC_TYPES,
    OTHER_TYPES,
    UNSIGNED_TYPES,
    assert_eq,
)


def test_df_set_index_from_series():
    df = DataFrame()
    df["a"] = list(range(10))
    df["b"] = list(range(0, 20, 2))

    # Check set_index(Series)
    df2 = df.set_index(df["b"])
    assert list(df2.columns) == ["a", "b"]
    sliced_strided = df2.loc[2:6]
    print(sliced_strided)
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_set_index_from_name():
    df = DataFrame()
    df["a"] = list(range(10))
    df["b"] = list(range(0, 20, 2))

    # Check set_index(column_name)
    df2 = df.set_index("b")
    print(df2)
    # 1 less column because 'b' is used as index
    assert list(df2.columns) == ["a"]
    sliced_strided = df2.loc[2:6]
    print(sliced_strided)
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_slice_empty_index():
    df = DataFrame()
    assert isinstance(df.index, RangeIndex)
    assert isinstance(df.index[:1], RangeIndex)
    with pytest.raises(IndexError):
        df.index[1]


def test_index_find_label_range():
    # Monotonic Index
    idx = Int64Index(np.asarray([4, 5, 6, 10]))
    assert idx.find_label_range(4, 6) == (0, 3)
    assert idx.find_label_range(5, 10) == (1, 4)
    assert idx.find_label_range(0, 6) == (0, 3)
    assert idx.find_label_range(4, 11) == (0, 4)

    # Non-monotonic Index
    idx_nm = Int64Index(np.asarray([5, 4, 6, 10]))
    assert idx_nm.find_label_range(4, 6) == (1, 3)
    assert idx_nm.find_label_range(5, 10) == (0, 4)
    # Last value not found
    with pytest.raises(ValueError) as raises:
        idx_nm.find_label_range(0, 6)
    raises.match("value not found")
    # Last value not found
    with pytest.raises(ValueError) as raises:
        idx_nm.find_label_range(4, 11)
    raises.match("value not found")


def test_index_comparision():
    start, stop = 10, 34
    rg = RangeIndex(start, stop)
    gi = Int64Index(np.arange(start, stop))
    assert rg.equals(gi)
    assert gi.equals(rg)
    assert not rg[:-1].equals(gi)
    assert rg[:-1].equals(gi[:-1])


@pytest.mark.parametrize(
    "func", [lambda x: x.min(), lambda x: x.max(), lambda x: x.sum()]
)
def test_reductions(func):
    x = np.asarray([4, 5, 6, 10])
    idx = Int64Index(np.asarray([4, 5, 6, 10]))

    assert func(x) == func(idx)


def test_name():
    idx = Int64Index(np.asarray([4, 5, 6, 10]), name="foo")
    assert idx.name == "foo"


def test_index_immutable():
    start, stop = 10, 34
    rg = RangeIndex(start, stop)
    with pytest.raises(TypeError):
        rg[1] = 5
    gi = Int64Index(np.arange(start, stop))
    with pytest.raises(TypeError):
        gi[1] = 5


def test_categorical_index():
    pdf = pd.DataFrame()
    pdf["a"] = [1, 2, 3]
    pdf["index"] = pd.Categorical(["a", "b", "c"])
    initial_df = DataFrame.from_pandas(pdf)
    pdf = pdf.set_index("index")
    gdf1 = DataFrame.from_pandas(pdf)
    gdf2 = DataFrame()
    gdf2["a"] = [1, 2, 3]
    gdf2["index"] = pd.Categorical(["a", "b", "c"])
    assert_eq(initial_df.index, gdf2.index)
    gdf2 = gdf2.set_index("index")

    assert isinstance(gdf1.index, CategoricalIndex)
    assert_eq(pdf, gdf1)
    assert_eq(pdf.index, gdf1.index)
    assert_eq(pdf.index.codes, gdf1.index.codes.to_array())

    assert isinstance(gdf2.index, CategoricalIndex)
    assert_eq(pdf, gdf2)
    assert_eq(pdf.index, gdf2.index)
    assert_eq(pdf.index.codes, gdf2.index.codes.to_array())


def test_pandas_as_index():
    # Define Pandas Indexes
    pdf_int_index = pd.Int64Index([1, 2, 3, 4, 5])
    pdf_uint_index = pd.UInt64Index([1, 2, 3, 4, 5])
    pdf_float_index = pd.Float64Index([1.0, 2.0, 3.0, 4.0, 5.0])
    pdf_datetime_index = pd.DatetimeIndex(
        [1000000, 2000000, 3000000, 4000000, 5000000]
    )
    pdf_category_index = pd.CategoricalIndex(["a", "b", "c", "b", "a"])

    # Define cudf Indexes
    gdf_int_index = as_index(pdf_int_index)
    gdf_uint_index = as_index(pdf_uint_index)
    gdf_float_index = as_index(pdf_float_index)
    gdf_datetime_index = as_index(pdf_datetime_index)
    gdf_category_index = as_index(pdf_category_index)

    # Check instance types
    assert isinstance(gdf_int_index, GenericIndex)
    assert isinstance(gdf_uint_index, GenericIndex)
    assert isinstance(gdf_float_index, GenericIndex)
    assert isinstance(gdf_datetime_index, DatetimeIndex)
    assert isinstance(gdf_category_index, CategoricalIndex)

    # Check equality
    assert_eq(pdf_int_index, gdf_int_index)
    assert_eq(pdf_uint_index, gdf_uint_index)
    assert_eq(pdf_float_index, gdf_float_index)
    assert_eq(pdf_datetime_index, gdf_datetime_index)
    assert_eq(pdf_category_index, gdf_category_index)

    assert_eq(pdf_category_index.codes, gdf_category_index.codes.to_array())


def test_index_rename():
    pds = pd.Index([1, 2, 3], name="asdf")
    gds = as_index(pds)

    expect = pds.rename("new_name")
    got = gds.rename("new_name")

    assert_eq(expect, got)
    """
    From here on testing recursive creation
    and if name is being handles in recursive creation.
    """
    pds = pd.Index(expect)
    gds = as_index(got)

    assert_eq(pds, gds)

    pds = pd.Index(pds, name="abc")
    gds = as_index(gds, name="abc")
    assert_eq(pds, gds)


def test_index_rename_inplace():
    pds = pd.Index([1, 2, 3], name="asdf")
    gds = as_index(pds)

    # inplace=False should yield a deep copy
    gds_renamed_deep = gds.rename("new_name", inplace=False)

    assert gds_renamed_deep._values.data_ptr != gds._values.data_ptr

    # inplace=True returns none
    expected_ptr = gds._values.data_ptr
    gds.rename("new_name", inplace=True)

    assert expected_ptr == gds._values.data_ptr


def test_index_rename_preserves_arg():
    idx1 = Int64Index([1, 2, 3], name="orig_name")

    # this should be an entirely new object
    idx2 = idx1.rename("new_name", inplace=False)

    assert idx2.name == "new_name"
    assert idx1.name == "orig_name"

    # a new object but referencing the same data
    idx3 = as_index(idx1, name="last_name")

    assert idx3.name == "last_name"
    assert idx1.name == "orig_name"


def test_set_index_as_property():
    cdf = DataFrame()
    col1 = np.arange(10)
    col2 = np.arange(0, 20, 2)
    cdf["a"] = col1
    cdf["b"] = col2

    # Check set_index(Series)
    cdf.index = cdf["b"]

    assert_eq(cdf.index._values.to_array(), col2)

    with pytest.raises(ValueError):
        cdf.index = [list(range(10))]

    idx = pd.Index(np.arange(0, 1000, 100))
    cdf.index = idx
    assert_eq(cdf.index.to_pandas(), idx)

    df = cdf.to_pandas()
    assert_eq(df.index, idx)

    head = cdf.head().to_pandas()
    assert_eq(head.index, idx[:5])


@pytest.mark.parametrize(
    "idx",
    [
        cudf.core.index.RangeIndex(1, 5),
        cudf.core.index.DatetimeIndex(["2001", "2003", "2003"]),
        cudf.core.index.StringIndex(["a", "b", "c"]),
        cudf.core.index.Int64Index([1, 2, 3]),
        cudf.core.index.CategoricalIndex(["a", "b", "c"]),
    ],
)
@pytest.mark.parametrize("deep", [True, False])
def test_index_copy(idx, deep):
    idx_copy = idx.copy(deep=deep)
    assert_eq(idx, idx_copy)
    assert type(idx) == type(idx_copy)


@pytest.mark.parametrize("idx", [[1, None, 3, None, 5]])
def test_index_isna(idx):
    pidx = pd.Index(idx, name="idx")
    gidx = cudf.core.index.Int64Index(idx, name="idx")
    assert_eq(gidx.isna().to_array(), pidx.isna())


@pytest.mark.parametrize("idx", [[1, None, 3, None, 5]])
def test_index_notna(idx):
    pidx = pd.Index(idx, name="idx")
    gidx = cudf.core.index.Int64Index(idx, name="idx")
    assert_eq(gidx.notna().to_array(), pidx.notna())


def test_rangeindex_slice_attr_name():
    start, stop = 0, 10
    rg = RangeIndex(start, stop, name="myindex")
    sliced_rg = rg[0:9]
    assert_eq(rg.name, sliced_rg.name)


def test_from_pandas_str():
    idx = ["a", "b", "c"]
    pidx = pd.Index(idx, name="idx")
    gidx_1 = cudf.core.index.StringIndex(idx, name="idx")
    gidx_2 = cudf.from_pandas(pidx)

    assert_eq(gidx_1, gidx_2)


def test_from_pandas_gen():
    idx = [2, 4, 6]
    pidx = pd.Index(idx, name="idx")
    gidx_1 = cudf.core.index.Int64Index(idx, name="idx")
    gidx_2 = cudf.from_pandas(pidx)

    assert_eq(gidx_1, gidx_2)


def test_index_names():
    idx = cudf.core.index.as_index([1, 2, 3], name="idx")
    assert idx.names == ("idx",)


@pytest.mark.parametrize(
    "data",
    [
        range(0),
        range(1),
        range(0, 1),
        range(0, 5),
        range(1, 10),
        range(1, 10, 1),
        range(-5, 10),
    ],
)
def test_range_index_from_range(data):
    assert_eq(pd.Index(data), cudf.core.index.as_index(data))


@pytest.mark.parametrize(
    "n", [-10, -5, -2, 0, 1, 0, 2, 5, 10],
)
def test_empty_df_head_tail_index(n):
    df = cudf.DataFrame()
    pdf = pd.DataFrame()
    assert_eq(df.head(n).index.values, pdf.head(n).index.values)
    assert_eq(df.tail(n).index.values, pdf.tail(n).index.values)

    df = cudf.DataFrame({"a": [11, 2, 33, 44, 55]})
    pdf = pd.DataFrame({"a": [11, 2, 33, 44, 55]})
    assert_eq(df.head(n).index.values, pdf.head(n).index.values)
    assert_eq(df.tail(n).index.values, pdf.tail(n).index.values)

    df = cudf.DataFrame(index=[1, 2, 3])
    pdf = pd.DataFrame(index=[1, 2, 3])
    assert_eq(df.head(n).index.values, pdf.head(n).index.values)
    assert_eq(df.tail(n).index.values, pdf.tail(n).index.values)


@pytest.mark.parametrize(
    "data,condition,other,error",
    [
        (pd.Index(range(5)), pd.Index(range(5)) > 0, None, None),
        (pd.Index([1, 2, 3]), pd.Index([1, 2, 3]) != 2, None, None),
        (pd.Index(list("abc")), pd.Index(list("abc")) == "c", None, None),
        (
            pd.Index(list("abc")),
            pd.Index(list("abc")) == "c",
            pd.Index(list("xyz")),
            None,
        ),
        (pd.Index(range(5)), pd.Index(range(4)) > 0, None, ValueError),
        (pd.Index(range(5)), pd.Index(range(5)) > 1, 10, None),
        (
            pd.Index(np.arange(10)),
            (pd.Index(np.arange(10)) % 3) == 0,
            -pd.Index(np.arange(10)),
            None,
        ),
        (pd.Index([1, 2, np.nan]), pd.Index([1, 2, np.nan]) == 4, None, None,),
        (pd.Index([1, 2, np.nan]), pd.Index([1, 2, np.nan]) != 4, None, None,),
        (pd.Index([-2, 3, -4, -79]), [True, True, True], None, ValueError,),
        (pd.Index([-2, 3, -4, -79]), [True, True, True, False], None, None,),
        (pd.Index([-2, 3, -4, -79]), [True, True, True, False], 17, None,),
        (pd.Index(list("abcdgh")), pd.Index(list("abcdgh")) != "g", "3", None),
        (
            pd.Index(list("abcdgh")),
            pd.Index(list("abcdg")) != "g",
            "3",
            ValueError,
        ),
        (
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]) != "a",
            "h",
            None,
        ),
        (
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]) != "a",
            "b",
            None,
        ),
        (
            pd.MultiIndex.from_tuples(
                list(
                    zip(
                        *[
                            [
                                "bar",
                                "bar",
                                "baz",
                                "baz",
                                "foo",
                                "foo",
                                "qux",
                                "qux",
                            ],
                            [
                                "one",
                                "two",
                                "one",
                                "two",
                                "one",
                                "two",
                                "one",
                                "two",
                            ],
                        ]
                    )
                )
            ),
            pd.MultiIndex.from_tuples(
                list(
                    zip(
                        *[
                            [
                                "bar",
                                "bar",
                                "baz",
                                "baz",
                                "foo",
                                "foo",
                                "qux",
                                "qux",
                            ],
                            [
                                "one",
                                "two",
                                "one",
                                "two",
                                "one",
                                "two",
                                "one",
                                "two",
                            ],
                        ]
                    )
                )
            )
            != "a",
            None,
            NotImplementedError,
        ),
    ],
)
def test_index_where(data, condition, other, error):
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

    if error is None:
        if pd.api.types.is_categorical_dtype(ps):
            expect = ps.where(ps_condition, other=ps_other)
            got = gs.where(gs_condition, other=gs_other)
            np.testing.assert_array_equal(
                expect.codes, got.codes.fillna(-1).to_array()
            )
            assert tuple(expect.categories) == tuple(got.categories)
        else:
            assert_eq(
                ps.where(ps_condition, other=ps_other)
                .fillna(gs._columns[0].default_na_value())
                .values,
                gs.where(gs_condition, other=gs_other)
                .to_pandas()
                .fillna(gs._columns[0].default_na_value())
                .values,
            )
    else:
        with pytest.raises(error):
            ps.where(ps_condition, other=ps_other)
        with pytest.raises(error):
            gs.where(gs_condition, other=gs_other)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + OTHER_TYPES)
@pytest.mark.parametrize("copy", [True, False])
def test_index_astype(dtype, copy):
    pdi = pd.Index([1, 2, 3])
    gdi = cudf.from_pandas(pdi)

    actual = gdi.astype(dtype=dtype, copy=copy)
    expected = pdi.astype(dtype=dtype, copy=copy)

    assert_eq(expected, actual)
    assert_eq(pdi, gdi)


@pytest.mark.parametrize(
    "data",
    [
        [1, 10, 2, 100, -10],
        ["z", "x", "a", "c", "b"],
        [-10.2, 100.1, -100.2, 0.0, 0.23],
    ],
)
def test_index_argsort(data):
    pdi = pd.Index(data)
    gdi = cudf.from_pandas(pdi)

    assert_eq(pdi.argsort(), gdi.argsort())


@pytest.mark.parametrize(
    "data",
    [
        [1, 10, 2, 100, -10],
        ["z", "x", "a", "c", "b"],
        [-10.2, 100.1, -100.2, 0.0, 0.23],
    ],
)
def test_index_to_series(data):
    pdi = pd.Index(data)
    gdi = cudf.from_pandas(pdi)

    assert_eq(pdi.to_series(), gdi.to_series())


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]
)
@pytest.mark.parametrize("name", [1, "a", None])
def test_index_basic(data, dtype, name):
    pdi = pd.Index(data, dtype=dtype, name=name)
    gdi = cudf.Index(data, dtype=dtype, name=name)

    assert_eq(pdi, gdi)


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("name", [1, "a", None])
@pytest.mark.parametrize("dtype", INTEGER_TYPES)
def test_integer_index_apis(data, name, dtype):
    pindex = pd.Int64Index(data, dtype=dtype, name=name)
    # Int8Index
    gindex = cudf.Int8Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("int8")

    # Int16Index
    gindex = cudf.Int16Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("int16")

    # Int32Index
    gindex = cudf.Int32Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("int32")

    # Int64Index
    gindex = cudf.Int64Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("int64")


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("name", [1, "a", None])
@pytest.mark.parametrize("dtype", UNSIGNED_TYPES)
def test_unisgned_integer_index_apis(data, name, dtype):
    pindex = pd.UInt64Index(data, dtype=dtype, name=name)
    # UInt8Index
    gindex = cudf.UInt8Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("uint8")

    # UInt16Index
    gindex = cudf.UInt16Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("uint16")

    # UInt32Index
    gindex = cudf.UInt32Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("uint32")

    # UInt64Index
    gindex = cudf.UInt64Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("uint64")


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("name", [1, "a", None])
@pytest.mark.parametrize("dtype", FLOAT_TYPES)
def test_float_index_apis(data, name, dtype):
    pindex = pd.Float64Index(data, dtype=dtype, name=name)
    # Float32Index
    gindex = cudf.Float32Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("float32")

    # Float64Index
    gindex = cudf.Float64Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("float64")


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("categories", [[1, 2], None])
@pytest.mark.parametrize(
    "dtype",
    [
        pd.CategoricalDtype([1, 2, 3], ordered=True),
        pd.CategoricalDtype([1, 2, 3], ordered=False),
        None,
    ],
)
@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("name", [1, "a", None])
def test_categorical_index_basic(data, categories, dtype, ordered, name):

    # can't have both dtype and categories/ordered
    if dtype is not None:
        categories = None
        ordered = None
    pindex = pd.CategoricalIndex(
        data=data,
        categories=categories,
        dtype=dtype,
        ordered=ordered,
        name=name,
    )
    gindex = CategoricalIndex(
        data=data,
        categories=categories,
        dtype=dtype,
        ordered=ordered,
        name=name,
    )

    assert_eq(pindex, gindex)
