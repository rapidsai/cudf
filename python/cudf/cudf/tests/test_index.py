# Copyright (c) 2018-2024, NVIDIA CORPORATION.

"""
Test related to Index
"""

import datetime
import operator
import re

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.api.types import is_bool_dtype
from cudf.core.index import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    RangeIndex,
    as_index,
)
from cudf.testing._utils import (
    ALL_TYPES,
    FLOAT_TYPES,
    NUMERIC_TYPES,
    OTHER_TYPES,
    SERIES_OR_INDEX_NAMES,
    SIGNED_INTEGER_TYPES,
    UNSIGNED_TYPES,
    assert_column_memory_eq,
    assert_column_memory_ne,
    assert_eq,
    assert_exceptions_equal,
    expect_warning_if,
)
from cudf.utils.utils import search_range


def test_df_set_index_from_series():
    df = cudf.DataFrame()
    df["a"] = list(range(10))
    df["b"] = list(range(0, 20, 2))

    # Check set_index(Series)
    df2 = df.set_index(df["b"])
    assert list(df2.columns) == ["a", "b"]
    sliced_strided = df2.loc[2:6]
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_set_index_from_name():
    df = cudf.DataFrame()
    df["a"] = list(range(10))
    df["b"] = list(range(0, 20, 2))

    # Check set_index(column_name)
    df2 = df.set_index("b")
    # 1 less column because 'b' is used as index
    assert list(df2.columns) == ["a"]
    sliced_strided = df2.loc[2:6]
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_slice_empty_index():
    df = cudf.DataFrame()
    assert isinstance(df.index, RangeIndex)
    assert isinstance(df.index[:1], RangeIndex)
    with pytest.raises(IndexError):
        df.index[1]


def test_index_find_label_range_genericindex():
    # Monotonic Index
    idx = cudf.Index(np.asarray([4, 5, 6, 10]))
    assert idx.find_label_range(slice(4, 6)) == slice(0, 3, 1)
    assert idx.find_label_range(slice(5, 10)) == slice(1, 4, 1)
    assert idx.find_label_range(slice(0, 6)) == slice(0, 3, 1)
    assert idx.find_label_range(slice(4, 11)) == slice(0, 4, 1)

    # Non-monotonic Index
    idx_nm = cudf.Index(np.asarray([5, 4, 6, 10]))
    assert idx_nm.find_label_range(slice(4, 6)) == slice(1, 3, 1)
    assert idx_nm.find_label_range(slice(5, 10)) == slice(0, 4, 1)
    # Last value not found
    with pytest.raises(KeyError) as raises:
        idx_nm.find_label_range(slice(0, 6))
    raises.match("not in index")
    # Last value not found
    with pytest.raises(KeyError) as raises:
        idx_nm.find_label_range(slice(4, 11))
    raises.match("not in index")


def test_index_find_label_range_rangeindex():
    """Cudf specific"""
    # step > 0
    # 3, 8, 13, 18
    ridx = RangeIndex(3, 20, 5)
    assert ridx.find_label_range(slice(3, 8)) == slice(0, 2, 1)
    assert ridx.find_label_range(slice(0, 7)) == slice(0, 1, 1)
    assert ridx.find_label_range(slice(3, 19)) == slice(0, 4, 1)
    assert ridx.find_label_range(slice(2, 21)) == slice(0, 4, 1)

    # step < 0
    # 20, 15, 10, 5
    ridx = RangeIndex(20, 3, -5)
    assert ridx.find_label_range(slice(15, 10)) == slice(1, 3, 1)
    assert ridx.find_label_range(slice(10, 15, -1)) == slice(2, 0, -1)
    assert ridx.find_label_range(slice(10, 0)) == slice(2, 4, 1)
    assert ridx.find_label_range(slice(30, 13)) == slice(0, 2, 1)
    assert ridx.find_label_range(slice(30, 0)) == slice(0, 4, 1)


def test_index_comparision():
    start, stop = 10, 34
    rg = cudf.RangeIndex(start, stop)
    gi = cudf.Index(np.arange(start, stop))
    assert rg.equals(gi)
    assert gi.equals(rg)
    assert not rg[:-1].equals(gi)
    assert rg[:-1].equals(gi[:-1])


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x.min(),
        lambda x: x.max(),
        lambda x: x.any(),
        lambda x: x.all(),
    ],
)
def test_reductions(func):
    x = np.asarray([4, 5, 6, 10])
    idx = cudf.Index(np.asarray([4, 5, 6, 10]))

    assert func(x) == func(idx)


def test_name():
    idx = cudf.Index(np.asarray([4, 5, 6, 10]), name="foo")
    assert idx.name == "foo"


def test_index_immutable():
    start, stop = 10, 34
    rg = RangeIndex(start, stop)
    with pytest.raises(TypeError):
        rg[1] = 5
    gi = cudf.Index(np.arange(start, stop))
    with pytest.raises(TypeError):
        gi[1] = 5


def test_categorical_index():
    pdf = pd.DataFrame()
    pdf["a"] = [1, 2, 3]
    pdf["index"] = pd.Categorical(["a", "b", "c"])
    initial_df = cudf.from_pandas(pdf)
    pdf = pdf.set_index("index")
    gdf1 = cudf.from_pandas(pdf)
    gdf2 = cudf.DataFrame()
    gdf2["a"] = [1, 2, 3]
    gdf2["index"] = pd.Categorical(["a", "b", "c"])
    assert_eq(initial_df.index, gdf2.index)
    gdf2 = gdf2.set_index("index")

    assert isinstance(gdf1.index, CategoricalIndex)
    assert_eq(pdf, gdf1)
    assert_eq(pdf.index, gdf1.index)
    assert_eq(
        pdf.index.codes,
        gdf1.index.codes.astype(pdf.index.codes.dtype).to_numpy(),
    )

    assert isinstance(gdf2.index, CategoricalIndex)
    assert_eq(pdf, gdf2)
    assert_eq(pdf.index, gdf2.index)
    assert_eq(
        pdf.index.codes,
        gdf2.index.codes.astype(pdf.index.codes.dtype).to_numpy(),
    )


def test_pandas_as_index():
    # Define Pandas Indexes
    pdf_int_index = pd.Index([1, 2, 3, 4, 5])
    pdf_uint_index = pd.Index([1, 2, 3, 4, 5])
    pdf_float_index = pd.Index([1.0, 2.0, 3.0, 4.0, 5.0])
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
    assert isinstance(gdf_int_index, Index)
    assert isinstance(gdf_uint_index, Index)
    assert isinstance(gdf_float_index, Index)
    assert isinstance(gdf_datetime_index, DatetimeIndex)
    assert isinstance(gdf_category_index, CategoricalIndex)

    # Check equality
    assert_eq(pdf_int_index, gdf_int_index)
    assert_eq(pdf_uint_index, gdf_uint_index)
    assert_eq(pdf_float_index, gdf_float_index)
    assert_eq(pdf_datetime_index, gdf_datetime_index)
    assert_eq(pdf_category_index, gdf_category_index)

    assert_eq(
        pdf_category_index.codes,
        gdf_category_index.codes.astype(
            pdf_category_index.codes.dtype
        ).to_numpy(),
    )


@pytest.mark.parametrize("initial_name", SERIES_OR_INDEX_NAMES)
@pytest.mark.parametrize("name", SERIES_OR_INDEX_NAMES)
def test_index_rename(initial_name, name):
    pds = pd.Index([1, 2, 3], name=initial_name)
    gds = as_index(pds)

    assert_eq(pds, gds)

    expect = pds.rename(name)
    got = gds.rename(name)

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
    idx1 = cudf.Index([1, 2, 3], name="orig_name")

    # this should be an entirely new object
    idx2 = idx1.rename("new_name", inplace=False)

    assert idx2.name == "new_name"
    assert idx1.name == "orig_name"

    # a new object but referencing the same data
    idx3 = as_index(idx1, name="last_name")

    assert idx3.name == "last_name"
    assert idx1.name == "orig_name"


def test_set_index_as_property():
    cdf = cudf.DataFrame()
    col1 = np.arange(10)
    col2 = np.arange(0, 20, 2)
    cdf["a"] = col1
    cdf["b"] = col2

    # Check set_index(Series)
    cdf.index = cdf["b"]

    assert_eq(cdf.index.to_numpy(), col2)

    with pytest.raises(ValueError):
        cdf.index = [list(range(10))]

    idx = pd.Index(np.arange(0, 1000, 100))
    cdf.index = idx
    assert_eq(cdf.index.to_pandas(), idx)

    df = cdf.to_pandas()
    assert_eq(df.index, idx)

    head = cdf.head().to_pandas()
    assert_eq(head.index, idx[:5])


@pytest.mark.parametrize("name", ["x"])
def test_index_copy_range(name, deep=True):
    cidx = cudf.RangeIndex(1, 5)
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep)
    cidx_copy = cidx.copy(name=name, deep=deep)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
def test_index_copy_datetime(name, deep=True):
    cidx = cudf.DatetimeIndex(["2001", "2002", "2003"])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep)
    cidx_copy = cidx.copy(name=name, deep=deep)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
def test_index_copy_string(name, deep=True):
    cidx = cudf.Index(["a", "b", "c"])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep)
    cidx_copy = cidx.copy(name=name, deep=deep)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
def test_index_copy_integer(name, deep=True):
    """Test for NumericIndex Copy Casts"""
    cidx = cudf.Index([1, 2, 3])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep)
    cidx_copy = cidx.copy(name=name, deep=deep)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
def test_index_copy_float(name, deep=True):
    """Test for NumericIndex Copy Casts"""
    cidx = cudf.Index([1.0, 2.0, 3.0])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep)
    cidx_copy = cidx.copy(name=name, deep=deep)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
def test_index_copy_category(name, deep=True):
    cidx = cudf.core.index.CategoricalIndex([1, 2, 3])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep)
    cidx_copy = cidx.copy(name=name, deep=deep)

    assert_column_memory_ne(cidx._values, cidx_copy._values)
    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize(
    "idx",
    [
        cudf.DatetimeIndex(["2001", "2002", "2003"]),
        cudf.Index(["a", "b", "c"]),
        cudf.Index([1, 2, 3]),
        cudf.Index([1.0, 2.0, 3.0]),
        cudf.CategoricalIndex([1, 2, 3]),
        cudf.CategoricalIndex(["a", "b", "c"]),
    ],
)
@pytest.mark.parametrize("copy_on_write", [True, False])
def test_index_copy_deep(idx, deep, copy_on_write):
    """Test if deep copy creates a new instance for device data."""
    idx_copy = idx.copy(deep=deep)
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", copy_on_write)
    if (
        isinstance(idx._values, cudf.core.column.StringColumn)
        or not deep
        or (cudf.get_option("copy_on_write") and not deep)
    ):
        # StringColumn is immutable hence, deep copies of a
        # Index with string dtype will share the same StringColumn.

        # When `copy_on_write` is turned on, Index objects will
        # have unique column object but they all point to same
        # data pointers.
        assert_column_memory_eq(idx._values, idx_copy._values)
    else:
        assert_column_memory_ne(idx._values, idx_copy._values)
    cudf.set_option("copy_on_write", original_cow_setting)


@pytest.mark.parametrize("idx", [[1, None, 3, None, 5]])
def test_index_isna(idx):
    pidx = pd.Index(idx, name="idx")
    gidx = cudf.Index(idx, name="idx")
    assert_eq(gidx.isna(), pidx.isna())


@pytest.mark.parametrize("idx", [[1, None, 3, None, 5]])
def test_index_notna(idx):
    pidx = pd.Index(idx, name="idx")
    gidx = cudf.Index(idx, name="idx")
    assert_eq(gidx.notna(), pidx.notna())


def test_rangeindex_slice_attr_name():
    start, stop = 0, 10
    rg = RangeIndex(start, stop, name="myindex")
    sliced_rg = rg[0:9]
    assert_eq(rg.name, sliced_rg.name)


def test_from_pandas_str():
    idx = ["a", "b", "c"]
    pidx = pd.Index(idx, name="idx")
    gidx_1 = cudf.Index(idx, name="idx")
    gidx_2 = cudf.from_pandas(pidx)

    assert_eq(gidx_1, gidx_2)


def test_from_pandas_gen():
    idx = [2, 4, 6]
    pidx = pd.Index(idx, name="idx")
    gidx_1 = cudf.Index(idx, name="idx")
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
        range(1, 10, 3),
        range(10, 1, -3),
        range(-5, 10),
    ],
)
def test_range_index_from_range(data):
    assert_eq(pd.Index(data), cudf.Index(data))


@pytest.mark.parametrize(
    "n",
    [-10, -5, -2, 0, 1, 0, 2, 5, 10],
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
        (
            pd.Index(range(5)),
            pd.Index(range(5)) > 1,
            10,
            None,
        ),
        (
            pd.Index(np.arange(10)),
            (pd.Index(np.arange(10)) % 3) == 0,
            -pd.Index(np.arange(10)),
            None,
        ),
        (
            pd.Index([1, 2, np.nan]),
            pd.Index([1, 2, np.nan]) == 4,
            None,
            None,
        ),
        (
            pd.Index([1, 2, np.nan]),
            pd.Index([1, 2, np.nan]) != 4,
            None,
            None,
        ),
        (
            pd.Index([-2, 3, -4, -79]),
            [True, True, True],
            None,
            ValueError,
        ),
        (
            pd.Index([-2, 3, -4, -79]),
            [True, True, True, False],
            None,
            None,
        ),
        (
            pd.Index([-2, 3, -4, -79]),
            [True, True, True, False],
            17,
            None,
        ),
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
            "a",
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
        if hasattr(ps, "dtype") and isinstance(ps.dtype, pd.CategoricalDtype):
            expect = ps.where(ps_condition, other=ps_other)
            got = gs.where(gs_condition, other=gs_other)
            np.testing.assert_array_equal(
                expect.codes,
                got.codes.astype(expect.codes.dtype).fillna(-1).to_numpy(),
            )
            assert_eq(expect.categories, got.categories)
        else:
            assert_eq(
                ps.where(ps_condition, other=ps_other),
                gs.where(gs_condition, other=gs_other).to_pandas(),
            )
    else:
        assert_exceptions_equal(
            lfunc=ps.where,
            rfunc=gs.where,
            lfunc_args_and_kwargs=([ps_condition], {"other": ps_other}),
            rfunc_args_and_kwargs=([gs_condition], {"other": gs_other}),
        )


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
        pd.Index([1, 10, 2, 100, -10], name="abc"),
        pd.Index(["z", "x", "a", "c", "b"]),
        pd.Index(["z", "x", "a", "c", "b"], dtype="category"),
        pd.Index(
            [-10.2, 100.1, -100.2, 0.0, 0.23], name="this is a float index"
        ),
        pd.Index([102, 1001, 1002, 0.0, 23], dtype="datetime64[ns]"),
        pd.Index([13240.2, 1001, 100.2, 0.0, 23], dtype="datetime64[ns]"),
        pd.RangeIndex(0, 10, 1),
        pd.RangeIndex(0, -100, -2),
        pd.Index([-10.2, 100.1, -100.2, 0.0, 23], dtype="timedelta64[ns]"),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("return_indexer", [True, False])
def test_index_sort_values(data, ascending, return_indexer):
    pdi = data
    gdi = cudf.from_pandas(pdi)

    expected = pdi.sort_values(
        ascending=ascending, return_indexer=return_indexer
    )
    actual = gdi.sort_values(
        ascending=ascending, return_indexer=return_indexer
    )

    if return_indexer:
        expected_indexer = expected[1]
        actual_indexer = actual[1]

        assert_eq(expected_indexer, actual_indexer)

        expected = expected[0]
        actual = actual[0]

    assert_eq(expected, actual)


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


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5, 6],
        [4, 5, 6, 10, 20, 30],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        ["5", "6", "2", "a", "b", "c"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 5.0, 6.0, 0.0, 1.3],
        ["ab", "cd", "ef"],
        pd.Series(["1", "2", "a", "3", None], dtype="category"),
        range(0, 10),
        [],
        [1, 1, 2, 2],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 5, 6],
        [4, 5, 6, 10, 20, 30],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        ["5", "6", "2", "a", "b", "c"],
        ["ab", "ef", None],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 5.0, 6.0, 0.0, 1.3],
        range(2, 4),
        pd.Series(["1", "a", "3", None], dtype="category"),
        [],
        [2],
    ],
)
@pytest.mark.parametrize("sort", [None, False, True])
@pytest.mark.parametrize(
    "name_data,name_other",
    [("abc", "c"), (None, "abc"), ("abc", pd.NA), ("abc", "abc")],
)
def test_index_difference(data, other, sort, name_data, name_other):
    pd_data = pd.Index(data, name=name_data)
    pd_other = pd.Index(other, name=name_other)

    gd_data = cudf.from_pandas(pd_data)
    gd_other = cudf.from_pandas(pd_other)

    expected = pd_data.difference(pd_other, sort=sort)
    actual = gd_data.difference(gd_other, sort=sort)

    assert_eq(expected, actual)


@pytest.mark.parametrize("other", ["a", 1, None])
def test_index_difference_invalid_inputs(other):
    pdi = pd.Index([1, 2, 3])
    gdi = cudf.Index([1, 2, 3])

    assert_exceptions_equal(
        pdi.difference,
        gdi.difference,
        ([other], {}),
        ([other], {}),
    )


def test_index_difference_sort_error():
    pdi = pd.Index([1, 2, 3])
    gdi = cudf.Index([1, 2, 3])

    assert_exceptions_equal(
        pdi.difference,
        gdi.difference,
        ([pdi], {"sort": "A"}),
        ([gdi], {"sort": "A"}),
    )


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["a"],
        ["b", "c", "d"],
        [1],
        [2, 3, 4],
        [],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["a"],
        [],
        ["b", "c", "d"],
        [1],
        [2, 3, 4],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
def test_index_equals(data, other):
    pd_data = pd.Index(data)
    pd_other = pd.Index(other)

    gd_data = cudf.core.index.as_index(data)
    gd_other = cudf.core.index.as_index(other)

    expected = pd_data.equals(pd_other)
    actual = gd_data.equals(gd_other)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["a"],
        ["b", "c", "d"],
        [1],
        [2, 3, 4],
        [],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["a"],
        ["b", "c", "d"],
        [1],
        [2, 3, 4],
        [],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
def test_index_categories_equal(data, other):
    pd_data = pd.Index(data).astype("category")
    pd_other = pd.Index(other)

    gd_data = cudf.core.index.as_index(data).astype("category")
    gd_other = cudf.core.index.as_index(other)

    expected = pd_data.equals(pd_other)
    actual = gd_data.equals(gd_other)
    assert_eq(expected, actual)

    expected = pd_other.equals(pd_data)
    actual = gd_other.equals(gd_data)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["a"],
        ["b", "c", "d"],
        [1],
        [2, 3, 4],
        [],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["a"],
        ["b", "c", "d"],
        [1],
        [2, 3, 4],
        [],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
def test_index_equal_misc(data, other):
    pd_data = pd.Index(data)
    pd_other = other

    gd_data = cudf.core.index.as_index(data)
    gd_other = other

    expected = pd_data.equals(pd_other)
    actual = gd_data.equals(gd_other)
    assert_eq(expected, actual)

    expected = pd_data.equals(np.array(pd_other))
    actual = gd_data.equals(np.array(gd_other))
    assert_eq(expected, actual)

    expected = pd_data.equals(pd.Series(pd_other))
    actual = gd_data.equals(cudf.Series(gd_other))
    assert_eq(expected, actual)

    expected = pd_data.astype("category").equals(pd_other)
    actual = gd_data.astype("category").equals(gd_other)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["a"],
        ["b", "c", "d"],
        [1],
        [2, 3, 4],
        [],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        ["1", "2", "3", "4", "5", "6"],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["a"],
        ["b", "c", "d"],
        [1],
        [2, 3, 4],
        [],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
def test_index_append(data, other):
    pd_data = pd.Index(data)
    pd_other = pd.Index(other)

    gd_data = cudf.Index(data)
    gd_other = cudf.Index(other)

    if cudf.utils.dtypes.is_mixed_with_object_dtype(gd_data, gd_other):
        gd_data = gd_data.astype("str")
        gd_other = gd_other.astype("str")

    with expect_warning_if(
        (len(data) == 0 or len(other) == 0) and pd_data.dtype != pd_other.dtype
    ):
        expected = pd_data.append(pd_other)
    with expect_warning_if(
        (len(data) == 0 or len(other) == 0) and gd_data.dtype != gd_other.dtype
    ):
        actual = gd_data.append(gd_other)
    if len(data) == 0 and len(other) == 0:
        # Pandas default dtype to "object" for empty list
        # cudf default dtype to "float" for empty list
        assert_eq(expected, actual.astype("str"))
    elif actual.dtype == "object":
        assert_eq(expected.astype("str"), actual)
    else:
        assert_eq(expected, actual)


def test_index_empty_append_name_conflict():
    empty = cudf.Index([], name="foo")
    non_empty = cudf.Index([1], name="bar")
    expected = cudf.Index([1])

    with pytest.warns(FutureWarning):
        result = non_empty.append(empty)
    assert_eq(result, expected)

    with pytest.warns(FutureWarning):
        result = empty.append(non_empty)
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1],
        [2, 3, 4],
        [10.0],
        [1100.112, 2323.2322, 2323.2322],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        ["1", "2", "3", "4", "5", "6"],
        ["a"],
        ["b", "c", "d"],
        ["abcd", "defgh", "werty", "poiu"],
    ],
)
def test_index_append_error(data, other):
    gd_data = cudf.core.index.as_index(data)
    gd_other = cudf.core.index.as_index(other)

    got_dtype = (
        gd_other.dtype
        if gd_data.dtype == np.dtype("object")
        else gd_data.dtype
    )
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"cudf does not support appending an Index of "
            f"dtype `{np.dtype('object')}` with an Index "
            f"of dtype `{got_dtype}`, please type-cast "
            f"either one of them to same dtypes."
        ),
    ):
        gd_data.append(gd_other)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"cudf does not support appending an Index of "
            f"dtype `{np.dtype('object')}` with an Index "
            f"of dtype `{got_dtype}`, please type-cast "
            f"either one of them to same dtypes."
        ),
    ):
        gd_other.append(gd_data)

    sr = gd_other.to_series()

    assert_exceptions_equal(
        lfunc=gd_data.to_pandas().append,
        rfunc=gd_data.append,
        lfunc_args_and_kwargs=([[sr.to_pandas()]],),
        rfunc_args_and_kwargs=([[sr]],),
    )


@pytest.mark.parametrize(
    "data,other",
    [
        (
            pd.Index([1, 2, 3, 4, 5, 6]),
            [
                pd.Index([1, 2, 3, 4, 5, 6]),
                pd.Index([1, 2, 3, 4, 5, 6, 10]),
                pd.Index([]),
            ],
        ),
        (
            pd.Index([]),
            [
                pd.Index([1, 2, 3, 4, 5, 6]),
                pd.Index([1, 2, 3, 4, 5, 6, 10]),
                pd.Index([1, 4, 5, 6]),
            ],
        ),
        (
            pd.Index([10, 20, 30, 40, 50, 60]),
            [
                pd.Index([10, 20, 30, 40, 50, 60]),
                pd.Index([10, 20, 30]),
                pd.Index([40, 50, 60]),
                pd.Index([10, 60]),
                pd.Index([60]),
            ],
        ),
        (
            pd.Index([]),
            [
                pd.Index([10, 20, 30, 40, 50, 60]),
                pd.Index([10, 20, 30]),
                pd.Index([40, 50, 60]),
                pd.Index([10, 60]),
                pd.Index([60]),
            ],
        ),
        (
            pd.Index(["1", "2", "3", "4", "5", "6"]),
            [
                pd.Index(["1", "2", "3", "4", "5", "6"]),
                pd.Index(["1", "2", "3"]),
                pd.Index(["6"]),
                pd.Index(["1", "6"]),
            ],
        ),
        (
            pd.Index([]),
            [
                pd.Index(["1", "2", "3", "4", "5", "6"]),
                pd.Index(["1", "2", "3"]),
                pd.Index(["6"]),
                pd.Index(["1", "6"]),
            ],
        ),
        (
            pd.Index([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [
                pd.Index([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                pd.Index([1.0, 6.0]),
                pd.Index([]),
                pd.Index([6.0]),
            ],
        ),
        (
            pd.Index([]),
            [
                pd.Index([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                pd.Index([1.0, 6.0]),
                pd.Index([1.0, 2.0, 6.0]),
                pd.Index([6.0]),
            ],
        ),
        (
            pd.Index(["a"]),
            [
                pd.Index(["a"]),
                pd.Index(["a", "b", "c"]),
                pd.Index(["c"]),
                pd.Index(["d"]),
                pd.Index(["ae", "hello", "world"]),
            ],
        ),
        (
            pd.Index([]),
            [
                pd.Index(["a"]),
                pd.Index(["a", "b", "c"]),
                pd.Index(["c"]),
                pd.Index(["d"]),
                pd.Index(["ae", "hello", "world"]),
                pd.Index([]),
            ],
        ),
    ],
)
def test_index_append_list(data, other):
    pd_data = data
    pd_other = other

    gd_data = cudf.from_pandas(data)
    gd_other = [cudf.from_pandas(i) for i in other]

    with expect_warning_if(
        (len(data) == 0 or any(len(d) == 0 for d in other))
        and (any(d.dtype != data.dtype for d in other))
    ):
        expected = pd_data.append(pd_other)
    with expect_warning_if(
        (len(data) == 0 or any(len(d) == 0 for d in other))
        and (any(d.dtype != data.dtype for d in other))
    ):
        actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


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
@pytest.mark.parametrize("dtype", SIGNED_INTEGER_TYPES)
def test_integer_index_apis(data, name, dtype):
    pindex = pd.Index(data, dtype=dtype, name=name)
    gindex = cudf.Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == dtype


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("name", [1, "a", None])
@pytest.mark.parametrize("dtype", UNSIGNED_TYPES)
def test_unsigned_integer_index_apis(data, name, dtype):
    pindex = pd.Index(data, dtype=dtype, name=name)
    gindex = cudf.Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == dtype


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("name", [1, "a", None])
@pytest.mark.parametrize("dtype", FLOAT_TYPES)
def test_float_index_apis(data, name, dtype):
    pindex = pd.Index(data, dtype=dtype, name=name)
    gindex = cudf.Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == dtype


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


@pytest.mark.parametrize(
    "data",
    [
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
            names=("number", "color"),
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], ["yellow", "violet", "pink", "white"]],
            names=("number1", "color2"),
        ),
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
            names=("number", "color"),
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], ["yellow", "violet", "pink", "white"]],
            names=("number1", "color2"),
        ),
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
        ),
    ],
)
def test_multiindex_append(data, other):
    pdi = data
    other_pd = other

    gdi = cudf.from_pandas(data)
    other_gd = cudf.from_pandas(other)

    expected = pdi.append(other_pd)
    actual = gdi.append(other_gd)

    assert_eq(expected, actual)


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]
)
def test_index_empty(data, dtype):
    pdi = pd.Index(data, dtype=dtype)
    gdi = cudf.Index(data, dtype=dtype)

    assert_eq(pdi.empty, gdi.empty)


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]
)
def test_index_size(data, dtype):
    pdi = pd.Index(data, dtype=dtype)
    gdi = cudf.Index(data, dtype=dtype)

    assert_eq(pdi.size, gdi.size)


@pytest.mark.parametrize("data", [[1, 2, 3, 1, 2, 3, 4], [], [1], [1, 2, 3]])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]
)
def test_index_drop_duplicates(data, dtype):
    pdi = pd.Index(data, dtype=dtype)
    gdi = cudf.Index(data, dtype=dtype)

    assert_eq(pdi.drop_duplicates(), gdi.drop_duplicates())


def test_dropna_bad_how():
    with pytest.raises(ValueError):
        cudf.Index([1]).dropna(how="foo")


@pytest.mark.parametrize("data", [[1, 2, 3, 1, 2, 3, 4], []])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]
)
def test_index_tolist(data, dtype):
    gdi = cudf.Index(data, dtype=dtype)

    with pytest.raises(
        TypeError,
        match=re.escape(
            r"cuDF does not support conversion to host memory "
            r"via the `tolist()` method. Consider using "
            r"`.to_arrow().to_pylist()` to construct a Python list."
        ),
    ):
        gdi.tolist()


@pytest.mark.parametrize("data", [[], [1], [1, 2, 3]])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]
)
def test_index_iter_error(data, dtype):
    gdi = cudf.Index(data, dtype=dtype)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gdi.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gdi)


@pytest.mark.parametrize("data", [[], [1], [1, 2, 3, 4, 5]])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]
)
def test_index_values_host(data, dtype):
    gdi = cudf.Index(data, dtype=dtype)
    pdi = pd.Index(data, dtype=dtype)

    np.testing.assert_array_equal(gdi.values_host, pdi.values)


@pytest.mark.parametrize(
    "data,fill_value",
    [
        ([1, 2, 3, 1, None, None], 1),
        ([None, None, 3.2, 1, None, None], 10.0),
        ([None, "a", "3.2", "z", None, None], "helloworld"),
        (pd.Series(["a", "b", None], dtype="category"), "b"),
        (pd.Series([None, None, 1.0], dtype="category"), 1.0),
        (
            np.array([1, 2, 3, None], dtype="datetime64[s]"),
            np.datetime64("2005-02-25"),
        ),
        (
            np.array(
                [None, None, 122, 3242234, None, 6237846],
                dtype="datetime64[ms]",
            ),
            np.datetime64("2005-02-25"),
        ),
    ],
)
def test_index_fillna(data, fill_value):
    pdi = pd.Index(data)
    gdi = cudf.Index(data)

    assert_eq(
        pdi.fillna(fill_value), gdi.fillna(fill_value), exact=False
    )  # Int64 v/s Float64


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 1, None, None],
        [None, None, 3.2, 1, None, None],
        [None, "a", "3.2", "z", None, None],
        pd.Series(["a", "b", None], dtype="category"),
        np.array([1, 2, 3, None], dtype="datetime64[s]"),
    ],
)
def test_index_to_arrow(data):
    pdi = pd.Index(data)
    gdi = cudf.Index(data)

    expected_arrow_array = pa.Array.from_pandas(pdi)
    got_arrow_array = gdi.to_arrow()

    assert_eq(expected_arrow_array, got_arrow_array)


@pytest.mark.parametrize(
    "data",
    [
        [None, None, 3.2, 1, None, None],
        [None, "a", "3.2", "z", None, None],
        pd.Series(["a", "b", None], dtype="category"),
        np.array([1, 2, 3, None], dtype="datetime64[s]"),
    ],
)
def test_index_from_arrow(data):
    pdi = pd.Index(data)

    arrow_array = pa.Array.from_pandas(pdi)
    expected_index = pd.Index(arrow_array.to_pandas())
    gdi = cudf.Index.from_arrow(arrow_array)

    assert_eq(expected_index, gdi)


def test_multiindex_to_arrow():
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 1, 2, 3],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "c": np.array([1, 2, 3, None, 5], dtype="datetime64[s]"),
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    pdf["a"] = pdf["a"].astype("category")
    df = cudf.from_pandas(pdf)
    gdi = cudf.MultiIndex.from_frame(df)

    expected = pa.Table.from_pandas(pdf)
    got = gdi.to_arrow()

    assert_eq(expected, got)


def test_multiindex_from_arrow():
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 1, 2, 3],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "c": np.array([1, 2, 3, None, 5], dtype="datetime64[s]"),
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    pdf["a"] = pdf["a"].astype("category")
    ptb = pa.Table.from_pandas(pdf)
    gdi = cudf.MultiIndex.from_arrow(ptb)
    pdi = pd.MultiIndex.from_frame(pdf)

    assert_eq(pdi, gdi)


def test_index_equals_categories():
    lhs = cudf.CategoricalIndex(
        ["a", "b", "c", "b", "a"], categories=["a", "b", "c"]
    )
    rhs = cudf.CategoricalIndex(
        ["a", "b", "c", "b", "a"], categories=["a", "b", "c", "_"]
    )

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)


def test_rangeindex_arg_validation():
    with pytest.raises(TypeError):
        RangeIndex("1")

    with pytest.raises(TypeError):
        RangeIndex(1, "2")

    with pytest.raises(TypeError):
        RangeIndex(1, 3, "1")

    with pytest.raises(ValueError):
        RangeIndex(1, dtype="float64")

    with pytest.raises(ValueError):
        RangeIndex(1, dtype="uint64")


def test_rangeindex_name_not_hashable():
    with pytest.raises(ValueError):
        RangeIndex(range(2), name=["foo"])

    with pytest.raises(ValueError):
        RangeIndex(range(2)).copy(name=["foo"])


def test_index_rangeindex_search_range():
    # step > 0
    ridx = RangeIndex(-13, 17, 4)
    ri = ridx._range
    for i in range(len(ridx)):
        assert i == search_range(ridx[i], ri, side="left")
        assert i + 1 == search_range(ridx[i], ri, side="right")


@pytest.mark.parametrize(
    "rge",
    [(1, 10, 1), (1, 10, 3), (10, -17, -1), (10, -17, -3)],
)
def test_index_rangeindex_get_item_basic(rge):
    pridx = pd.RangeIndex(*rge)
    gridx = cudf.RangeIndex(*rge)

    for i in range(-len(pridx), len(pridx)):
        assert pridx[i] == gridx[i]


@pytest.mark.parametrize(
    "rge",
    [(1, 10, 3), (10, 1, -3)],
)
def test_index_rangeindex_get_item_out_of_bounds(rge):
    gridx = cudf.RangeIndex(*rge)
    with pytest.raises(IndexError):
        _ = gridx[4]


@pytest.mark.parametrize(
    "rge",
    [(10, 1, 1), (-17, 10, -3)],
)
def test_index_rangeindex_get_item_null_range(rge):
    gridx = cudf.RangeIndex(*rge)

    with pytest.raises(IndexError):
        gridx[0]


@pytest.mark.parametrize(
    "rge", [(-17, 21, 2), (21, -17, -3), (0, 0, 1), (0, 1, -3), (10, 0, 5)]
)
@pytest.mark.parametrize(
    "sl",
    [
        slice(1, 7, 1),
        slice(1, 7, 2),
        slice(-1, 7, 1),
        slice(-1, 7, 2),
        slice(-3, 7, 2),
        slice(7, 1, -2),
        slice(7, -3, -2),
        slice(None, None, 1),
        slice(0, None, 2),
        slice(0, None, 3),
        slice(0, 0, 3),
    ],
)
def test_index_rangeindex_get_item_slices(rge, sl):
    pridx = pd.RangeIndex(*rge)
    gridx = cudf.RangeIndex(*rge)

    assert_eq(pridx[sl], gridx[sl])


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 2, 3]),
        pd.Index(["abc", "def", "ghi"]),
        pd.RangeIndex(0, 10, 1),
        pd.Index([0.324, 0.234, 1.3], name="abc"),
    ],
)
@pytest.mark.parametrize("names", [None, "a", "new name", ["another name"]])
@pytest.mark.parametrize("inplace", [True, False])
def test_index_set_names(idx, names, inplace):
    pi = idx.copy()
    gi = cudf.from_pandas(idx)

    expected = pi.set_names(names=names, inplace=inplace)
    actual = gi.set_names(names=names, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize("idx", [pd.Index([1, 2, 3], name="abc")])
@pytest.mark.parametrize("level", [1, [0], "abc"])
@pytest.mark.parametrize("names", [None, "a"])
def test_index_set_names_error(idx, level, names):
    pi = idx.copy()
    gi = cudf.from_pandas(idx)

    assert_exceptions_equal(
        lfunc=pi.set_names,
        rfunc=gi.set_names,
        lfunc_args_and_kwargs=([], {"names": names, "level": level}),
        rfunc_args_and_kwargs=([], {"names": names, "level": level}),
    )


@pytest.mark.parametrize(
    "idx",
    [pd.Index([1, 3, 6]), pd.Index([6, 1, 3])],  # monotonic  # non-monotonic
)
@pytest.mark.parametrize("key", [list(range(0, 8))])
@pytest.mark.parametrize("method", [None, "ffill", "bfill", "nearest"])
def test_get_indexer_single_unique_numeric(idx, key, method):
    pi = idx
    gi = cudf.from_pandas(pi)

    if (
        # `method` only applicable to monotonic index
        not pi.is_monotonic_increasing and method is not None
    ):
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
        )
    else:
        expected = pi.get_indexer(key, method=method)
        got = gi.get_indexer(key, method=method)

        assert_eq(expected, got)

        with cudf.option_context("mode.pandas_compatible", True):
            got = gi.get_indexer(key, method=method)
        assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "idx",
    [pd.RangeIndex(3, 100, 4)],
)
@pytest.mark.parametrize(
    "key",
    [
        list(range(1, 20, 3)),
        list(range(20, 35, 3)),
        list(range(35, 77, 3)),
        list(range(77, 110, 3)),
    ],
)
@pytest.mark.parametrize("method", [None, "ffill", "bfill", "nearest"])
@pytest.mark.parametrize("tolerance", [None, 0, 1, 13, 20])
def test_get_indexer_rangeindex(idx, key, method, tolerance):
    pi = idx
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(
        key, method=method, tolerance=None if method is None else tolerance
    )
    got = gi.get_indexer(
        key, method=method, tolerance=None if method is None else tolerance
    )

    assert_eq(expected, got)

    with cudf.option_context("mode.pandas_compatible", True):
        got = gi.get_indexer(
            key, method=method, tolerance=None if method is None else tolerance
        )
    assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "idx",
    [pd.RangeIndex(3, 100, 4)],
)
@pytest.mark.parametrize("key", list(range(1, 110, 3)))
def test_get_loc_rangeindex(idx, key):
    pi = idx
    gi = cudf.from_pandas(pi)
    if (
        (key not in pi)
        # Get key before the first element is KeyError
        or (key < pi.start)
        # Get key after the last element is KeyError
        or (key >= pi.stop)
    ):
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 3, 3, 6]),  # monotonic increasing
        pd.Index([6, 1, 3, 3]),  # non-monotonic
        pd.Index([4, 3, 2, 1, 0]),  # monotonic decreasing
    ],
)
@pytest.mark.parametrize("key", [0, 3, 6, 7, 4])
def test_get_loc_duplicate_numeric(idx, key):
    pi = idx
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([-1, 2, 3, 6]),  # monotonic
        pd.Index([6, 1, 3, 4]),  # non-monotonic
    ],
)
@pytest.mark.parametrize("key", [[0, 3, 1], [6, 7]])
@pytest.mark.parametrize("method", [None, "ffill", "bfill", "nearest"])
@pytest.mark.parametrize("tolerance", [None, 1, 2])
def test_get_indexer_single_duplicate_numeric(idx, key, method, tolerance):
    pi = idx
    gi = cudf.from_pandas(pi)

    if not pi.is_monotonic_increasing and method is not None:
        assert_exceptions_equal(
            lfunc=pi.get_indexer,
            rfunc=gi.get_indexer,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
        )
    else:
        expected = pi.get_indexer(
            key, method=method, tolerance=None if method is None else tolerance
        )
        got = gi.get_indexer(
            key, method=method, tolerance=None if method is None else tolerance
        )

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx", [pd.Index(["b", "f", "m", "q"]), pd.Index(["m", "f", "b", "q"])]
)
@pytest.mark.parametrize("key", ["a", "f", "n", "z"])
def test_get_loc_single_unique_string(idx, key):
    pi = idx
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx", [pd.Index(["b", "f", "m", "q"]), pd.Index(["m", "f", "b", "q"])]
)
@pytest.mark.parametrize("key", [["a", "f", "n", "z"], ["p", "p", "b"]])
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_single_unique_string(idx, key, method):
    pi = idx
    gi = cudf.from_pandas(pi)

    if not pi.is_monotonic_increasing and method is not None:
        assert_exceptions_equal(
            lfunc=pi.get_indexer,
            rfunc=gi.get_indexer,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
        )
    else:
        expected = pi.get_indexer(key, method=method)
        got = gi.get_indexer(key, method=method)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx", [pd.Index(["b", "m", "m", "q"]), pd.Index(["m", "f", "m", "q"])]
)
@pytest.mark.parametrize("key", ["a", "f", "n", "z"])
def test_get_loc_single_duplicate_string(idx, key):
    pi = idx
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx", [pd.Index(["b", "m", "m", "q"]), pd.Index(["a", "f", "m", "q"])]
)
@pytest.mark.parametrize("key", [["a"], ["f", "n", "z"]])
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_single_duplicate_string(idx, key, method):
    pi = idx
    gi = cudf.from_pandas(pi)

    if (
        # `method` only applicable to monotonic index
        (not pi.is_monotonic_increasing and method is not None)
        or not pi.is_unique
    ):
        assert_exceptions_equal(
            lfunc=pi.get_indexer,
            rfunc=gi.get_indexer,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
        )
    else:
        expected = pi.get_indexer(key, method=method)
        got = gi.get_indexer(key, method=method)

        assert_eq(expected, got)

        with cudf.option_context("mode.pandas_compatible", True):
            got = gi.get_indexer(key, method=method)

        assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "idx",
    [
        pd.MultiIndex.from_tuples(
            [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 1, 1), (2, 2, 1)]
        ),
        pd.MultiIndex.from_tuples(
            [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 2), (2, 2, 1), (1, 1, 1)]
        ),
        pd.MultiIndex.from_tuples(
            [(1, 1, 1), (1, 1, 2), (1, 1, 2), (1, 2, 3), (2, 1, 1), (2, 2, 1)]
        ),
    ],
)
@pytest.mark.parametrize("key", [1, (1, 2), (1, 2, 3), (2, 1, 1), (9, 9, 9)])
def test_get_loc_multi_numeric(idx, key):
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx",
    [
        pd.MultiIndex.from_tuples(
            [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 1, 1), (2, 2, 1)]
        ),
        pd.MultiIndex.from_tuples(
            [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 2), (2, 2, 1), (1, 1, 1)]
        ),
        pd.MultiIndex.from_tuples(
            [(1, 1, 1), (1, 1, 2), (1, 1, 24), (1, 2, 3), (2, 1, 1), (2, 2, 1)]
        ),
    ],
)
@pytest.mark.parametrize("key", [[(1, 2, 3)], [(9, 9, 9)]])
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_multi_numeric(idx, key, method):
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(key, method=method)
    got = gi.get_indexer(key, method=method)

    assert_eq(expected, got)

    with cudf.option_context("mode.pandas_compatible", True):
        got = gi.get_indexer(key, method=method)

    assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "idx",
    [
        pd.MultiIndex.from_tuples(
            [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 1), (1, 1, 1), (2, 2, 1)]
        )
    ],
)
@pytest.mark.parametrize(
    "key, result",
    [
        (1, slice(1, 5, 1)),  # deviates
        ((1, 2), slice(1, 3, 1)),
        ((1, 2, 3), slice(1, 2, None)),
        ((2, 1, 1), slice(0, 1, None)),
        ((9, 9, 9), None),
    ],
)
def test_get_loc_multi_numeric_deviate(idx, key, result):
    pi = idx
    gi = cudf.from_pandas(pi)

    with expect_warning_if(
        isinstance(key, tuple), pd.errors.PerformanceWarning
    ):
        key_flag = key not in pi

    if key_flag:
        with expect_warning_if(
            isinstance(key, tuple), pd.errors.PerformanceWarning
        ):
            assert_exceptions_equal(
                lfunc=pi.get_loc,
                rfunc=gi.get_loc,
                lfunc_args_and_kwargs=([], {"key": key}),
                rfunc_args_and_kwargs=([], {"key": key}),
            )
    else:
        expected = result
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "key",
    [
        ((1, 2, 3),),
        ((2, 1, 1),),
        ((9, 9, 9),),
    ],
)
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_multi_numeric_deviate(key, method):
    pi = pd.MultiIndex.from_tuples(
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 10), (1, 1, 1), (2, 2, 1)]
    ).sort_values()
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(key, method=method)
    got = gi.get_indexer(key, method=method)

    assert_eq(expected, got)


@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_get_indexer_multi_error(method):
    pi = pd.MultiIndex.from_tuples(
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 10), (1, 1, 1), (2, 2, 1)]
    )
    gi = cudf.from_pandas(pi)

    assert_exceptions_equal(
        pi.get_indexer,
        gi.get_indexer,
        lfunc_args_and_kwargs=(
            [],
            {"target": ((1, 2, 3),), "method": method},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"target": ((1, 2, 3),), "method": method},
        ),
    )


@pytest.mark.parametrize(
    "idx",
    [
        pd.MultiIndex.from_tuples(
            [
                ("a", "a", "a"),
                ("a", "a", "b"),
                ("a", "b", "a"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("b", "c", "a"),
            ]
        ),
        pd.MultiIndex.from_tuples(
            [
                ("a", "a", "b"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("a", "a", "a"),
                ("a", "b", "a"),
                ("b", "c", "a"),
            ]
        ),
        pd.MultiIndex.from_tuples(
            [
                ("a", "a", "a"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("a", "a", "b"),
                ("a", "b", "a"),
                ("b", "c", "a"),
            ]
        ),
        pd.MultiIndex.from_tuples(
            [
                ("a", "a", "a"),
                ("a", "a", "b"),
                ("a", "a", "b"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("b", "c", "a"),
            ]
        ),
        pd.MultiIndex.from_tuples(
            [
                ("a", "a", "b"),
                ("b", "a", "a"),
                ("b", "a", "a"),
                ("a", "a", "a"),
                ("a", "b", "a"),
                ("b", "c", "a"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "key", ["a", ("a", "a"), ("a", "b", "c"), ("b", "c", "a"), ("z", "z", "z")]
)
def test_get_loc_multi_string(idx, key):
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx",
    [
        pd.MultiIndex.from_tuples(
            [
                ("a", "a", "a"),
                ("a", "a", "b"),
                ("a", "b", "a"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("b", "c", "a"),
            ]
        ),
        pd.MultiIndex.from_tuples(
            [
                ("a", "a", "b"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("a", "a", "a"),
                ("a", "b", "a"),
                ("b", "c", "a"),
            ]
        ),
        pd.MultiIndex.from_tuples(
            [
                ("a", "a", "a"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("a", "a", "b"),
                ("a", "b", "a"),
                ("b", "c", "a"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "key", [[("a", "b", "c"), ("b", "c", "a")], [("z", "z", "z")]]
)
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_multi_string(idx, key, method):
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(key, method=method)
    got = gi.get_indexer(key, method=method)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx1",
    [
        lambda: cudf.Index(["a", "b", "c"]),
        lambda: cudf.RangeIndex(0, 10),
        lambda: cudf.Index([1, 2, 3], dtype="category"),
        lambda: cudf.Index(["a", "b", "c", "d"], dtype="category"),
        lambda: cudf.MultiIndex.from_tuples(
            [
                ("a", "a", "a"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("a", "a", "b"),
                ("a", "b", "a"),
                ("b", "c", "a"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "idx2",
    [
        lambda: cudf.Index(["a", "b", "c"]),
        lambda: cudf.RangeIndex(0, 10),
        lambda: cudf.Index([1, 2, 3], dtype="category"),
        lambda: cudf.Index(["a", "b", "c", "d"], dtype="category"),
    ],
)
def test_get_indexer_invalid(idx1, idx2):
    idx1 = idx1()
    idx2 = idx2()
    assert_eq(
        idx1.get_indexer(idx2), idx1.to_pandas().get_indexer(idx2.to_pandas())
    )


@pytest.mark.parametrize(
    "objs",
    [
        [pd.RangeIndex(0, 10), pd.RangeIndex(10, 20)],
        [pd.RangeIndex(10, 20), pd.RangeIndex(22, 40), pd.RangeIndex(50, 60)],
        [pd.RangeIndex(10, 20, 2), pd.RangeIndex(20, 40, 2)],
    ],
)
def test_range_index_concat(objs):
    cudf_objs = [cudf.from_pandas(obj) for obj in objs]

    actual = cudf.concat(cudf_objs)

    expected = objs[0]
    for obj in objs[1:]:
        expected = expected.append(obj)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (pd.RangeIndex(0, 10), pd.RangeIndex(3, 7)),
        (pd.RangeIndex(0, 10), pd.RangeIndex(10, 20)),
        (pd.RangeIndex(0, 10, 2), pd.RangeIndex(1, 5, 3)),
        (pd.RangeIndex(1, 5, 3), pd.RangeIndex(0, 10, 2)),
        (pd.RangeIndex(1, 10, 3), pd.RangeIndex(1, 5, 2)),
        (pd.RangeIndex(1, 5, 2), pd.RangeIndex(1, 10, 3)),
        (pd.RangeIndex(1, 100, 3), pd.RangeIndex(1, 50, 3)),
        (pd.RangeIndex(1, 100, 3), pd.RangeIndex(1, 50, 6)),
        (pd.RangeIndex(1, 100, 6), pd.RangeIndex(1, 50, 3)),
        (pd.RangeIndex(0, 10, name="a"), pd.RangeIndex(90, 100, name="b")),
        (pd.Index([0, 1, 2, 30], name="a"), pd.Index([90, 100])),
        (pd.Index([0, 1, 2, 30], name="a"), [90, 100]),
        (pd.Index([0, 1, 2, 30]), pd.Index([0, 10, 1.0, 11])),
        (pd.Index(["a", "b", "c", "d", "c"]), pd.Index(["a", "c", "z"])),
        (
            pd.IntervalIndex.from_tuples([(0, 2), (0, 2), (2, 4)]),
            pd.IntervalIndex.from_tuples([(0, 2), (2, 4)]),
        ),
        (pd.RangeIndex(0, 10), pd.Index([8, 1, 2, 4])),
        (pd.Index([8, 1, 2, 4], name="a"), pd.Index([8, 1, 2, 4], name="b")),
        (
            pd.Index([8, 1, 2, 4], name="a"),
            pd.Index([], name="b", dtype="int64"),
        ),
        (pd.Index([], dtype="int64", name="a"), pd.Index([10, 12], name="b")),
        (pd.Index([True, True, True], name="a"), pd.Index([], dtype="bool")),
        (
            pd.Index([True, True, True]),
            pd.Index([False, True], dtype="bool", name="b"),
        ),
    ],
)
@pytest.mark.parametrize("sort", [None, False, True])
def test_union_index(idx1, idx2, sort):
    expected = idx1.union(idx2, sort=sort)

    idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.Index) else idx1
    idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.Index) else idx2

    actual = idx1.union(idx2, sort=sort)

    assert_eq(expected, actual)


def test_union_bool_with_other():
    idx1 = cudf.Index([True, True, True])
    idx2 = cudf.Index([0, 1], name="b")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(cudf.errors.MixedTypeError):
            idx1.union(idx2)


@pytest.mark.parametrize("dtype1", ["int8", "int32", "int32"])
@pytest.mark.parametrize("dtype2", ["uint32", "uint64"])
def test_union_unsigned_vs_signed(dtype1, dtype2):
    idx1 = cudf.Index([10, 20, 30], dtype=dtype1)
    idx2 = cudf.Index([0, 1], dtype=dtype2)
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(cudf.errors.MixedTypeError):
            idx1.union(idx2)


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (pd.RangeIndex(0, 10), pd.RangeIndex(3, 7)),
        (pd.RangeIndex(0, 10), pd.RangeIndex(-10, 20)),
        (pd.RangeIndex(0, 10, name="a"), pd.RangeIndex(90, 100, name="b")),
        (pd.Index([0, 1, 2, 30], name=pd.NA), pd.Index([30, 0, 90, 100])),
        (pd.Index([0, 1, 2, 30], name="a"), [90, 100]),
        (pd.Index([0, 1, 2, 30]), pd.Index([0, 10, 1.0, 11])),
        (
            pd.Index(["a", "b", "c", "d", "c"]),
            pd.Index(["a", "c", "z"], name="abc"),
        ),
        (
            pd.Index(["a", "b", "c", "d", "c"]),
            pd.Index(["a", "b", "c", "d", "c"]),
        ),
        (pd.Index([True, False, True, True]), pd.Index([10, 11, 12, 0, 1, 2])),
        (pd.Index([True, False, True, True]), pd.Index([True, True])),
        (pd.RangeIndex(0, 10, name="a"), pd.Index([5, 6, 7], name="b")),
        (pd.Index(["a", "b", "c"], dtype="category"), pd.Index(["a", "b"])),
        (pd.Index(["a", "b", "c"], dtype="category"), pd.Index([1, 2, 3])),
        (pd.Index([0, 1, 2], dtype="category"), pd.RangeIndex(0, 10)),
        (pd.Index(["a", "b", "c"], name="abc"), []),
        (pd.Index([], name="abc"), pd.RangeIndex(0, 4)),
        (pd.Index([1, 2, 3]), pd.Index([1, 2], dtype="category")),
        (pd.Index([]), pd.Index([1, 2], dtype="category")),
    ],
)
@pytest.mark.parametrize("sort", [None, False, True])
@pytest.mark.parametrize("pandas_compatible", [True, False])
def test_intersection_index(idx1, idx2, sort, pandas_compatible):
    expected = idx1.intersection(idx2, sort=sort)

    with cudf.option_context("mode.pandas_compatible", pandas_compatible):
        idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.Index) else idx1
        idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.Index) else idx2

        actual = idx1.intersection(idx2, sort=sort)

        # TODO: Resolve the bool vs ints mixed issue
        # once pandas has a direction on this issue
        # https://github.com/pandas-dev/pandas/issues/44000
        assert_eq(
            expected,
            actual,
            exact=False
            if (is_bool_dtype(idx1.dtype) and not is_bool_dtype(idx2.dtype))
            or (not is_bool_dtype(idx1.dtype) or is_bool_dtype(idx2.dtype))
            else True,
        )


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        ["a", "v", "d"],
        [234.243, 2432.3, None],
        [True, False, True],
        pd.Series(["a", " ", "v"], dtype="category"),
        pd.IntervalIndex.from_breaks([0, 1, 2, 3]),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        "is_numeric",
        "is_boolean",
        "is_integer",
        "is_floating",
        "is_object",
        "is_categorical",
        "is_interval",
    ],
)
def test_index_type_methods(data, func):
    pidx = pd.Index(data)
    gidx = cudf.from_pandas(pidx)

    with pytest.warns(FutureWarning):
        expected = getattr(pidx, func)()
    with pytest.warns(FutureWarning):
        actual = getattr(gidx, func)()

    if gidx.dtype == np.dtype("bool") and func == "is_object":
        assert_eq(False, actual)
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "s", "ms", "us", "ns"]
)
def test_index_datetime_ceil(resolution):
    cuidx = cudf.DatetimeIndex([1000000, 2000000, 3000000, 4000000, 5000000])
    pidx = cuidx.to_pandas()

    pidx_ceil = pidx.ceil(resolution)
    cuidx_ceil = cuidx.ceil(resolution)

    assert_eq(pidx_ceil, cuidx_ceil)


@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "s", "ms", "us", "ns"]
)
def test_index_datetime_floor(resolution):
    cuidx = cudf.DatetimeIndex([1000000, 2000000, 3000000, 4000000, 5000000])
    pidx = cuidx.to_pandas()

    pidx_floor = pidx.floor(resolution)
    cuidx_floor = cuidx.floor(resolution)

    assert_eq(pidx_floor, cuidx_floor)


@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "s", "ms", "us", "ns"]
)
def test_index_datetime_round(resolution):
    cuidx = cudf.DatetimeIndex([1000000, 2000000, 3000000, 4000000, 5000000])
    pidx = cuidx.to_pandas()

    pidx_floor = pidx.round(resolution)
    cuidx_floor = cuidx.round(resolution)

    assert_eq(pidx_floor, cuidx_floor)


@pytest.mark.parametrize(
    "data,nan_idx,NA_idx",
    [([1, 2, 3, None], None, 3), ([2, 3, np.nan, None], 2, 3)],
)
@pytest.mark.parametrize("nan_as_null", [True, False])
def test_index_nan_as_null(data, nan_idx, NA_idx, nan_as_null):
    idx = cudf.Index(data, nan_as_null=nan_as_null)

    if nan_as_null:
        if nan_idx is not None:
            assert idx[nan_idx] is cudf.NA
    else:
        if nan_idx is not None:
            assert np.isnan(idx[nan_idx])

    if NA_idx is not None:
        assert idx[NA_idx] is cudf.NA


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([]),
        pd.Index(["a", "b", "c", "d", "e"]),
        pd.Index([0, None, 9]),
        pd.date_range("2019-01-01", periods=3),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        [0, 19, 13],
        ["2019-01-01 04:00:00", "2019-01-01 06:00:00", "2018-03-02 10:00:00"],
    ],
)
def test_isin_index(index, values):
    pidx = index
    gidx = cudf.Index.from_pandas(pidx)

    is_dt_str = (
        next(iter(values), None) == "2019-01-01 04:00:00"
        and len(pidx)
        and pidx.dtype.kind == "M"
    )
    with expect_warning_if(is_dt_str):
        got = gidx.isin(values)
    with expect_warning_if(is_dt_str):
        expected = pidx.isin(values)

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
        pd.MultiIndex.from_product(
            [[0, 1], ["red", "blue", "green"]], names=("number", "color")
        ),
    ],
)
@pytest.mark.parametrize(
    "values,level,err",
    [
        ([(1, "red"), (2, "blue"), (0, "green")], None, None),
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
    gmdx = cudf.from_pandas(data)

    if err is None:
        expected = pmdx.isin(values, level=level)
        if isinstance(values, pd.MultiIndex):
            values = cudf.from_pandas(values)
        got = gmdx.isin(values, level=level)

        assert_eq(got, expected)
    else:
        assert_exceptions_equal(
            lfunc=pmdx.isin,
            rfunc=gmdx.isin,
            lfunc_args_and_kwargs=([values], {"level": level}),
            rfunc_args_and_kwargs=([values], {"level": level}),
            check_exception_type=False,
        )


range_data = [
    range(np.random.randint(0, 100)),
    range(9, 12, 2),
    range(20, 30),
    range(100, 1000, 10),
    range(0, 10, -2),
    range(0, -10, 2),
    range(0, -10, -2),
]


@pytest.fixture(params=range_data)
def rangeindex(request):
    """Create a cudf RangeIndex of different `nrows`"""
    return RangeIndex(request.param)


@pytest.mark.parametrize(
    "func",
    ["nunique", "min", "max", "any", "values"],
)
def test_rangeindex_methods(rangeindex, func):
    gidx = rangeindex
    pidx = gidx.to_pandas()

    if func == "values":
        expected = pidx.values
        actual = gidx.values
    else:
        expected = getattr(pidx, func)()
        actual = getattr(gidx, func)()

    assert_eq(expected, actual)


def test_index_constructor_integer(default_integer_bitwidth):
    got = cudf.Index([1, 2, 3])
    expect = cudf.Index([1, 2, 3], dtype=f"int{default_integer_bitwidth}")

    assert_eq(expect, got)


def test_index_constructor_float(default_float_bitwidth):
    got = cudf.Index([1.0, 2.0, 3.0])
    expect = cudf.Index(
        [1.0, 2.0, 3.0], dtype=f"float{default_float_bitwidth}"
    )

    assert_eq(expect, got)


def test_rangeindex_union_default_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for union operation.
    idx1 = cudf.RangeIndex(0, 2)
    idx2 = cudf.RangeIndex(5, 6)

    expected = cudf.Index([0, 1, 5], dtype=f"int{default_integer_bitwidth}")
    actual = idx1.union(idx2)

    assert_eq(expected, actual)


def test_rangeindex_intersection_default_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for intersection operation.
    idx1 = cudf.RangeIndex(0, 100)
    # Intersecting two RangeIndex will _always_ result in a RangeIndex, use
    # regular index here to force materializing.
    idx2 = cudf.Index([50, 102])

    expected = cudf.Index([50], dtype=f"int{default_integer_bitwidth}")
    actual = idx1.intersection(idx2)

    assert_eq(expected, actual)


def test_rangeindex_take_default_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for take operation.
    idx = cudf.RangeIndex(0, 100)
    actual = idx.take([0, 3, 7, 62])
    expected = cudf.Index(
        [0, 3, 7, 62], dtype=f"int{default_integer_bitwidth}"
    )
    assert_eq(expected, actual)


def test_rangeindex_apply_boolean_mask_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for apply boolean mask operation.
    idx = cudf.RangeIndex(0, 8)
    mask = [True, True, True, False, False, False, True, False]
    actual = idx[mask]
    expected = cudf.Index([0, 1, 2, 6], dtype=f"int{default_integer_bitwidth}")
    assert_eq(expected, actual)


def test_rangeindex_repeat_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for repeat operation.
    idx = cudf.RangeIndex(0, 3)
    actual = idx.repeat(3)
    expected = cudf.Index(
        [0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=f"int{default_integer_bitwidth}"
    )
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "op, expected, expected_kind",
    [
        (lambda idx: 2**idx, [2, 4, 8, 16], "int"),
        (lambda idx: idx**2, [1, 4, 9, 16], "int"),
        (lambda idx: idx / 2, [0.5, 1, 1.5, 2], "float"),
        (lambda idx: 2 / idx, [2, 1, 2 / 3, 0.5], "float"),
        (lambda idx: idx % 3, [1, 2, 0, 1], "int"),
        (lambda idx: 3 % idx, [0, 1, 0, 3], "int"),
    ],
)
def test_rangeindex_binops_user_option(
    op, expected, expected_kind, default_integer_bitwidth
):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for binary operation.
    idx = cudf.RangeIndex(1, 5)
    actual = op(idx)
    expected = cudf.Index(
        expected, dtype=f"{expected_kind}{default_integer_bitwidth}"
    )
    assert_eq(
        expected,
        actual,
    )


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_rangeindex_binop_diff_names_none(op):
    idx1 = cudf.RangeIndex(10, 13, name="foo")
    idx2 = cudf.RangeIndex(13, 16, name="bar")
    result = op(idx1, idx2)
    expected = op(idx1.to_pandas(), idx2.to_pandas())
    assert_eq(result, expected)
    assert result.name is None


def test_rangeindex_join_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for join.
    idx1 = cudf.RangeIndex(0, 10, name="a")
    idx2 = cudf.RangeIndex(5, 15, name="b")

    actual = idx1.join(idx2, how="inner", sort=True)
    expected = idx1.to_pandas().join(idx2.to_pandas(), how="inner", sort=True)
    assert actual.dtype == cudf.dtype(f"int{default_integer_bitwidth}")
    # exact=False to ignore dtype comparison,
    # because `default_integer_bitwidth` is cudf only option
    assert_eq(expected, actual, exact=False)


def test_rangeindex_where_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for where operation.
    idx = cudf.RangeIndex(0, 10)
    mask = [True, False, True, False, True, False, True, False, True, False]
    actual = idx.where(mask, -1)
    expected = cudf.Index(
        [0, -1, 2, -1, 4, -1, 6, -1, 8, -1],
        dtype=f"int{default_integer_bitwidth}",
    )
    assert_eq(expected, actual)


def test_rangeindex_append_return_rangeindex():
    idx = cudf.RangeIndex(0, 10)
    result = idx.append([])
    assert_eq(idx, result)

    result = idx.append(cudf.Index([10]))
    expected = cudf.RangeIndex(0, 11)
    assert_eq(result, expected)


index_data = [
    range(np.random.randint(0, 100)),
    range(0, 10, -2),
    range(0, -10, 2),
    range(0, -10, -2),
    range(0, 1),
    [1, 2, 3, 1, None, None],
    [None, None, 3.2, 1, None, None],
    [None, "a", "3.2", "z", None, None],
    pd.Series(["a", "b", None], dtype="category"),
    np.array([1, 2, 3, None], dtype="datetime64[s]"),
]


@pytest.fixture(params=index_data)
def index(request):
    """Create a cudf Index of different dtypes"""
    return cudf.Index(request.param)


@pytest.mark.parametrize(
    "func",
    [
        "to_series",
        "isna",
        "notna",
        "append",
    ],
)
def test_index_methods(index, func):
    gidx = index
    pidx = gidx.to_pandas()

    if func == "append":
        expected = pidx.append(other=pidx)
        actual = gidx.append(other=gidx)
    else:
        expected = getattr(pidx, func)()
        actual = getattr(gidx, func)()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx, values",
    [
        (range(100, 1000, 10), [200, 600, 800]),
        ([None, "a", "3.2", "z", None, None], ["a", "z"]),
        (pd.Series(["a", "b", None], dtype="category"), [10, None]),
    ],
)
def test_index_isin_values(idx, values):
    gidx = cudf.Index(idx)
    pidx = gidx.to_pandas()

    actual = gidx.isin(values)
    expected = pidx.isin(values)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx, scalar",
    [
        (range(0, -10, -2), -4),
        ([None, "a", "3.2", "z", None, None], "x"),
        (pd.Series(["a", "b", None], dtype="category"), 10),
    ],
)
def test_index_isin_scalar_values(idx, scalar):
    gidx = cudf.Index(idx)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"only list-like objects are allowed to be passed "
            f"to isin(), you passed a {type(scalar).__name__}"
        ),
    ):
        gidx.isin(scalar)


def test_index_any():
    gidx = cudf.Index([1, 2, 3])
    pidx = gidx.to_pandas()

    assert_eq(pidx.any(), gidx.any())


def test_index_values():
    gidx = cudf.Index([1, 2, 3])
    pidx = gidx.to_pandas()

    assert_eq(pidx.values, gidx.values)


def test_index_null_values():
    gidx = cudf.Index([1.0, None, 3, 0, None])
    with pytest.raises(ValueError):
        gidx.values


def test_index_error_list_index():
    s = cudf.Series([[1, 2], [2], [4]])
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Unsupported column type passed to create an "
            "Index: <class 'cudf.core.column.lists.ListColumn'>"
        ),
    ):
        cudf.Index(s)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        pytest.param(
            [np.nan, 10, 15, 16],
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/49818"
            ),
        ),
        range(0, 10),
        [np.nan, None, 10, 20],
        ["ab", "zx", "pq"],
        ["ab", "zx", None, "pq"],
    ],
)
def test_index_hasnans(data):
    gs = cudf.Index(data, nan_as_null=False)
    if isinstance(gs, cudf.RangeIndex):
        with pytest.raises(NotImplementedError):
            gs.to_pandas(nullable=True)
    else:
        ps = gs.to_pandas(nullable=True)
        # Check type to avoid mixing Python bool and NumPy bool
        assert isinstance(gs.hasnans, bool)
        assert gs.hasnans == ps.hasnans


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 1, 1, 3, 2, 3],
        [np.nan, 10, 15, 16, np.nan, 10, 16],
        range(0, 10),
        ["ab", "zx", None, "pq", "ab", None, "zx", None],
    ],
)
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_index_duplicated(data, keep):
    gs = cudf.Index(data)
    ps = gs.to_pandas()

    expected = ps.duplicated(keep=keep)
    actual = gs.duplicated(keep=keep)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,expected_dtype",
    [
        ([10, 11, 12], pd.Int64Dtype()),
        ([0.1, 10.2, 12.3], pd.Float64Dtype()),
        (["abc", None, "def"], pd.StringDtype()),
    ],
)
def test_index_to_pandas_nullable(data, expected_dtype):
    gi = cudf.Index(data)
    pi = gi.to_pandas(nullable=True)
    expected = pd.Index(data, dtype=expected_dtype)

    assert_eq(pi, expected)


class TestIndexScalarGetItem:
    @pytest.fixture(
        params=[range(1, 10, 2), [1, 2, 3], ["a", "b", "c"], [1.5, 2.5, 3.5]]
    )
    def index_values(self, request):
        return request.param

    @pytest.fixture(params=[int, np.int8, np.int32, np.int64])
    def i(self, request):
        return request.param(1)

    def test_scalar_getitem(self, index_values, i):
        index = cudf.Index(index_values)

        assert not isinstance(index[i], cudf.Index)
        assert index[i] == index_values[i]
        assert_eq(index, index.to_pandas())


@pytest.mark.parametrize(
    "data",
    [
        [
            pd.Timestamp("1970-01-01 00:00:00.000000001"),
            pd.Timestamp("1970-01-01 00:00:00.000000002"),
            12,
            20,
        ],
        [
            pd.Timedelta(10),
            pd.Timedelta(20),
            12,
            20,
        ],
        [1, 2, 3, 4],
    ],
)
def test_index_mixed_dtype_error(data):
    pi = pd.Index(data, dtype="object")
    with pytest.raises(TypeError):
        cudf.Index(pi)


@pytest.mark.parametrize("cls", [pd.DatetimeIndex, pd.TimedeltaIndex])
def test_index_date_duration_freq_error(cls):
    s = cls([1, 2, 3], freq="infer")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.Index(s)


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_index_getitem_time_duration(dtype):
    gidx = cudf.Index([1, 2, 3, 4, None], dtype=dtype)
    pidx = gidx.to_pandas()
    with cudf.option_context("mode.pandas_compatible", True):
        for i in range(len(gidx)):
            if i == 4:
                assert gidx[i] is pidx[i]
            else:
                assert_eq(gidx[i], pidx[i])


@pytest.mark.parametrize("dtype", ALL_TYPES)
def test_index_empty_from_pandas(dtype):
    pidx = pd.Index([], dtype=dtype)
    gidx = cudf.from_pandas(pidx)

    assert_eq(pidx, gidx)


def test_empty_index_init():
    pidx = pd.Index([])
    gidx = cudf.Index([])

    assert_eq(pidx, gidx)


@pytest.mark.parametrize(
    "data", [[1, 2, 3], ["ab", "cd", "e", None], range(0, 10)]
)
@pytest.mark.parametrize("data_name", [None, 1, "abc"])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("name", [None, no_default, 1, "abc"])
def test_index_to_frame(data, data_name, index, name):
    pidx = pd.Index(data, name=data_name)
    gidx = cudf.from_pandas(pidx)

    expected = pidx.to_frame(index=index, name=name)
    actual = gidx.to_frame(index=index, name=name)

    assert_eq(expected, actual)


@pytest.mark.parametrize("data", [[1, 2, 3], range(0, 10)])
@pytest.mark.parametrize("dtype", ["str", "int64", "float64"])
def test_index_with_index_dtype(data, dtype):
    pidx = pd.Index(data)
    gidx = cudf.Index(data)

    expected = pd.Index(pidx, dtype=dtype)
    actual = cudf.Index(gidx, dtype=dtype)

    assert_eq(expected, actual)


def test_period_index_error():
    pidx = pd.PeriodIndex(data=[pd.Period("2020-01")])
    with pytest.raises(NotImplementedError):
        cudf.from_pandas(pidx)
    with pytest.raises(NotImplementedError):
        cudf.Index(pidx)
    with pytest.raises(NotImplementedError):
        cudf.Series(pidx)
    with pytest.raises(NotImplementedError):
        cudf.Series(pd.Series(pidx))
    with pytest.raises(NotImplementedError):
        cudf.Series(pd.array(pidx))


def test_index_from_dataframe_valueerror():
    with pytest.raises(ValueError):
        cudf.Index(cudf.DataFrame(range(1)))


def test_index_from_scalar_valueerror():
    with pytest.raises(ValueError):
        cudf.Index(11)


@pytest.mark.parametrize("idx", [0, np.int64(0)])
def test_index_getitem_from_int(idx):
    result = cudf.Index([1, 2])[idx]
    assert result == 1


@pytest.mark.parametrize("idx", [1.5, True, "foo"])
def test_index_getitem_from_nonint_raises(idx):
    with pytest.raises(ValueError):
        cudf.Index([1, 2])[idx]


@pytest.mark.parametrize(
    "data",
    [
        cp.ones(5, dtype=cp.float16),
        np.ones(5, dtype="float16"),
        pd.Series([0.1, 1.2, 3.3], dtype="float16"),
        pytest.param(
            pa.array(np.ones(5, dtype="float16")),
            marks=pytest.mark.xfail(
                reason="https://issues.apache.org/jira/browse/ARROW-13762"
            ),
        ),
    ],
)
def test_index_raises_float16(data):
    with pytest.raises(TypeError):
        cudf.Index(data)


def test_from_pandas_rangeindex_return_rangeindex():
    pidx = pd.RangeIndex(start=3, stop=9, step=3, name="a")
    result = cudf.Index.from_pandas(pidx)
    expected = cudf.RangeIndex(start=3, stop=9, step=3, name="a")
    assert_eq(result, expected, exact=True)


@pytest.mark.parametrize(
    "idx",
    [
        cudf.RangeIndex(1),
        cudf.DatetimeIndex(np.array([1, 2], dtype="datetime64[ns]")),
        cudf.TimedeltaIndex(np.array([1, 2], dtype="timedelta64[ns]")),
    ],
)
def test_index_to_pandas_nullable_notimplemented(idx):
    with pytest.raises(NotImplementedError):
        idx.to_pandas(nullable=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
    ],
)
def test_index_to_pandas_arrow_type_nullable_raises(scalar):
    pa_array = pa.array([scalar, None])
    idx = cudf.Index(pa_array)
    with pytest.raises(ValueError):
        idx.to_pandas(nullable=True, arrow_type=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
    ],
)
def test_index_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    idx = cudf.Index(pa_array)
    result = idx.to_pandas(arrow_type=True)
    expected = pd.Index(pd.arrays.ArrowExtensionArray(pa_array))
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("data", [range(-3, 3), range(1, 3), range(0)])
def test_rangeindex_all(data):
    result = cudf.RangeIndex(data).all()
    expected = cudf.Index(list(data)).all()
    assert result == expected


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("data", [range(2), range(2, -1, -1)])
def test_rangeindex_factorize(sort, data):
    res_codes, res_uniques = cudf.RangeIndex(data).factorize(sort=sort)
    exp_codes, exp_uniques = cudf.Index(list(data)).factorize(sort=sort)
    assert_eq(res_codes, exp_codes)
    assert_eq(res_uniques, exp_uniques)


def test_rangeindex_dropna():
    ri = cudf.RangeIndex(range(2))
    result = ri.dropna()
    expected = ri.copy()
    assert_eq(result, expected)


@pytest.mark.parametrize("data", [range(2), [10, 11, 12]])
def test_index_contains_hashable(data):
    gidx = cudf.Index(data)
    pidx = gidx.to_pandas()

    assert_exceptions_equal(
        lambda: [] in gidx,
        lambda: [] in pidx,
        lfunc_args_and_kwargs=((),),
        rfunc_args_and_kwargs=((),),
    )


@pytest.mark.parametrize("data", [[0, 1, 2], [1.1, 2.3, 4.5]])
@pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
@pytest.mark.parametrize("needle", [0, 1, 2.3])
def test_index_contains_float_int(data, dtype, needle):
    gidx = cudf.Index(data=data, dtype=dtype)
    pidx = gidx.to_pandas()

    actual = needle in gidx
    expected = needle in pidx

    assert_eq(actual, expected)


def test_Index_init_with_nans():
    with cudf.option_context("mode.pandas_compatible", True):
        gi = cudf.Index([1, 2, 3, np.nan])
    assert gi.dtype == np.dtype("float64")
    pi = pd.Index([1, 2, 3, np.nan])
    assert_eq(pi, gi)


def test_index_datetime_repeat():
    gidx = cudf.date_range("2021-01-01", periods=3, freq="D")
    pidx = gidx.to_pandas()

    actual = gidx.repeat(5)
    expected = pidx.repeat(5)

    assert_eq(actual, expected)

    actual = gidx.to_frame().repeat(5)

    assert_eq(actual.index, expected)
