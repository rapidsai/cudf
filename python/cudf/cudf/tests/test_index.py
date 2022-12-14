# Copyright (c) 2018-2022, NVIDIA CORPORATION.

"""
Test related to Index
"""
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_110, PANDAS_GE_133
from cudf.core.index import (
    CategoricalIndex,
    DatetimeIndex,
    GenericIndex,
    IntervalIndex,
    RangeIndex,
    as_index,
)
from cudf.testing._utils import (
    FLOAT_TYPES,
    NUMERIC_TYPES,
    OTHER_TYPES,
    SIGNED_INTEGER_TYPES,
    SIGNED_TYPES,
    UNSIGNED_TYPES,
    _create_pandas_series,
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
    assert idx.find_label_range(4, 6) == (0, 3)
    assert idx.find_label_range(5, 10) == (1, 4)
    assert idx.find_label_range(0, 6) == (0, 3)
    assert idx.find_label_range(4, 11) == (0, 4)

    # Non-monotonic Index
    idx_nm = cudf.Index(np.asarray([5, 4, 6, 10]))
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


def test_index_find_label_range_rangeindex():
    """Cudf specific"""
    # step > 0
    # 3, 8, 13, 18
    ridx = RangeIndex(3, 20, 5)
    assert ridx.find_label_range(3, 8) == (0, 2)
    assert ridx.find_label_range(0, 7) == (0, 1)
    assert ridx.find_label_range(3, 19) == (0, 4)
    assert ridx.find_label_range(2, 21) == (0, 4)

    # step < 0
    # 20, 15, 10, 5
    ridx = RangeIndex(20, 3, -5)
    assert ridx.find_label_range(15, 10) == (1, 3)
    assert ridx.find_label_range(10, 0) == (2, 4)
    assert ridx.find_label_range(30, 13) == (0, 2)
    assert ridx.find_label_range(30, 0) == (0, 4)


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
        lambda x: x.sum(),
        lambda x: x.mean(),
        lambda x: x.any(),
        lambda x: x.all(),
        lambda x: x.prod(),
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

    assert_eq(
        pdf_category_index.codes,
        gdf_category_index.codes.astype(
            pdf_category_index.codes.dtype
        ).to_numpy(),
    )


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
@pytest.mark.parametrize("dtype", SIGNED_INTEGER_TYPES)
def test_index_copy_range(name, dtype, deep=True):
    cidx = cudf.RangeIndex(1, 5)
    pidx = cidx.to_pandas()

    with pytest.warns(FutureWarning):
        pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    with pytest.warns(FutureWarning):
        cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype,", ["datetime64[ns]", "int64"])
def test_index_copy_datetime(name, dtype, deep=True):
    cidx = cudf.DatetimeIndex(["2001", "2002", "2003"])
    pidx = cidx.to_pandas()

    with pytest.warns(FutureWarning):
        pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    with pytest.warns(FutureWarning):
        cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype", ["category", "object"])
def test_index_copy_string(name, dtype, deep=True):
    cidx = cudf.StringIndex(["a", "b", "c"])
    pidx = cidx.to_pandas()

    with pytest.warns(FutureWarning):
        pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    with pytest.warns(FutureWarning):
        cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize(
    "dtype",
    NUMERIC_TYPES + ["datetime64[ns]", "timedelta64[ns]"] + OTHER_TYPES,
)
def test_index_copy_integer(name, dtype, deep=True):
    """Test for NumericIndex Copy Casts"""
    cidx = cudf.Index([1, 2, 3])
    pidx = cidx.to_pandas()

    with pytest.warns(FutureWarning):
        pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    with pytest.warns(FutureWarning):
        cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype", SIGNED_TYPES)
def test_index_copy_float(name, dtype, deep=True):
    """Test for NumericIndex Copy Casts"""
    cidx = cudf.Index([1.0, 2.0, 3.0])
    pidx = cidx.to_pandas()

    with pytest.warns(FutureWarning):
        pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    with pytest.warns(FutureWarning):
        cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["category"])
def test_index_copy_category(name, dtype, deep=True):
    cidx = cudf.core.index.CategoricalIndex([1, 2, 3])
    pidx = cidx.to_pandas()

    with pytest.warns(FutureWarning):
        pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    with pytest.warns(FutureWarning):
        cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize(
    "idx",
    [
        cudf.DatetimeIndex(["2001", "2002", "2003"]),
        cudf.StringIndex(["a", "b", "c"]),
        cudf.Index([1, 2, 3]),
        cudf.Index([1.0, 2.0, 3.0]),
        cudf.CategoricalIndex([1, 2, 3]),
        cudf.CategoricalIndex(["a", "b", "c"]),
    ],
)
def test_index_copy_deep(idx, deep):
    """Test if deep copy creates a new instance for device data."""
    idx_copy = idx.copy(deep=deep)
    if not deep:
        assert_column_memory_eq(idx._values, idx_copy._values)
    else:
        assert_column_memory_ne(idx._values, idx_copy._values)


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
    gidx_1 = cudf.StringIndex(idx, name="idx")
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
        pytest.param(
            pd.Index(range(5)),
            pd.Index(range(5)) > 1,
            10,
            None,
            marks=pytest.mark.xfail(
                condition=not PANDAS_GE_133,
                reason="https://github.com/pandas-dev/pandas/issues/43240",
            ),
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
        if pd.api.types.is_categorical_dtype(ps):
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
            compare_error_message=False,
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
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 5.0, 6.0, 0.0, 1.3],
        [],
    ],
)
@pytest.mark.parametrize("sort", [None, False])
def test_index_difference(data, other, sort):
    pd_data = pd.Index(data)
    pd_other = pd.Index(other)

    gd_data = cudf.core.index.as_index(data)
    gd_other = cudf.core.index.as_index(other)

    if (
        gd_data.dtype.kind == "f"
        and gd_other.dtype.kind != "f"
        or (gd_data.dtype.kind != "f" and gd_other.dtype.kind == "f")
    ):
        pytest.mark.xfail(
            condition=not PANDAS_GE_110,
            reason="Bug in Pandas: "
            "https://github.com/pandas-dev/pandas/issues/35217",
        )

    expected = pd_data.difference(pd_other, sort=sort)
    actual = gd_data.difference(gd_other, sort=sort)
    assert_eq(expected, actual)


def test_index_difference_sort_error():
    pdi = pd.Index([1, 2, 3])
    gdi = cudf.Index([1, 2, 3])

    assert_exceptions_equal(
        pdi.difference,
        gdi.difference,
        ([pdi], {"sort": True}),
        ([gdi], {"sort": True}),
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

    if (
        gd_data.dtype.kind == "f" or gd_other.dtype.kind == "f"
    ) and cudf.utils.dtypes.is_mixed_with_object_dtype(gd_data, gd_other):
        pytest.mark.xfail(
            condition=not PANDAS_GE_110,
            reason="Bug in Pandas: "
            "https://github.com/pandas-dev/pandas/issues/35217",
        )

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

    if (
        gd_data.dtype.kind == "f"
        and gd_other.dtype.kind != "f"
        or (gd_data.dtype.kind != "f" and gd_other.dtype.kind == "f")
    ):
        pytest.mark.xfail(
            condition=not PANDAS_GE_110,
            reason="Bug in Pandas: "
            "https://github.com/pandas-dev/pandas/issues/35217",
        )

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

    expected = pd_data.equals(_create_pandas_series(pd_other))
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

    gd_data = cudf.core.index.as_index(data)
    gd_other = cudf.core.index.as_index(other)

    if cudf.utils.dtypes.is_mixed_with_object_dtype(gd_data, gd_other):
        gd_data = gd_data.astype("str")
        gd_other = gd_other.astype("str")

    expected = pd_data.append(pd_other)

    actual = gd_data.append(gd_other)
    if len(data) == 0 and len(other) == 0:
        # Pandas default dtype to "object" for empty list
        # cudf default dtype to "float" for empty list
        assert_eq(expected, actual.astype("str"))
    elif actual.dtype == "object":
        assert_eq(expected.astype("str"), actual)
    else:
        assert_eq(expected, actual)


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
        expected_error_message=r"all inputs must be Index",
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

    expected = pd_data.append(pd_other)
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
    with pytest.warns(FutureWarning):
        pindex = pd.Int64Index(data, dtype=dtype, name=name)
    # Int8Index
    with pytest.warns(FutureWarning):
        gindex = cudf.Int8Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("int8")

    # Int16Index
    with pytest.warns(FutureWarning):
        gindex = cudf.Int16Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("int16")

    # Int32Index
    with pytest.warns(FutureWarning):
        gindex = cudf.Int32Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("int32")

    # Int64Index
    with pytest.warns(FutureWarning):
        gindex = cudf.Int64Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("int64")


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("name", [1, "a", None])
@pytest.mark.parametrize("dtype", UNSIGNED_TYPES)
def test_unsigned_integer_index_apis(data, name, dtype):
    with pytest.warns(FutureWarning):
        pindex = pd.UInt64Index(data, dtype=dtype, name=name)
    # UInt8Index
    with pytest.warns(FutureWarning):
        gindex = cudf.UInt8Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("uint8")

    # UInt16Index
    with pytest.warns(FutureWarning):
        gindex = cudf.UInt16Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("uint16")

    # UInt32Index
    with pytest.warns(FutureWarning):
        gindex = cudf.UInt32Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("uint32")

    # UInt64Index
    with pytest.warns(FutureWarning):
        gindex = cudf.UInt64Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("uint64")


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("name", [1, "a", None])
@pytest.mark.parametrize("dtype", FLOAT_TYPES)
def test_float_index_apis(data, name, dtype):
    with pytest.warns(FutureWarning):
        pindex = pd.Float64Index(data, dtype=dtype, name=name)
    # Float32Index
    with pytest.warns(FutureWarning):
        gindex = cudf.Float32Index(data, dtype=dtype, name=name)

    assert_eq(pindex, gindex)
    assert gindex.dtype == np.dtype("float32")

    # Float64Index
    with pytest.warns(FutureWarning):
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


INTERVAL_BOUNDARY_TYPES = [
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    cudf.Scalar,
]


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("end", [4, 5, 6, 7])
def test_interval_range_basic(start, end, closed):
    pindex = pd.interval_range(start=start, end=end, closed=closed)
    gindex = cudf.interval_range(start=start, end=end, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_dtype_basic(start_t, end_t):
    start, end = start_t(24), end_t(42)
    start_val = start.value if isinstance(start, cudf.Scalar) else start
    end_val = end.value if isinstance(end, cudf.Scalar) else end
    pindex = pd.interval_range(start=start_val, end=end_val, closed="left")
    gindex = cudf.interval_range(start=start, end=end, closed="left")

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("start", [0])
@pytest.mark.parametrize("end", [0])
def test_interval_range_empty(start, end, closed):
    pindex = pd.interval_range(start=start, end=end, closed=closed)
    gindex = cudf.interval_range(start=start, end=end, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("freq", [1, 2, 3])
@pytest.mark.parametrize("start", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("end", [6, 8, 10, 43, 70])
def test_interval_range_freq_basic(start, end, freq, closed):
    pindex = pd.interval_range(start=start, end=end, freq=freq, closed=closed)
    gindex = cudf.interval_range(
        start=start, end=end, freq=freq, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_freq_basic_dtype(start_t, end_t, freq_t):
    start, end, freq = start_t(5), end_t(70), freq_t(3)
    start_val = start.value if isinstance(start, cudf.Scalar) else start
    end_val = end.value if isinstance(end, cudf.Scalar) else end
    freq_val = freq.value if isinstance(freq, cudf.Scalar) else freq
    pindex = pd.interval_range(
        start=start_val, end=end_val, freq=freq_val, closed="left"
    )
    gindex = cudf.interval_range(
        start=start, end=end, freq=freq, closed="left"
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 1.0, 2, 2.0, 3.0, 3])
@pytest.mark.parametrize("start", [0, 0.0, 1.0, 1, 2, 2.0, 3.0, 3])
@pytest.mark.parametrize("end", [4, 4.0, 5.0, 5, 6, 6.0, 7.0, 7])
def test_interval_range_periods_basic(start, end, periods, closed):
    pindex = pd.interval_range(
        start=start, end=end, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        start=start, end=end, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("periods_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_periods_basic_dtype(start_t, end_t, periods_t):
    start, end, periods = start_t(0), end_t(4), periods_t(1.0)
    start_val = start.value if isinstance(start, cudf.Scalar) else start
    end_val = end.value if isinstance(end, cudf.Scalar) else end
    periods_val = (
        periods.value if isinstance(periods, cudf.Scalar) else periods
    )
    pindex = pd.interval_range(
        start=start_val, end=end_val, periods=periods_val, closed="left"
    )
    gindex = cudf.interval_range(
        start=start, end=end, periods=periods, closed="left"
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 2, 3])
@pytest.mark.parametrize("freq", [1, 2, 3, 4])
@pytest.mark.parametrize("end", [4, 8, 9, 10])
def test_interval_range_periods_freq_end(end, freq, periods, closed):
    pindex = pd.interval_range(
        end=end, freq=freq, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        end=end, freq=freq, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("periods_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_periods_freq_end_dtype(periods_t, freq_t, end_t):
    periods, freq, end = periods_t(2), freq_t(3), end_t(10)
    freq_val = freq.value if isinstance(freq, cudf.Scalar) else freq
    end_val = end.value if isinstance(end, cudf.Scalar) else end
    periods_val = (
        periods.value if isinstance(periods, cudf.Scalar) else periods
    )
    pindex = pd.interval_range(
        end=end_val, freq=freq_val, periods=periods_val, closed="left"
    )
    gindex = cudf.interval_range(
        end=end, freq=freq, periods=periods, closed="left"
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 2, 3])
@pytest.mark.parametrize("freq", [1, 2, 3, 4])
@pytest.mark.parametrize("start", [1, 4, 9, 12])
def test_interval_range_periods_freq_start(start, freq, periods, closed):
    pindex = pd.interval_range(
        start=start, freq=freq, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        start=start, freq=freq, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("periods_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_periods_freq_start_dtype(periods_t, freq_t, start_t):
    periods, freq, start = periods_t(2), freq_t(3), start_t(9)
    freq_val = freq.value if isinstance(freq, cudf.Scalar) else freq
    start_val = start.value if isinstance(start, cudf.Scalar) else start
    periods_val = (
        periods.value if isinstance(periods, cudf.Scalar) else periods
    )
    pindex = pd.interval_range(
        start=start_val, freq=freq_val, periods=periods_val, closed="left"
    )
    gindex = cudf.interval_range(
        start=start, freq=freq, periods=periods, closed="left"
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
@pytest.mark.parametrize(
    "data",
    [
        ([pd.Interval(30, 50)]),
        ([pd.Interval(0, 3), pd.Interval(1, 7)]),
        ([pd.Interval(0.2, 60.3), pd.Interval(1, 7), pd.Interval(0, 0)]),
        ([]),
    ],
)
def test_interval_index_basic(data, closed):
    pindex = pd.IntervalIndex(data, closed=closed)
    gindex = IntervalIndex(data, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
def test_interval_index_empty(closed):
    pindex = pd.IntervalIndex([], closed=closed)
    gindex = IntervalIndex([], closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
@pytest.mark.parametrize(
    "data",
    [
        ([pd.Interval(1, 6), pd.Interval(1, 10), pd.Interval(1, 3)]),
        (
            [
                pd.Interval(3.5, 6.0),
                pd.Interval(1.0, 7.0),
                pd.Interval(0.0, 10.0),
            ]
        ),
        (
            [
                pd.Interval(50, 100, closed="left"),
                pd.Interval(1.0, 7.0, closed="left"),
                pd.Interval(16, 322, closed="left"),
            ]
        ),
        (
            [
                pd.Interval(50, 100, closed="right"),
                pd.Interval(1.0, 7.0, closed="right"),
                pd.Interval(16, 322, closed="right"),
            ]
        ),
    ],
)
def test_interval_index_many_params(data, closed):

    pindex = pd.IntervalIndex(data, closed=closed)
    gindex = IntervalIndex(data, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_interval_index_from_breaks(closed):
    breaks = [0, 3, 6, 10]
    pindex = pd.IntervalIndex.from_breaks(breaks, closed=closed)
    gindex = IntervalIndex.from_breaks(breaks, closed=closed)

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
    )  # Int64Index v/s Float64Index


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
    gdi = cudf.Index(df)

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


def test_index_rangeindex_search_range():
    # step > 0
    ridx = RangeIndex(-13, 17, 4)
    stop = ridx._start + ridx._step * len(ridx)
    for i in range(len(ridx)):
        assert i == search_range(
            ridx._start, stop, ridx[i], ridx._step, side="left"
        )
        assert i + 1 == search_range(
            ridx._start, stop, ridx[i], ridx._step, side="right"
        )


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
@pytest.mark.parametrize("key", list(range(0, 8)))
@pytest.mark.parametrize("method", [None, "ffill", "bfill", "nearest"])
def test_get_loc_single_unique_numeric(idx, key, method):
    pi = idx
    gi = cudf.from_pandas(pi)

    if (
        (key not in pi and method is None)
        # `method` only applicable to monotonic index
        or (not pi.is_monotonic_increasing and method is not None)
        # Get key before the first element is KeyError
        or (key == 0 and method in "ffill")
        # Get key after the last element is KeyError
        or (key == 7 and method in "bfill")
    ):
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
            compare_error_message=False,
        )
    else:
        with expect_warning_if(method is not None):
            expected = pi.get_loc(key, method=method)
        with expect_warning_if(method is not None):
            got = gi.get_loc(key, method=method)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx",
    [pd.RangeIndex(3, 100, 4)],
)
@pytest.mark.parametrize("key", list(range(1, 110, 3)))
@pytest.mark.parametrize("method", [None, "ffill"])
def test_get_loc_rangeindex(idx, key, method):
    pi = idx
    gi = cudf.from_pandas(pi)

    if (
        (key not in pi and method is None)
        # Get key before the first element is KeyError
        or (key < pi.start and method in "ffill")
        # Get key after the last element is KeyError
        or (key >= pi.stop and method in "bfill")
    ):
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
            compare_error_message=False,
        )
    else:
        with expect_warning_if(method is not None):
            expected = pi.get_loc(key, method=method)
        with expect_warning_if(method is not None):
            got = gi.get_loc(key, method=method)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 3, 3, 6]),  # monotonic
        pd.Index([6, 1, 3, 3]),  # non-monotonic
    ],
)
@pytest.mark.parametrize("key", [0, 3, 6, 7])
@pytest.mark.parametrize("method", [None])
def test_get_loc_single_duplicate_numeric(idx, key, method):
    pi = idx
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
            compare_error_message=False,
        )
    else:
        with expect_warning_if(method is not None):
            expected = pi.get_loc(key, method=method)
        with expect_warning_if(method is not None):
            got = gi.get_loc(key, method=method)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx", [pd.Index(["b", "f", "m", "q"]), pd.Index(["m", "f", "b", "q"])]
)
@pytest.mark.parametrize("key", ["a", "f", "n", "z"])
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_loc_single_unique_string(idx, key, method):
    pi = idx
    gi = cudf.from_pandas(pi)

    if (
        (key not in pi and method is None)
        # `method` only applicable to monotonic index
        or (not pi.is_monotonic_increasing and method is not None)
        # Get key before the first element is KeyError
        or (key == "a" and method == "ffill")
        # Get key after the last element is KeyError
        or (key == "z" and method == "bfill")
    ):
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
            compare_error_message=False,
        )
    else:
        with expect_warning_if(method is not None):
            expected = pi.get_loc(key, method=method)
        with expect_warning_if(method is not None):
            got = gi.get_loc(key, method=method)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "idx", [pd.Index(["b", "m", "m", "q"]), pd.Index(["m", "f", "m", "q"])]
)
@pytest.mark.parametrize("key", ["a", "f", "n", "z"])
@pytest.mark.parametrize("method", [None])
def test_get_loc_single_duplicate_string(idx, key, method):
    pi = idx
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
            compare_error_message=False,
        )
    else:
        with expect_warning_if(method is not None):
            expected = pi.get_loc(key, method=method)
        with expect_warning_if(method is not None):
            got = gi.get_loc(key, method=method)

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
            [(1, 1, 1), (1, 1, 2), (1, 1, 2), (1, 2, 3), (2, 1, 1), (2, 2, 1)]
        ),
    ],
)
@pytest.mark.parametrize("key", [1, (1, 2), (1, 2, 3), (2, 1, 1), (9, 9, 9)])
@pytest.mark.parametrize("method", [None])
def test_get_loc_multi_numeric(idx, key, method):
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
            compare_error_message=False,
        )
    else:
        with expect_warning_if(method is not None):
            expected = pi.get_loc(key, method=method)
        with expect_warning_if(method is not None):
            got = gi.get_loc(key, method=method)

        assert_eq(expected, got)


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
@pytest.mark.parametrize("method", [None])
def test_get_loc_multi_numeric_deviate(idx, key, result, method):
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
                lfunc_args_and_kwargs=([], {"key": key, "method": method}),
                rfunc_args_and_kwargs=([], {"key": key, "method": method}),
                compare_error_message=False,
            )
    else:
        expected = result
        with expect_warning_if(method is not None):
            got = gi.get_loc(key, method=method)

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
@pytest.mark.parametrize("method", [None])
def test_get_loc_multi_string(idx, key, method):
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
            compare_error_message=False,
        )
    else:
        with expect_warning_if(method is not None):
            expected = pi.get_loc(key, method=method)
        with expect_warning_if(method is not None):
            got = gi.get_loc(key, method=method)

        assert_eq(expected, got)


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
    ],
)
@pytest.mark.parametrize("sort", [None, False])
def test_union_index(idx1, idx2, sort):
    expected = idx1.union(idx2, sort=sort)

    idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.Index) else idx1
    idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.Index) else idx2

    actual = idx1.union(idx2, sort=sort)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (pd.RangeIndex(0, 10), pd.RangeIndex(3, 7)),
        (pd.RangeIndex(0, 10), pd.RangeIndex(-10, 20)),
        (pd.RangeIndex(0, 10, name="a"), pd.RangeIndex(90, 100, name="b")),
        (pd.Index([0, 1, 2, 30], name="a"), pd.Index([30, 0, 90, 100])),
        (pd.Index([0, 1, 2, 30], name="a"), [90, 100]),
        (pd.Index([0, 1, 2, 30]), pd.Index([0, 10, 1.0, 11])),
        (pd.Index(["a", "b", "c", "d", "c"]), pd.Index(["a", "c", "z"])),
        (
            pd.Index(["a", "b", "c", "d", "c"]),
            pd.Index(["a", "b", "c", "d", "c"]),
        ),
        (pd.Index([True, False, True, True]), pd.Index([10, 11, 12, 0, 1, 2])),
        (pd.Index([True, False, True, True]), pd.Index([True, True])),
    ],
)
@pytest.mark.parametrize("sort", [None, False])
def test_intersection_index(idx1, idx2, sort):

    expected = idx1.intersection(idx2, sort=sort)

    idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.Index) else idx1
    idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.Index) else idx2

    actual = idx1.intersection(idx2, sort=sort)

    assert_eq(expected, actual, exact=False)


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

    expected = getattr(pidx, func)()
    actual = getattr(gidx, func)()

    if gidx.dtype == np.dtype("bool") and func == "is_object":
        assert_eq(False, actual)
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize(
    "resolution", ["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"]
)
def test_index_datetime_ceil(resolution):
    cuidx = cudf.DatetimeIndex([1000000, 2000000, 3000000, 4000000, 5000000])
    pidx = cuidx.to_pandas()

    pidx_ceil = pidx.ceil(resolution)
    cuidx_ceil = cuidx.ceil(resolution)

    assert_eq(pidx_ceil, cuidx_ceil)


@pytest.mark.parametrize(
    "resolution", ["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"]
)
def test_index_datetime_floor(resolution):
    cuidx = cudf.DatetimeIndex([1000000, 2000000, 3000000, 4000000, 5000000])
    pidx = cuidx.to_pandas()

    pidx_floor = pidx.floor(resolution)
    cuidx_floor = cuidx.floor(resolution)

    assert_eq(pidx_floor, cuidx_floor)


@pytest.mark.parametrize(
    "resolution", ["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"]
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
    psr = _create_pandas_series(data)
    gsr = cudf.Series.from_pandas(psr)

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
            expected_error_message=re.escape(
                "values need to be a Multi-Index or set/list-like tuple "
                "squences  when `level=None`."
            ),
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


def test_rangeindex_join_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for join.
    idx1 = cudf.RangeIndex(0, 10)
    idx2 = cudf.RangeIndex(5, 15)

    actual = idx1.join(idx2, how="inner", sort=True)
    expected = cudf.Index(
        [5, 6, 7, 8, 9], dtype=f"int{default_integer_bitwidth}", name=0
    )

    assert_eq(expected, actual)


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
    ps = gs.to_pandas(nullable=True)

    assert_eq(gs.hasnans, ps.hasnans)


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
