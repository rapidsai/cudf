# Copyright (c) 2018-2020, NVIDIA CORPORATION.

"""
Test related to Index
"""
import re

import numpy as np
import pandas as pd
import pyarrow as pa
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
    NUMERIC_TYPES,
    OTHER_TYPES,
    SIGNED_INTEGER_TYPES,
    SIGNED_TYPES,
    UNSIGNED_TYPES,
    assert_eq,
    assert_exceptions_equal,
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
    assert_eq(
        pdf.index.codes,
        gdf1.index.codes.astype(pdf.index.codes.dtype).to_array(),
    )

    assert isinstance(gdf2.index, CategoricalIndex)
    assert_eq(pdf, gdf2)
    assert_eq(pdf.index, gdf2.index)
    assert_eq(
        pdf.index.codes,
        gdf2.index.codes.astype(pdf.index.codes.dtype).to_array(),
    )


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

    assert_eq(
        pdf_category_index.codes,
        gdf_category_index.codes.astype(
            pdf_category_index.codes.dtype
        ).to_array(),
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


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype", SIGNED_INTEGER_TYPES)
def test_index_copy_range(name, dtype, deep=True):
    idx = cudf.core.index.RangeIndex(1, 5)
    idx_copy = idx.copy(name=name, deep=deep, dtype=dtype)

    idx.name = name
    assert_eq(idx, idx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype,", ["datetime64[ns]", "int64"])
def test_index_copy_datetime(name, dtype, deep=True):
    cidx = cudf.DatetimeIndex(["2001", "2002", "2003"])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    # By default, cudf.DatetimeIndex uses [ms] as base unit, pandas uses [ns]
    if dtype == "int64":
        cidx_copy = cidx_copy * 1000000
    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype", ["category", "object"])
def test_index_copy_string(name, dtype, deep=True):
    cidx = cudf.core.index.StringIndex(["a", "b", "c"])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize(
    "dtype",
    NUMERIC_TYPES + ["datetime64[ns]", "timedelta64[ns]"] + OTHER_TYPES,
)
def test_index_copy_integer(name, dtype, deep=True):
    """Test for NumericIndex Copy Casts
    """
    cidx = cudf.Int64Index([1, 2, 3])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype", SIGNED_TYPES)
def test_index_copy_float(name, dtype, deep=True):
    """Test for NumericIndex Copy Casts
    """
    cidx = cudf.Float64Index([1.0, 2.0, 3.0])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("name", ["x"])
@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["category"])
def test_index_copy_category(name, dtype, deep=True):
    cidx = cudf.core.index.CategoricalIndex([1, 2, 3])
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep, dtype=dtype)
    cidx_copy = cidx.copy(name=name, deep=deep, dtype=dtype)

    assert_eq(pidx_copy, cidx_copy)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize(
    "idx",
    [
        cudf.DatetimeIndex(["2001", "2002", "2003"]),
        cudf.core.index.StringIndex(["a", "b", "c"]),
        cudf.Int64Index([1, 2, 3]),
        cudf.Float64Index([1.0, 2.0, 3.0]),
        cudf.CategoricalIndex([1, 2, 3]),
        cudf.CategoricalIndex(["a", "b", "c"]),
    ],
)
def test_index_copy_deep(idx, deep):
    """Test if deep copy creates a new instance for device data.
    The general criterion is to compare `Buffer.ptr` between two data objects.
    Specifically for:
        - CategoricalIndex, this applies to both `.codes` and `.categories`
        - StringIndex, to every element in `._base_children`
        - Others, to `.base_data`
    No test is defined for RangeIndex.
    """
    idx_copy = idx.copy(deep=deep)
    same_ref = not deep
    if isinstance(idx, cudf.CategoricalIndex):
        assert (
            idx._values.codes.base_data.ptr
            == idx_copy._values.codes.base_data.ptr
        ) == same_ref
        if isinstance(
            idx._values.categories, cudf.core.column.string.StringColumn
        ):
            children = idx._values.categories._base_children
            copy_children = idx_copy._values.categories._base_children
            assert all(
                [
                    (
                        children[i].base_data.ptr
                        == copy_children[i].base_data.ptr
                    )
                    == same_ref
                    for i in range(len(children))
                ]
            )
        elif isinstance(
            idx._values.categories, cudf.core.column.numerical.NumericalColumn
        ):
            assert (
                idx._values.categories.base_data.ptr
                == idx_copy._values.categories.base_data.ptr
            ) == same_ref
    elif isinstance(idx, cudf.core.index.StringIndex):
        children = idx._values._base_children
        copy_children = idx_copy._values._base_children
        assert all(
            [
                (
                    (
                        children[i].base_data.ptr
                        == copy_children[i].base_data.ptr
                    )
                    == same_ref
                )
                for i in range(len(children))
            ]
        )
    else:
        assert (
            idx._values.base_data.ptr == idx_copy._values.base_data.ptr
        ) == same_ref


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
                expect.codes,
                got.codes.astype(expect.codes.dtype).fillna(-1).to_array(),
            )
            assert_eq(expect.categories, got.categories)
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
        pytest.xfail(
            "Bug in Pandas: https://github.com/pandas-dev/pandas/issues/35217"
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
        pytest.xfail(
            "Bug in Pandas: https://github.com/pandas-dev/pandas/issues/35217"
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
        pytest.xfail(
            "Bug in Pandas: https://github.com/pandas-dev/pandas/issues/35217"
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


@pytest.mark.parametrize("n", [0, 2, 5, 10, None])
@pytest.mark.parametrize("frac", [0.1, 0.5, 1, 2, None])
@pytest.mark.parametrize("replace", [True, False])
def test_index_sample_basic(n, frac, replace):
    psr = pd.Series([1, 2, 3, 4, 5])
    gindex = cudf.Index(psr)
    random_state = 0

    try:
        pout = psr.sample(
            n=n, frac=frac, replace=replace, random_state=random_state
        )
    except BaseException:
        assert_exceptions_equal(
            lfunc=psr.sample,
            rfunc=gindex.sample,
            lfunc_args_and_kwargs=(
                [],
                {
                    "n": n,
                    "frac": frac,
                    "replace": replace,
                    "random_state": random_state,
                },
            ),
            rfunc_args_and_kwargs=(
                [],
                {
                    "n": n,
                    "frac": frac,
                    "replace": replace,
                    "random_state": random_state,
                },
            ),
        )
    else:
        gout = gindex.sample(
            n=n, frac=frac, replace=replace, random_state=random_state
        )

        assert pout.shape == gout.shape


@pytest.mark.parametrize("n", [2, 5, 10, None])
@pytest.mark.parametrize("frac", [0.5, 1, 2, None])
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("axis", [0, 1])
def test_multiindex_sample_basic(n, frac, replace, axis):
    # as we currently don't support column with same name
    if axis == 1 and replace:
        return
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "float": [0.05, 0.2, 0.3, 0.2, 0.25],
            "int": [1, 3, 5, 4, 2],
        },
    )
    mul_index = cudf.Index(DataFrame.from_pandas(pdf))
    random_state = 0

    try:
        pout = pdf.sample(
            n=n,
            frac=frac,
            replace=replace,
            random_state=random_state,
            axis=axis,
        )
    except BaseException:
        assert_exceptions_equal(
            lfunc=pdf.sample,
            rfunc=mul_index.sample,
            lfunc_args_and_kwargs=(
                [],
                {
                    "n": n,
                    "frac": frac,
                    "replace": replace,
                    "random_state": random_state,
                    "axis": axis,
                },
            ),
            rfunc_args_and_kwargs=(
                [],
                {
                    "n": n,
                    "frac": frac,
                    "replace": replace,
                    "random_state": random_state,
                    "axis": axis,
                },
            ),
        )
    else:
        gout = mul_index.sample(
            n=n,
            frac=frac,
            replace=replace,
            random_state=random_state,
            axis=axis,
        )
        assert pout.shape == gout.shape


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
            r"via `tolist()` method. Consider using "
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

    assert_eq(pdi.fillna(fill_value), gdi.fillna(fill_value))


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
