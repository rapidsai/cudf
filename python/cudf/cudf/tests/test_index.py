# Copyright (c) 2018-2025, NVIDIA CORPORATION.

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
from cudf.core.index import CategoricalIndex, RangeIndex
from cudf.testing import assert_eq
from cudf.testing._utils import (
    ALL_TYPES,
    assert_exceptions_equal,
)


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
    "rangeindex",
    [
        range(np.random.default_rng(seed=0).integers(0, 100)),
        range(9, 12, 2),
        range(20, 30),
        range(100, 1000, 10),
        range(0, 10, -2),
        range(0, -10, 2),
        range(0, -10, -2),
    ],
)
@pytest.mark.parametrize(
    "func",
    ["nunique", "min", "max", "any", "values"],
)
def test_rangeindex_methods(rangeindex, func):
    gidx = cudf.RangeIndex(rangeindex)
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


@pytest.mark.parametrize(
    "index",
    [
        range(np.random.default_rng(seed=0).integers(0, 100)),
        range(0, 10, -2),
        range(0, -10, 2),
        range(0, -10, -2),
        range(0, 1),
        [1, 2, 3, 1, None, None],
        [None, None, 3.2, 1, None, None],
        [None, "a", "3.2", "z", None, None],
        pd.Series(["a", "b", None], dtype="category"),
        np.array([1, 2, 3, None], dtype="datetime64[s]"),
    ],
)
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
    gidx = cudf.Index(index)
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


@pytest.mark.parametrize(
    "index_values",
    [range(1, 10, 2), [1, 2, 3], ["a", "b", "c"], [1.5, 2.5, 3.5]],
)
@pytest.mark.parametrize("i_type", [int, np.int8, np.int32, np.int64])
def test_scalar_getitem(index_values, i_type):
    i = i_type(1)
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


@pytest.mark.parametrize("value", [cudf.DataFrame(range(1)), 11])
def test_index_from_dataframe_scalar_raises(value):
    with pytest.raises(TypeError):
        cudf.Index(value)


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
    "data",
    [
        range(1),
        np.array([1, 2], dtype="datetime64[ns]"),
        np.array([1, 2], dtype="timedelta64[ns]"),
    ],
)
def test_index_to_pandas_nullable_notimplemented(data):
    idx = cudf.Index(data)
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
        pd.Interval(1, 2),
    ],
)
def test_index_to_pandas_arrow_type_nullable_raises(scalar):
    data = [scalar, None]
    idx = cudf.Index(data)
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


def test_rangeindex_unique_shallow_copy():
    ri_pandas = pd.RangeIndex(1)
    result = ri_pandas.unique()
    assert result is not ri_pandas

    ri_cudf = cudf.RangeIndex(1)
    result = ri_cudf.unique()
    assert result is not ri_cudf
    assert_eq(result, ri_cudf)


def test_rename_shallow_copy():
    idx = pd.Index([1])
    result = idx.rename("a")
    assert idx.to_numpy(copy=False) is result.to_numpy(copy=False)

    idx = cudf.Index([1])
    result = idx.rename("a")
    assert idx._column is result._column


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


@pytest.mark.parametrize(
    "index",
    [
        lambda: cudf.Index([1]),
        lambda: cudf.RangeIndex(1),
        lambda: cudf.MultiIndex(levels=[[0]], codes=[[0]]),
    ],
)
def test_index_assignment_no_shallow_copy(index):
    index = index()
    df = cudf.DataFrame(range(1))
    df.index = index
    assert df.index is index


def test_bool_rangeindex_raises():
    assert_exceptions_equal(
        lfunc=bool,
        rfunc=bool,
        lfunc_args_and_kwargs=[[pd.RangeIndex(0)]],
        rfunc_args_and_kwargs=[[cudf.RangeIndex(0)]],
    )


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("name", [None, "test"])
def test_categoricalindex_from_codes(ordered, name):
    codes = [0, 1, 2, 3, 4]
    categories = ["a", "b", "c", "d", "e"]
    result = cudf.CategoricalIndex.from_codes(codes, categories, ordered, name)
    expected = pd.CategoricalIndex(
        pd.Categorical.from_codes(codes, categories, ordered=ordered),
        name=name,
    )
    assert_eq(result, expected)


@pytest.mark.parametrize("klass", [cudf.RangeIndex, pd.RangeIndex])
@pytest.mark.parametrize("name_inner", [None, "a"])
@pytest.mark.parametrize("name_outer", [None, "b"])
def test_rangeindex_accepts_rangeindex(klass, name_inner, name_outer):
    result = cudf.RangeIndex(klass(range(1), name=name_inner), name=name_outer)
    expected = pd.RangeIndex(
        pd.RangeIndex(range(1), name=name_inner), name=name_outer
    )
    assert_eq(result, expected)


def test_roundtrip_index_plc_column():
    index = cudf.Index([1])
    expect = cudf.Index(index)
    actual = cudf.Index.from_pylibcudf(*expect.to_pylibcudf())
    assert_eq(expect, actual)
