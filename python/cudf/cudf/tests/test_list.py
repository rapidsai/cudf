# Copyright (c) 2020-2021, NVIDIA CORPORATION.
import functools

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[[]]],
        [[0]],
        [[0, 1]],
        [[0, 1], [2, 3]],
        [[[0, 1], [2]], [[3, 4]]],
        [[None]],
        [[[None]]],
        [[None], None],
        [[1, None], [1]],
        [[1, None], None],
        [[[1, None], None], None],
    ],
)
def test_create_list_series(data):
    expect = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[None]]},
        {"a": [[1, 2, 3]]},
        {"a": [[1, 2, 3]], "b": [[2, 3, 4]]},
        {"a": [[1, 2, 3, None], [None]], "b": [[2, 3, 4], [5]], "c": None},
        {"a": [[1]], "b": [[1, 2, 3]]},
        pd.DataFrame({"a": [[1, 2, 3]]}),
    ],
)
def test_df_list_dtypes(data):
    expect = pd.DataFrame(data)
    got = cudf.DataFrame(data)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[[]]],
        [[0]],
        [[0, 1]],
        [[0, 1], [2, 3]],
        [[[0, 1], [2]], [[3, 4]]],
        [[[0, 1, None], None], None, [[3, 2, None], None]],
        [[["a", "c", None], None], None, [["b", "d", None], None]],
    ],
)
def test_leaves(data):
    pa_array = pa.array(data)
    while hasattr(pa_array, "flatten"):
        pa_array = pa_array.flatten()
    dtype = "int8" if isinstance(pa_array, pa.NullArray) else None
    expect = cudf.Series(pa_array, dtype=dtype)
    got = cudf.Series(data).list.leaves
    assert_eq(expect, got)


def test_list_to_pandas_nullable_true():
    df = cudf.DataFrame({"a": cudf.Series([[1, 2, 3]])})
    actual = df.to_pandas(nullable=True)
    expected = pd.DataFrame({"a": pd.Series([[1, 2, 3]])})

    assert_eq(actual, expected)


def test_listdtype_hash():
    a = cudf.core.dtypes.ListDtype("int64")
    b = cudf.core.dtypes.ListDtype("int64")

    assert hash(a) == hash(b)

    c = cudf.core.dtypes.ListDtype("int32")

    assert hash(a) != hash(c)


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[1, 2, 3], [4, 5]],
        [[1, 2, 3], [], [4, 5]],
        [[1, 2, 3], None, [4, 5]],
        [[None, None], [None]],
        [[[[[[1, 2, 3]]]]]],
        cudf.Series([[1, 2]]).iloc[0:0],
        cudf.Series([None, [1, 2]]).iloc[0:1],
    ],
)
def test_len(data):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas()

    expect = psr.map(lambda x: len(x) if x is not None else None)
    got = gsr.list.len()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    ("data", "idx"),
    [
        ([[1, 2, 3], [3, 4, 5], [4, 5, 6]], [[0, 1], [2], [1, 2]]),
        ([[1, 2, 3], [3, 4, 5], [4, 5, 6]], [[1, 2, 0], [1, 0, 2], [0, 1, 2]]),
        ([[1, 2, 3], []], [[0, 1], []]),
        ([[1, 2, 3], [None]], [[0, 1], []]),
        ([[1, None, 3], None], [[0, 1], []]),
    ],
)
def test_take(data, idx):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.Series(zip(ps, idx)).map(
        lambda x: [x[0][i] for i in x[1]] if x[0] is not None else None
    )
    got = gs.list.take(idx)
    assert_eq(expected, got)


@pytest.mark.parametrize(
    ("invalid", "exception"),
    [
        ([[0]], pytest.raises(ValueError, match="different size")),
        ([1, 2, 3, 4], pytest.raises(ValueError, match="should be list type")),
        (
            [["a", "b"], ["c"]],
            pytest.raises(
                TypeError, match="should be column of values of index types"
            ),
        ),
        (
            [[[1], [0]], [[0]]],
            pytest.raises(
                TypeError, match="should be column of values of index types"
            ),
        ),
        ([[0, 1], None], pytest.raises(ValueError, match="contains null")),
    ],
)
def test_take_invalid(invalid, exception):
    gs = cudf.Series([[0, 1], [2, 3]])
    with exception:
        gs.list.take(invalid)


def key_func_builder(x, na_position):
    if x is None:
        if na_position == "first":
            return -1e8
        else:
            return 1e8
    else:
        return x


@pytest.mark.parametrize(
    "data",
    [
        [[4, 2, None, 9], [8, 8, 2], [2, 1]],
        [[4, 2, None, 9], [8, 8, 2], None],
        [[4, 2, None, 9], [], None],
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        None,
        pd.Index(["a", "b", "c"]),
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (1, "a")], names=["l0", "l1"]
        ),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_sort_values(data, index, ascending, na_position, ignore_index):
    key_func = functools.partial(key_func_builder, na_position=na_position)

    ps = pd.Series(data, index=index)
    gs = cudf.from_pandas(ps)

    expected = ps.apply(
        lambda x: sorted(x, key=key_func, reverse=not ascending)
        if x is not None
        else None
    )
    if ignore_index:
        expected.reset_index(drop=True, inplace=True)
    got = gs.list.sort_values(
        ascending=ascending, na_position=na_position, ignore_index=ignore_index
    )

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data, index, expect",
    [
        ([[None, None], [None, None]], 0, [None, None]),
        ([[1, 2], [3, 4]], 0, [1, 3]),
        ([["a", "b"], ["c", "d"]], 1, ["b", "d"]),
        ([[1, None], [None, 2]], 1, [None, 2]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 1, [[3, 4], [7, 8]]),
    ],
)
def test_get(data, index, expect):
    sr = cudf.Series(data)
    expect = cudf.Series(expect)
    got = sr.list.get(index)
    assert_eq(expect, got)


def test_get_nested_lists():
    sr = cudf.Series(
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [], [[3, 4], [7, 8]]],
            [[], [[9, 10]], [[11, 12], [13, 14]]],
        ]
    )
    expect = cudf.Series([[[1, 2], [3, 4]], []])
    got = sr.list.get(0)
    assert_eq(expect, got)


def test_get_nulls():
    with pytest.raises(IndexError, match="list index out of range"):
        sr = cudf.Series([[], [], []])
        sr.list.get(100)
