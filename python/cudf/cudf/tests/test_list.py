# Copyright (c) 2020-2021, NVIDIA CORPORATION.

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
    # TODO: do we really want this?
    # this test fails because 100 (index) > 0 (min)
    sr = cudf.Series([[], [], []])
    got = sr.list.get(100)
    expect = cudf.Series([None, None, None], dtype="int8")
    assert_eq(expect, got)
