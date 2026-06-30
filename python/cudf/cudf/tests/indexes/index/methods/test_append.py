# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

import cudf
import cudf.utils
import cudf.utils.dtypes
from cudf.core.index import Index
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
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
        # As of pandas 3.0, mixed string/numbers will return object type (of PyObjects)
        pd_data = pd_data.astype("str")
        pd_other = pd_other.astype("str")

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)
    if len(data) == 0 and len(other) == 0:
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        assert_eq(expected.astype(actual.dtype), actual)
    else:
        assert_eq(expected, actual)


def test_index_empty_append_name_conflict():
    empty = cudf.Index([], name="foo")
    non_empty = cudf.Index([1], name="bar")
    expected = cudf.Index([1])

    result = non_empty.append(empty)
    assert_eq(result, expected)

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
    gd_data = Index(data)
    gd_other = Index(other)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"cudf does not support appending an Index of "
            f"dtype `{gd_other.dtype}` with an Index "
            f"of dtype `{gd_data.dtype}`, please type-cast "
            f"either one of them to same dtypes."
        ),
    ):
        gd_data.append(gd_other)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"cudf does not support appending an Index of "
            f"dtype `{gd_other.dtype}` with an Index "
            f"of dtype `{gd_data.dtype}`, please type-cast "
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

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


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


def test_index_append_multiindex_returns_object_index():
    pd_data = pd.Index([1, 2])
    pd_other = pd.MultiIndex.from_tuples([(3, "a"), (4, "b")])

    gd_data = cudf.from_pandas(pd_data)
    gd_other = cudf.from_pandas(pd_other)

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


def test_categorical_index_append_multiindex_returns_object_index():
    pd_data = pd.CategoricalIndex(["a", "b"])
    pd_other = pd.MultiIndex.from_tuples([("c", "d"), ("e", "f")])

    gd_data = cudf.from_pandas(pd_data)
    gd_other = cudf.from_pandas(pd_other)

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


def test_empty_index_append_multiindex_returns_object_index():
    pd_data = pd.Index([])
    pd_other = pd.MultiIndex.from_tuples([(1, 2)])

    gd_data = cudf.from_pandas(pd_data)
    gd_other = cudf.from_pandas(pd_other)

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


def test_index_append_empty_multiindex_preserves_pandas_result():
    pd_data = pd.Index([1, 2])
    pd_other = pd.MultiIndex.from_arrays([[], []])

    gd_data = cudf.from_pandas(pd_data)
    gd_other = cudf.from_pandas(pd_other)

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


def test_index_append_list_with_multiindex_returns_object_index():
    pd_data = pd.Index([1])
    pd_other = [pd.MultiIndex.from_tuples([(2, "a")]), pd.Index([3])]

    gd_data = cudf.from_pandas(pd_data)
    gd_other = [cudf.from_pandas(other) for other in pd_other]

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


def test_index_append_multiindex_name_mismatch_returns_none_name():
    pd_data = pd.Index([1, 2], name="x")
    pd_other = pd.MultiIndex.from_tuples([(3, 4)], names=["x", "y"])

    gd_data = cudf.from_pandas(pd_data)
    gd_other = cudf.from_pandas(pd_other)

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


def test_string_index_append_multiindex_returns_object_index():
    pd_data = pd.Index(["x", "y"])
    pd_other = pd.MultiIndex.from_arrays([["a", "b"], [1, 2]])

    gd_data = cudf.from_pandas(pd_data)
    gd_other = cudf.from_pandas(pd_other)

    expected = pd_data.append(pd_other)
    actual = gd_data.append(gd_other)

    assert_eq(expected, actual)


def test_index_append_multiindex_name_propagation():
    # When both operands are unnamed, the result name is None (preserved).
    # When either operand is named, the result name is also None because
    # cross-type append (Index + MultiIndex) never preserves a non-None name.
    pd_data_named = pd.Index([1, 2], name="x")
    pd_other_named = pd.MultiIndex.from_tuples([(3, 4)], names=["x", "y"])
    pd_data_unnamed = pd.Index([1, 2])
    pd_other_unnamed = pd.MultiIndex.from_tuples([(3, 4)])

    for pd_data, pd_other in [
        (pd_data_named, pd_other_named),
        (pd_data_named, pd_other_unnamed),
        (pd_data_unnamed, pd_other_named),
        (pd_data_unnamed, pd_other_unnamed),
    ]:
        gd_data = cudf.from_pandas(pd_data)
        gd_other = cudf.from_pandas(pd_other)
        expected = pd_data.append(pd_other)
        actual = gd_data.append(gd_other)
        assert_eq(expected, actual)
