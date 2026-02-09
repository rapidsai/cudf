# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.core.index import Index
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
    expect_warning_if,
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
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Does not warn on older versions of pandas",
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
    gd_data = Index(data)
    gd_other = Index(other)

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
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Does not warn on older versions of pandas",
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
