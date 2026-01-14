# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


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
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_index_difference(data, other, sort, name_data, name_other):
    pd_data = pd.Index(data, name=name_data)
    pd_other = pd.Index(other, name=name_other)
    if (
        not PANDAS_GE_220
        and isinstance(pd_data.dtype, pd.CategoricalDtype)
        and not isinstance(pd_other.dtype, pd.CategoricalDtype)
        and pd_other.isnull().any()
    ):
        pytest.skip(reason="https://github.com/pandas-dev/pandas/issues/57318")

    if (
        not PANDAS_GE_220
        and len(pd_other) == 0
        and len(pd_data) != len(pd_data.unique())
    ):
        pytest.skip(reason="Bug fixed in pandas-2.2+")

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
