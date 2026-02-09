# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_220
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


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
    gidx = cudf.Index(pidx)

    is_dt_str = (
        next(iter(values), None) == "2019-01-01 04:00:00"
        and len(pidx)
        and pidx.dtype.kind == "M"
    )
    with expect_warning_if(is_dt_str):
        got = gidx.isin(values)
    with expect_warning_if(PANDAS_GE_220 and is_dt_str):
        expected = pidx.isin(values)

    assert_eq(got, expected)


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
