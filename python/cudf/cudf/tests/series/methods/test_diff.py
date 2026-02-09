# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("period", [-20, 1, 0, 1, 2])
@pytest.mark.parametrize("data_empty", [False, True])
def test_diff(numeric_types_as_str, period, data_empty):
    if data_empty:
        data = None
    else:
        rng = np.random.default_rng(0)
        dtype = np.dtype(numeric_types_as_str)
        if dtype == np.int8:
            data = rng.integers(-2, 2, size=100000).astype(np.int8)
        elif dtype.kind == "f":
            data = rng.random(100000).astype(dtype) * 2 - 1
        elif dtype.kind in ("i", "u"):
            if dtype == np.int8:
                low, high = -2, 2
            elif dtype == np.int16:
                low, high = -32, 32
            elif dtype.kind == "i":
                low, high = -10000, 10000
            elif dtype in (np.uint8, np.uint16):
                low, high = 0, 32
            else:
                low, high = 0, 128
            data = rng.integers(low=low, high=high, size=100000).astype(dtype)

    gs = cudf.Series(data, dtype=numeric_types_as_str)
    ps = pd.Series(data, dtype=numeric_types_as_str)

    expected_outcome = ps.diff(period)
    diffed_outcome = gs.diff(period).astype(expected_outcome.dtype)

    if data_empty:
        assert_eq(diffed_outcome, expected_outcome, check_index_type=False)
    else:
        assert_eq(diffed_outcome, expected_outcome)


def test_diff_unsupported_dtypes():
    gs = cudf.Series(["a", "b", "c", "d", "e"])
    with pytest.raises(
        TypeError,
        match=r"unsupported operand type\(s\)",
    ):
        gs.diff()


@pytest.mark.parametrize(
    "data",
    [
        pd.date_range("2020-01-01", "2020-01-06", freq="D"),
        [True, True, True, False, True, True],
        [1.0, 2.0, 3.5, 4.0, 5.0, -1.7],
        [1, 2, 3, 3, 4, 5],
        [np.nan, None, None, np.nan, np.nan, None],
    ],
)
def test_diff_many_dtypes(data):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)
    assert_eq(ps.diff(), gs.diff())
    assert_eq(ps.diff(periods=2), gs.diff(periods=2))
