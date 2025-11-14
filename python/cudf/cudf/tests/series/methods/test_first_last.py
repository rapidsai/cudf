# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "idx",
    [
        pd.DatetimeIndex([]),
        pd.DatetimeIndex(["2010-05-31"]),
        pd.date_range("2000-01-01", "2000-12-31", periods=21),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        "10Y",
        "6M",
        "M",
        "31D",
        "0H",
        "44640T",
        "44640min",
        "2678000S",
        "2678000000L",
        "2678000000ms",
        "2678000000000U",
        "2678000000000us",
        "2678000000000000N",
        "2678000000000000ns",
    ],
)
def test_first(idx, offset):
    p = pd.Series(range(len(idx)), dtype="int64", index=idx)
    g = cudf.from_pandas(p)

    with pytest.warns(FutureWarning):
        expect = p.first(offset=offset)
    with pytest.warns(FutureWarning):
        got = g.first(offset=offset)

    assert_eq(expect, got)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_first_start_at_end_of_month():
    idx = pd.DatetimeIndex(
        [
            "2020-01-31",
            "2020-02-15",
            "2020-02-29",
            "2020-03-15",
            "2020-03-31",
            "2020-04-15",
            "2020-04-30",
        ]
    )
    offset = "3M"
    p = pd.Series(range(len(idx)), index=idx)
    g = cudf.from_pandas(p)

    with pytest.warns(FutureWarning):
        expect = p.first(offset=offset)
    with pytest.warns(FutureWarning):
        got = g.first(offset=offset)

    assert_eq(expect, got)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "idx",
    [
        pd.DatetimeIndex([]),
        pd.DatetimeIndex(["2010-05-31"]),
        pd.date_range("2000-01-01", "2000-12-31", periods=21),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        "10Y",
        "6M",
        "M",
        "31D",
        "0H",
        "44640T",
        "44640min",
        "2678000S",
        "2678000000L",
        "2678000000ms",
        "2678000000000U",
        "2678000000000us",
        "2678000000000000N",
        "2678000000000000ns",
    ],
)
def test_last(idx, offset):
    p = pd.Series(range(len(idx)), dtype="int64", index=idx)
    g = cudf.from_pandas(p)

    with pytest.warns(FutureWarning):
        expect = p.last(offset=offset)
    with pytest.warns(FutureWarning):
        got = g.last(offset=offset)

    assert_eq(expect, got)
