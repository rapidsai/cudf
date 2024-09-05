# Copyright (c) 2024, NVIDIA CORPORATION.

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture
def ts():
    pts = pd.Timestamp("2024-08-31 12:34:56.789123456")
    gts = cudf.from_pandas(pts)
    return pts, gts


@pytest.mark.parametrize(
    "attr",
    [
        "value",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
    ],
)
def test_timestamp_properties(ts, attr):
    pts, gts = ts
    res = getattr(pts, attr)
    expect = getattr(gts, attr)

    assert_eq(res, expect)


def test_timestamp_to_scalar(ts):
    pts, gts = ts

    res = gts.to_scalar()
    expect = cudf.Scalar(pts)

    assert_eq(res, expect)


def test_timestamp_from_scalar(ts):
    pts, gts = ts
    s = cudf.Scalar(pts)

    res = cudf.Timestamp.from_scalar(s)
    expect = gts

    assert_eq(res, expect)


def test_add_timestamp_timedelta(ts):
    pts, gts = ts
    ptd = pd.Timedelta(1)

    res = gts + ptd
    expect = pts + ptd

    assert_eq(res, expect)


@pytest.mark.parametrize(
    "lhs",
    [
        pd.Timedelta(1),
        datetime(2024, 9, 5, 1, 1, 1, 1),
        np.timedelta64(5, "D"),
    ],
)
def test_subtract_timestamp_timedelta(ts, lhs):
    pts, gts = ts

    res = gts - lhs
    expect = pts - lhs

    assert_eq(res, expect)
