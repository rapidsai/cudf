# Copyright (c) 2021, NVIDIA CORPORATION.

import re

import numpy as np
import pytest

from cudf import DateOffset

INT64MAX = np.iinfo("int64").max


@pytest.mark.parametrize("period", [1.5, 0.5, "string", "1", "1.0"])
@pytest.mark.parametrize("freq", ["years", "months"])
def test_construction_invalid(period, freq):
    kwargs = {freq: period}
    with pytest.raises(ValueError):
        DateOffset(**kwargs)


@pytest.mark.parametrize(
    "unit", ["nanoseconds", "microseconds", "milliseconds", "seconds"]
)
def test_construct_max_offset(unit):
    DateOffset(**{unit: np.iinfo("int64").max})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seconds": INT64MAX + 1},
        {"seconds": INT64MAX, "minutes": 1},
        {"minutes": INT64MAX},
    ],
)
def test_offset_construction_overflow(kwargs):
    with pytest.raises(NotImplementedError):
        DateOffset(**kwargs)


@pytest.mark.parametrize(
    "unit",
    [
        "years",
        "months",
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
)
@pytest.mark.parametrize("period", [0.5, -0.5, 0.71])
def test_offset_no_fractional_periods(unit, period):
    with pytest.raises(
        ValueError, match=re.escape("Non-integer periods not supported")
    ):
        DateOffset(**{unit: period})
