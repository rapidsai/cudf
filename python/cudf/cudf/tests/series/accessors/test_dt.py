# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(
    params=[
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [
            136457654736252,
            134736784364431,
            245345345545332,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ]
)
def timedelta_data(request):
    return request.param


def test_timedelta_dt_components(timedelta_data, timedelta_types_as_str):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.dt.components
    actual = gsr.dt.components

    if gsr.isnull().any():
        assert_eq(expected, actual.astype("float"))
    else:
        assert_eq(expected, actual)


def test_timedelta_dt_properties(timedelta_data, timedelta_types_as_str):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    def local_assert(expected, actual, **kwargs):
        if gsr.isnull().any():
            assert_eq(expected, actual.astype("float"), **kwargs)
        else:
            assert_eq(expected, actual, **kwargs)

    expected_days = psr.dt.days
    actual_days = gsr.dt.days

    local_assert(expected_days, actual_days, check_dtype=False)

    expected_seconds = psr.dt.seconds
    actual_seconds = gsr.dt.seconds

    local_assert(expected_seconds, actual_seconds, check_dtype=False)

    expected_microseconds = psr.dt.microseconds
    actual_microseconds = gsr.dt.microseconds

    local_assert(expected_microseconds, actual_microseconds, check_dtype=False)

    expected_nanoseconds = psr.dt.nanoseconds
    actual_nanoseconds = gsr.dt.nanoseconds

    local_assert(expected_nanoseconds, actual_nanoseconds, check_dtype=False)


def test_timedelta_series_total_seconds(
    timedelta_data, timedelta_types_as_str
):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.dt.total_seconds()
    actual = gsr.dt.total_seconds()
    assert_eq(expected, actual)
