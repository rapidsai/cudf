# Copyright (c) 2025, NVIDIA CORPORATION.


import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
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
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize("name", ["abcd", None])
def test_timedelta_index_properties(data, timedelta_types_as_str, name):
    gdi = cudf.TimedeltaIndex(data, dtype=timedelta_types_as_str, name=name)
    pdi = gdi.to_pandas()

    def local_assert(expected, actual):
        if actual._column.null_count:
            assert_eq(expected, actual.astype("float64"))
        else:
            assert_eq(expected, actual)

    expected_days = pdi.days
    actual_days = gdi.days

    local_assert(expected_days, actual_days)

    expected_seconds = pdi.seconds
    actual_seconds = gdi.seconds

    local_assert(expected_seconds, actual_seconds)

    expected_microseconds = pdi.microseconds
    actual_microseconds = gdi.microseconds

    local_assert(expected_microseconds, actual_microseconds)

    expected_nanoseconds = pdi.nanoseconds
    actual_nanoseconds = gdi.nanoseconds

    local_assert(expected_nanoseconds, actual_nanoseconds)

    expected_components = pdi.components
    actual_components = gdi.components

    if actual_components.isnull().any().any():
        assert_eq(expected_components, actual_components.astype("float"))
    else:
        assert_eq(
            expected_components,
            actual_components,
            check_index_type=not actual_components.empty,
        )


def test_tdi_unit():
    pd_tdi = pd.TimedeltaIndex(
        ["1 day", "2 days", "3 days"], dtype="timedelta64[ns]"
    )
    cudf_tdi = cudf.from_pandas(pd_tdi)

    result = pd_tdi.unit
    expected = cudf_tdi.unit
    assert result == expected


def test_tdi_asi8():
    pd_tdi = pd.TimedeltaIndex(["1 day", "2 days", "3 days"])
    cudf_tdi = cudf.from_pandas(pd_tdi)

    result = pd_tdi.asi8
    expected = cudf_tdi.asi8
    assert_eq(result, expected)


def test_error_values():
    s = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    with pytest.raises(NotImplementedError, match="cupy does not support"):
        s.values
