# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf


@pytest.mark.parametrize("value", [1, 1.1])
def test_nans_to_nulls_noop_copies_column(value):
    ser1 = cudf.Series([value])
    ser2 = ser1.nans_to_nulls()
    assert ser1._column is not ser2._column


def test_nans_to_nulls_sliced():
    ser1 = cudf.Series(
        [0, 1, float("nan"), 2, float("nan"), 3.5], nan_as_null=False
    )
    ser2 = ser1.iloc[2:]
    actual = ser2.nans_to_nulls().reset_index(drop=True)
    expected = cudf.Series([None, 2, None, 3.5], nan_as_null=True)
    cudf.testing.assert_series_equal(actual, expected)
