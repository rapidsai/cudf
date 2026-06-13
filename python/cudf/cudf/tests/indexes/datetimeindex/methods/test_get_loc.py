# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def _assert_scalar_or_slice_get_loc_eq(pi, key):
    gi = cudf.from_pandas(pi)

    expected = pi.get_loc(key)
    got = gi.get_loc(key)

    assert type(got) is type(expected)
    assert_eq(expected, got)


def _assert_array_get_loc_eq(pi, key):
    gi = cudf.from_pandas(pi)

    expected = pi.get_loc(key)
    got = gi.get_loc(key)

    assert isinstance(expected, np.ndarray)
    assert isinstance(got, (cp.ndarray, np.ndarray))
    assert got.dtype == expected.dtype
    assert_eq(expected, got)


@pytest.mark.xfail(
    reason=(
        "_datetime_index_resolution only checks min/max and misses interior "
        "seconds, so get_loc falls through instead of partial-string matching"
    ),
    strict=True,
)
def test_monotonic_increasing_coarse_key_returns_slice():
    pi = pd.DatetimeIndex(["2020-01-01", "2020-01-01 00:00:01", "2020-01-02"])

    _assert_scalar_or_slice_get_loc_eq(pi, "2020-01-01")


@pytest.mark.xfail(
    reason=(
        "DatetimeIndex.get_loc currently falls through to superclass string "
        "lookup for exact-resolution datetime strings"
    ),
    strict=True,
)
def test_monotonic_increasing_exact_key_returns_int():
    pi = pd.DatetimeIndex(["2020-01-01", "2020-01-01 00:00:01", "2020-01-02"])

    _assert_scalar_or_slice_get_loc_eq(pi, "2020-01-01 00:00:01")


def test_monotonic_increasing_year_key_on_day_index():
    pi = pd.date_range("2020-01-01", "2020-03-01", freq="D")

    _assert_scalar_or_slice_get_loc_eq(pi, "2020")


def test_monotonic_increasing_month_key_on_day_index():
    pi = pd.date_range("2020-01-01", "2020-03-31", freq="D")

    _assert_scalar_or_slice_get_loc_eq(pi, "2020-01")


@pytest.mark.xfail(
    reason="Missing coarse partial datetime keys currently return an empty slice",
    strict=True,
)
def test_monotonic_increasing_key_not_found():
    pi = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    gi = cudf.from_pandas(pi)

    assert_exceptions_equal(
        lfunc=pi.get_loc,
        rfunc=gi.get_loc,
        lfunc_args_and_kwargs=(["2025"], {}),
        rfunc_args_and_kwargs=(["2025"], {}),
    )


@pytest.mark.xfail(
    reason=(
        "_datetime_index_resolution only checks min/max and misses interior "
        "seconds on decreasing indexes"
    ),
    strict=True,
)
def test_monotonic_decreasing_coarse_key_returns_ndarray():
    pi = pd.DatetimeIndex(["2020-01-02", "2020-01-01 00:00:01", "2020-01-01"])

    _assert_array_get_loc_eq(pi, "2020-01-01")


@pytest.mark.xfail(
    reason=(
        "Non-monotonic partial-string get_loc currently takes the wrong cudf "
        "path after resolution inference"
    ),
    strict=True,
)
def test_non_monotonic_coarse_key_raises_keyerror():
    pi = pd.DatetimeIndex(["2020-01-02", "2020-01-01 00:00:01", "2020-01-03"])

    _assert_array_get_loc_eq(pi, "2020-01-01")


@pytest.mark.xfail(
    reason=(
        "_datetime_index_resolution only checks min/max and misses finer "
        "interior timestamp resolution"
    ),
    strict=True,
)
def test_resolution_inference_interior_seconds():
    pi = pd.DatetimeIndex(["2020-01-01", "2020-01-01 12:00:00", "2020-01-02"])

    _assert_scalar_or_slice_get_loc_eq(pi, "2020-01-01")


@pytest.mark.xfail(
    reason=(
        "DatetimeIndex.get_loc currently falls through to superclass string "
        "lookup for duplicate exact-resolution datetime strings"
    ),
    strict=True,
)
def test_monotonic_increasing_duplicates_return_slice():
    pi = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])

    _assert_scalar_or_slice_get_loc_eq(pi, "2020-01-01")
