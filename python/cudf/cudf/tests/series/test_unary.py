# Copyright (c) 2019-2025, NVIDIA CORPORATION.

from decimal import Decimal

import numpy as np

from cudf import Series
from cudf.testing import assert_eq


def test_series_abs(numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    arr = (rng.random(100) * 100).astype(numeric_types_as_str)
    sr = Series(arr)
    np.testing.assert_equal(sr.abs().to_numpy(), np.abs(arr))
    np.testing.assert_equal(abs(sr).to_numpy(), abs(arr))


def test_series_invert(integer_types_as_str):
    arr = np.array([0, 1, 2], dtype=integer_types_as_str)
    sr = Series(arr)
    np.testing.assert_equal((~sr).to_numpy(), np.invert(arr))
    np.testing.assert_equal((~sr).to_numpy(), ~arr)


def test_series_neg():
    rng = np.random.default_rng(seed=0)
    arr = rng.random(100) * 100
    sr = Series(arr)
    np.testing.assert_equal((-sr).to_numpy(), -arr)


def test_series_bool_neg():
    sr = Series([True, False, True, None, False, None, True, True])
    psr = sr.to_pandas(nullable=True)
    assert_eq((-sr).to_pandas(nullable=True), -psr, check_dtype=True)


def test_series_decimal_neg():
    sr = Series([Decimal("0.0"), Decimal("1.23"), Decimal("4.567")])
    psr = sr.to_pandas()
    assert_eq((-sr).to_pandas(), -psr, check_dtype=True)
