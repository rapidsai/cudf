# Copyright (c) 2019-2025, NVIDIA CORPORATION.

import itertools
import operator
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from cudf import Series
from cudf.testing import _utils as utils, assert_eq

_unaops = [operator.abs, operator.invert, operator.neg, np.ceil, np.floor]


@pytest.mark.parametrize("dtype", utils.NUMERIC_TYPES)
def test_series_abs(dtype):
    rng = np.random.default_rng(seed=0)
    arr = (rng.random(1000) * 100).astype(dtype)
    sr = Series(arr)
    np.testing.assert_equal(sr.abs().to_numpy(), np.abs(arr))
    np.testing.assert_equal(abs(sr).to_numpy(), abs(arr))


@pytest.mark.parametrize("dtype", utils.INTEGER_TYPES)
def test_series_invert(dtype):
    rng = np.random.default_rng(seed=0)
    arr = (rng.random(1000) * 100).astype(dtype)
    sr = Series(arr)
    np.testing.assert_equal((~sr).to_numpy(), np.invert(arr))
    np.testing.assert_equal((~sr).to_numpy(), ~arr)


def test_series_neg():
    rng = np.random.default_rng(seed=0)
    arr = rng.random(100) * 100
    sr = Series(arr)
    np.testing.assert_equal((-sr).to_numpy(), -arr)


@pytest.mark.parametrize("mth", ["min", "max", "sum", "product"])
def test_series_pandas_methods(mth):
    rng = np.random.default_rng(seed=0)
    arr = (1 + rng.random(5) * 100).astype(np.int64)
    sr = Series(arr)
    psr = pd.Series(arr)
    np.testing.assert_equal(getattr(sr, mth)(), getattr(psr, mth)())


@pytest.mark.parametrize("mth", ["min", "max", "sum", "product", "quantile"])
def test_series_pandas_methods_empty(mth):
    arr = np.array([])
    sr = Series(arr)
    psr = pd.Series(arr)
    np.testing.assert_equal(getattr(sr, mth)(), getattr(psr, mth)())


def generate_valid_scalar_unaop_combos():
    results = []

    # All ops valid for integer values
    int_values = [0, 1, -1]
    int_dtypes = utils.INTEGER_TYPES
    int_ops = _unaops

    results += list(itertools.product(int_values, int_dtypes, int_ops))

    float_values = [0.0, 1.0, -1.1]
    float_dtypes = utils.FLOAT_TYPES
    float_ops = [op for op in _unaops if op is not operator.invert]
    results += list(itertools.product(float_values, float_dtypes, float_ops))

    bool_values = [True, False]
    bool_dtypes = ["bool"]
    bool_ops = [op for op in _unaops if op is not operator.neg]
    results += list(itertools.product(bool_values, bool_dtypes, bool_ops))

    return results


def test_series_bool_neg():
    sr = Series([True, False, True, None, False, None, True, True])
    psr = sr.to_pandas(nullable=True)
    assert_eq((-sr).to_pandas(nullable=True), -psr, check_dtype=True)


def test_series_decimal_neg():
    sr = Series([Decimal("0.0"), Decimal("1.23"), Decimal("4.567")])
    psr = sr.to_pandas()
    assert_eq((-sr).to_pandas(), -psr, check_dtype=True)
