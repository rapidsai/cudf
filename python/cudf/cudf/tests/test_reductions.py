# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from __future__ import division, print_function

import re
from decimal import Decimal
from itertools import product

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import Series
from cudf.core.dtypes import Decimal64Dtype
from cudf.testing import _utils as utils
from cudf.testing._utils import NUMERIC_TYPES, assert_eq, gen_rand

params_dtype = NUMERIC_TYPES

params_sizes = [1, 2, 3, 127, 128, 129, 200, 10000]

params = list(product(params_dtype, params_sizes))


@pytest.mark.parametrize("dtype,nelem", params)
def test_sum(dtype, nelem):
    dtype = np.dtype(dtype).type
    data = gen_rand(dtype, nelem)
    sr = Series(data)

    got = sr.sum()
    expect = dtype(data.sum())

    significant = 4 if dtype == np.float32 else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


def test_sum_string():
    s = Series(["Hello", "there", "World"])

    got = s.sum()
    expected = "HellothereWorld"

    assert got == expected

    s = Series(["Hello", None, "World"])

    got = s.sum()
    expected = "HelloWorld"

    assert got == expected


@pytest.mark.parametrize(
    "dtype",
    [Decimal64Dtype(6, 3), Decimal64Dtype(10, 6), Decimal64Dtype(16, 7)],
)
@pytest.mark.parametrize("nelem", params_sizes)
def test_sum_decimal(dtype, nelem):
    data = [str(x) for x in gen_rand("int64", nelem) / 100]

    expected = pd.Series([Decimal(x) for x in data]).sum()
    got = cudf.Series(data).astype(dtype).sum()

    assert_eq(expected, got)


@pytest.mark.parametrize("dtype,nelem", params)
def test_product(dtype, nelem):
    np.random.seed(0)
    dtype = np.dtype(dtype).type
    if np.dtype(dtype).kind in {"u", "i"}:
        data = np.ones(nelem, dtype=dtype)
        # Set at most 30 items to [0..2) to keep the value within 2^32
        for _ in range(30):
            data[np.random.randint(low=0, high=nelem, size=1)] = (
                np.random.uniform() * 2
            )
    else:
        data = gen_rand(dtype, nelem)

    sr = Series(data)

    got = sr.product()
    expect = np.product(data)

    significant = 4 if dtype == np.float32 else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


@pytest.mark.parametrize(
    "dtype",
    [Decimal64Dtype(6, 2), Decimal64Dtype(8, 4), Decimal64Dtype(10, 5)],
)
def test_product_decimal(dtype):
    data = [str(x) for x in gen_rand("int8", 3) / 10]

    expected = pd.Series([Decimal(x) for x in data]).product()
    got = cudf.Series(data).astype(dtype).product()

    assert_eq(expected, got)


accuracy_for_dtype = {np.float64: 6, np.float32: 5}


@pytest.mark.parametrize("dtype,nelem", params)
def test_sum_of_squares(dtype, nelem):
    dtype = np.dtype(dtype).type
    data = gen_rand(dtype, nelem)
    sr = Series(data)

    got = sr.sum_of_squares()
    # got = dtype(got)
    expect = (data ** 2).sum()

    if np.dtype(dtype).kind in {"u", "i"}:
        if 0 <= expect <= np.iinfo(dtype).max:
            np.testing.assert_array_almost_equal(expect, got)
        else:
            print("overflow, passing")
    else:
        np.testing.assert_approx_equal(
            expect, got, significant=accuracy_for_dtype[dtype]
        )


@pytest.mark.parametrize(
    "dtype",
    [Decimal64Dtype(6, 2), Decimal64Dtype(8, 4), Decimal64Dtype(10, 5)],
)
def test_sum_of_squares_decimal(dtype):
    data = [str(x) for x in gen_rand("int8", 3) / 10]

    expected = pd.Series([Decimal(x) for x in data]).pow(2).sum()
    got = cudf.Series(data).astype(dtype).sum_of_squares()

    assert_eq(expected, got)


@pytest.mark.parametrize("dtype,nelem", params)
def test_min(dtype, nelem):
    dtype = np.dtype(dtype).type
    data = gen_rand(dtype, nelem)
    sr = Series(data)

    got = sr.min()
    expect = dtype(data.min())

    assert expect == got


@pytest.mark.parametrize(
    "dtype",
    [Decimal64Dtype(6, 3), Decimal64Dtype(10, 6), Decimal64Dtype(16, 7)],
)
@pytest.mark.parametrize("nelem", params_sizes)
def test_min_decimal(dtype, nelem):
    data = [str(x) for x in gen_rand("int64", nelem) / 100]

    expected = pd.Series([Decimal(x) for x in data]).min()
    got = cudf.Series(data).astype(dtype).min()

    assert_eq(expected, got)


@pytest.mark.parametrize("dtype,nelem", params)
def test_max(dtype, nelem):
    dtype = np.dtype(dtype).type
    data = gen_rand(dtype, nelem)
    sr = Series(data)

    got = sr.max()
    expect = dtype(data.max())

    assert expect == got


@pytest.mark.parametrize(
    "dtype",
    [Decimal64Dtype(6, 3), Decimal64Dtype(10, 6), Decimal64Dtype(16, 7)],
)
@pytest.mark.parametrize("nelem", params_sizes)
def test_max_decimal(dtype, nelem):
    data = [str(x) for x in gen_rand("int64", nelem) / 100]

    expected = pd.Series([Decimal(x) for x in data]).max()
    got = cudf.Series(data).astype(dtype).max()

    assert_eq(expected, got)


@pytest.mark.parametrize("nelem", params_sizes)
def test_sum_masked(nelem):
    dtype = np.float64
    data = gen_rand(dtype, nelem)

    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]
    null_count = utils.count_zero(bitmask)

    sr = Series.from_masked_array(data, mask, null_count)

    got = sr.sum()
    res_mask = np.asarray(bitmask, dtype=np.bool_)[: data.size]
    expect = data[res_mask].sum()

    significant = 4 if dtype == np.float32 else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


def test_sum_boolean():
    s = Series(np.arange(100000))
    got = (s > 1).sum(dtype=np.int32)
    expect = 99998

    assert expect == got

    got = (s > 1).sum(dtype=np.bool_)
    expect = True

    assert expect == got


def test_date_minmax():
    np_data = np.random.normal(size=10 ** 3)
    gdf_data = Series(np_data)

    np_casted = np_data.astype("datetime64[ms]")
    gdf_casted = gdf_data.astype("datetime64[ms]")

    np_min = np_casted.min()
    gdf_min = gdf_casted.min()
    assert np_min == gdf_min

    np_max = np_casted.max()
    gdf_max = gdf_casted.max()
    assert np_max == gdf_max


@pytest.mark.parametrize(
    "op", ["sum", "product", "var", "kurt", "kurtosis", "skew"],
)
def test_datetime_unsupported_reductions(op):
    gsr = cudf.Series([1, 2, 3, None], dtype="datetime64[ns]")
    psr = gsr.to_pandas()

    utils.assert_exceptions_equal(
        lfunc=getattr(psr, op),
        rfunc=getattr(gsr, op),
        expected_error_message=re.escape(
            "cannot perform "
            + ("kurtosis" if op == "kurt" else op)
            + " with type datetime64[ns]"
        ),
    )


@pytest.mark.parametrize("op", ["product", "var", "kurt", "kurtosis", "skew"])
def test_timedelta_unsupported_reductions(op):
    gsr = cudf.Series([1, 2, 3, None], dtype="timedelta64[ns]")
    psr = gsr.to_pandas()

    utils.assert_exceptions_equal(
        lfunc=getattr(psr, op),
        rfunc=getattr(gsr, op),
        expected_error_message=re.escape(
            "cannot perform "
            + ("kurtosis" if op == "kurt" else op)
            + " with type timedelta64[ns]"
        ),
    )


@pytest.mark.parametrize("op", ["sum", "product", "std", "var"])
def test_categorical_reductions(op):
    gsr = cudf.Series([1, 2, 3, None], dtype="category")
    psr = gsr.to_pandas()

    utils.assert_exceptions_equal(
        getattr(psr, op), getattr(gsr, op), compare_error_message=False
    )
