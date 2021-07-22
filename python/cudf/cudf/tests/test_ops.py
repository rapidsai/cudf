# Copyright (c) 2021, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq, gen_rand


def test_sqrt_float():
    assert cudf.sqrt(16.0) == 4.0
    assert_eq(cudf.sqrt(cudf.Series([4.0, 9, 16])), cudf.Series([2.0, 3, 4]))
    assert_eq(
        cudf.sqrt(cudf.DataFrame({"x": [4.0, 9, 16]})),
        cudf.DataFrame({"x": [2.0, 3, 4]}),
    )


def test_sqrt_integer():
    assert cudf.sqrt(16) == 4
    assert_eq(cudf.sqrt(cudf.Series([4, 9, 16])), cudf.Series([2, 3, 4]))
    assert_eq(
        cudf.sqrt(cudf.DataFrame({"x": [4, 9, 16]})),
        cudf.DataFrame({"x": [2, 3, 4]}),
    )


def math_op_test(
    dtype, fn, nelem=128, test_df=False, positive_only=False, check_dtype=True
):
    np.random.seed(0)
    randvals = gen_rand(dtype, nelem, positive_only=positive_only)
    h_series = pd.Series(randvals.astype(dtype))
    d_series = cudf.Series(h_series)

    if test_df:
        d_in = cudf.DataFrame()
        d_in[0] = d_series
        h_in = pd.DataFrame()
        h_in[0] = h_series
    else:
        d_in = d_series
        h_in = h_series

    expect = fn(h_in)
    got = fn(d_in)

    assert_eq(expect, got, check_dtype=check_dtype)


params_real_types = [np.float64, np.float32]
int_type = [np.int64, np.int32]


# trig


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_sin(dtype, test_df):
    math_op_test(dtype, np.sin, test_df=test_df)


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_cos(dtype, test_df):
    math_op_test(dtype, np.cos, test_df=test_df)


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_tan(dtype, test_df):
    math_op_test(dtype, np.tan, test_df=test_df)


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_asin(dtype, test_df):
    math_op_test(dtype, np.arcsin, test_df=test_df)


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_acos(dtype, test_df):
    math_op_test(dtype, np.arccos, test_df=test_df, check_dtype=False)


@pytest.mark.parametrize("dtype", int_type)
@pytest.mark.parametrize("test_df", [False, True])
def test_acos_integer(dtype, test_df):
    math_op_test(dtype, np.arccos, test_df=test_df, check_dtype=False)


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_atan(dtype, test_df):
    math_op_test(dtype, np.arctan, test_df=test_df)


# exponential


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_exp(dtype, test_df):
    math_op_test(dtype, np.exp, test_df=test_df)


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_log(dtype, test_df):
    math_op_test(dtype, np.log, test_df=test_df, positive_only=True)


# power


@pytest.mark.parametrize("dtype", params_real_types)
@pytest.mark.parametrize("test_df", [False, True])
def test_sqrt(dtype, test_df):
    math_op_test(dtype, np.sqrt, test_df=test_df, positive_only=True)
