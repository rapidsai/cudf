import pytest
import pandas as pd
import cudf
import numpy as np
from cudf.testing._utils import assert_eq

@pytest.mark.parametrize('data', [
    [1.0, 2.0, 3.0, 4.0, 5.0]
])
@pytest.mark.parametrize('params', [
    {'com': 0.1},
    {'com': 0.5},
    {'span': 1.5},
    {'span': 2.5},
    {'halflife': 0.5},
    {'halflife': 1.5},
    {'alpha': 0.1},
    {'alpha': 0.5},
])
@pytest.mark.parametrize('adjust', [True, False])
def test_ewma(data, params, adjust):
    """
    the most basic test asserts that we obtain
    the same numerical values as pandas for various
    sets of keyword arguemnts that effect the raw 
    coefficients of the formula
    """
    params["adjust"] = adjust

    gsr = cudf.Series(data, dtype='float64')
    psr = gsr.to_pandas()

    expect = psr.ewm(**params).mean()
    got = gsr.ewm(**params).mean()

    assert_eq(expect, got)

@pytest.mark.parametrize('data', [
    [1.0, 2.0, 3.0, 4.0, 5.0]
])
@pytest.mark.parametrize('params', [
    {'com': 0.1},
    {'com': 0.5},
    {'span': 1.5},
    {'span': 2.5},
    {'halflife': 0.5},
    {'halflife': 1.5},
    {'alpha': 0.1},
    {'alpha': 0.5},
])
@pytest.mark.parametrize('adjust', [True, False])
@pytest.mark.parametrize('bias', [True])
def test_ewmvar(data, params, adjust, bias):
    """
    the most basic test asserts that we obtain
    the same numerical values as pandas for various
    sets of keyword arguemnts that effect the raw 
    coefficients of the formula
    """
    params["adjust"] = adjust

    gsr = cudf.Series(data, dtype='float64')
    psr = gsr.to_pandas()

    expect = psr.ewm(**params).var(bias=bias)
    got = gsr.ewm(**params).var(bias=bias)

    assert_eq(expect, got)

@pytest.mark.parametrize('data', [
    [1.0, 2.0, 3.0, 4.0, 5.0]
])
@pytest.mark.parametrize('params', [
    {'com': 0.1},
    {'com': 0.5},
    {'span': 1.5},
    {'span': 2.5},
    {'halflife': 0.5},
    {'halflife': 1.5},
    {'alpha': 0.1},
    {'alpha': 0.5},
])
@pytest.mark.parametrize('adjust', [True, False])
@pytest.mark.parametrize('bias', [True])
def test_ewmstd(data, params, adjust, bias):
    """
    the most basic test asserts that we obtain
    the same numerical values as pandas for various
    sets of keyword arguemnts that effect the raw 
    coefficients of the formula
    """
    params["adjust"] = adjust

    gsr = cudf.Series(data, dtype='float64')
    psr = gsr.to_pandas()

    expect = psr.ewm(**params).std(bias=bias)
    got = gsr.ewm(**params).std(bias=bias)

    assert_eq(expect, got)

@pytest.mark.parametrize('data', [
    [1.0, 2.0, np.nan, 4.0, 5.0]
])
@pytest.mark.parametrize('params', [
    {'com': 0.1},
    {'com': 0.5},
    {'span': 1.5},
    {'span': 2.5},
    {'halflife': 0.5},
    {'halflife': 1.5},
    {'alpha': 0.1},
    {'alpha': 0.5},
])
@pytest.mark.parametrize('adjust', [True, False])
def test_ewma_nulls(data, params, adjust):
    """
    the most basic test asserts that we obtain
    the same numerical values as pandas for various
    sets of keyword arguemnts that effect the raw 
    coefficients of the formula
    """
    params["adjust"] = adjust

    gsr = cudf.Series(data, dtype='float64')
    psr = gsr.to_pandas()

    expect = psr.ewm(**params).mean()
    got = gsr.ewm(**params).mean()

    assert_eq(expect, got)
