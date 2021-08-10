import pytest
import pandas as pd
import cudf
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
def test_ewm_basic_mean(data, params):
    """
    the most basic test asserts that we obtain
    the same numerical values as pandas for various
    sets of keyword arguemnts that effect the raw 
    coefficients of the formula
    """

    gsr = cudf.Series(data, dtype='float64')
    psr = gsr.to_pandas()

    expect = psr.ewm(**params).mean()
    got = gsr.ewm(**params).mean()
