import pytest
import pandas as pd
import cudf
import numpy as np
from cudf.testing._utils import assert_eq
from dataclasses import dataclass

@dataclass
class EWMParams:        
    def __init__(self, data, nulls, adjust, bias=None, com=None, span=None, halflife=None, alpha=None):
        self.nulls = nulls
        if self.nulls:
            data[1] = cudf.NA
            data[3] = cudf.NA
            data[4] = cudf.NA

        initial_ewm_args = {
            'adjust': adjust,
            'com': com,
            'span': span,
            'halflife': halflife,
            'alpha': alpha
        }
        self.ewm_args = {k: v for k, v in initial_ewm_args.items() if v is not None}
        self.call_args = {'bias': bias} if bias is not None else {}

    def __repr__(self):
        return str({**{"nulls": self.nulls},**self.ewm_args, **self.call_args})




@pytest.mark.parametrize('data', [
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [5.0, cudf.NA, 3.0, cudf.NA, 8.5],
    [5.0, cudf.NA, 3.0, cudf.NA, cudf.NA, 4.5]
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
