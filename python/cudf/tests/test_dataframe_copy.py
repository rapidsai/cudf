# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd

from cudf.dataframe.dataframe import DataFrame

"""
DataFrame copy expectations
* A shallow copy constructs a new compound object and then (to the extent
  possible) inserts references into it to the objects found in the original.
* A deep copy constructs a new compound object and then, recursively, inserts
  copies into it of the objects found in the original.

  A cuDF DataFrame is a compound object containing a few members:
  _index, _size, _cols, where _cols is an OrderedDict
"""
from copy import copy
from copy import deepcopy
@pytest.mark.parametrize('copy_parameters', [
    {'fn':'lambda x:x.copy()', 'expected':False},
    {'fn':'lambda x:x.copy(deep=True)', 'expected':False},
    {'fn':'lambda x:copy(x)', 'expected':False},
    {'fn':'lambda x:deepcopy(x)', 'expected':False},
    {'fn':'lambda x:x.copy(deep=False)', 'expected':True},
    ])
def test_dataframe_copy(copy_parameters):
    pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       columns=['a', 'b', 'c'])
    cdf = DataFrame.from_pandas(pdf)
    copy_pdf = eval(copy_parameters['fn'])(pdf)
    copy_cdf = eval(copy_parameters['fn'])(cdf)
    copy_pdf['b'] = [0, 0, 0]
    copy_cdf['b'] = [0, 0, 0]
    pdf_pass = np.array_equal(pdf['b'].values, copy_pdf['b'].values)
    cdf_pass = np.array_equal(cdf['b'].to_array(), copy_cdf['b'].to_array())
    print(pdf)
    print(copy_pdf)
    assert cdf_pass == copy_parameters['expected']
    assert pdf_pass == copy_parameters['expected']

"""
DataFrame copy bounds checking - sizes 0 through 10 perform as expected
"""
@pytest.mark.parametrize('copy_fn', [
    'lambda x:x.copy()',
    'lambda x:x.copy(deep=True)',
    'lambda x:copy(x)',
    'lambda x:deepcopy(x)',
    'lambda x:x.copy(deep=False)',
    ])
@pytest.mark.parametrize('ncols', [0, 1, 2, 10])
@pytest.mark.parametrize(
    'data_type',
    ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'datetime64[ms]']
)
def test_cudf_dataframe_copy(copy_fn, ncols, data_type):
    pdf = pd.DataFrame()
    for i in range(ncols):
        pdf[chr(i+ord('a'))] = np.random.randint(0, 1000, 20).astype(data_type)
    df = DataFrame.from_pandas(pdf)
    copy_df = eval(copy_fn)(df)
    assert df.to_string().split() == copy_df.to_string().split()

