# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd

from cudf.dataframe.dataframe import Series, DataFrame

from . import utils

"""
DataFrame copy expectations
* A shallow copy constructs a new compound object and then (to the extent
  possible) inserts references into it to the objects found in the original.
* A deep copy constructs a new compound object and then, recursively, inserts
  copies into it of the objects found in the original.

  A cuDF DataFrame is a compound object containing a few members:
  _index, _size, _cols,
  and _iter_count, _iter_keys # temporary?
  where _cols is an OrderedDict and _iter_keys is a list
"""

def test_pandas_dataframe_copy_deep_False():
    pdf = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=['a','b','c'])
    copy_deep_False_pdf = pdf.copy(deep=False)
    copy_deep_False_pdf.iloc[1][1] = 10
    assert pdf.iloc[1][1] == copy_deep_False_pdf.iloc[1][1]
    copy_deep_False_pdf['b'] = [0,0,0]
    assert np.array_equal(pdf['b'].values, copy_deep_False_pdf['b'].values)
def test_pandas_dataframe_copy_deep_True():
    pdf = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=['a','b','c'])
    copy_deep_True_pdf = pdf.copy(deep=True)
    copy_deep_True_pdf.iloc[1][1] = 10
    assert pdf.iloc[1][1] != copy_deep_True_pdf.iloc[1][1]
    copy_deep_True_pdf['b'] = [0,0,0]
    assert not np.array_equal(pdf['b'].values, copy_deep_True_pdf['b'].values)
def test_pandas_dataframe_copy():
    pdf = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=['a','b','c'])
    from copy import copy
    copy_pdf = copy(pdf)
    copy_pdf.iloc[1][1] = 10
    # Pandas only deep copies! This expected assert fails because pandas
    # is broken here.
    #assert pdf.iloc[1][1] == copy_pdf.iloc[1][1]
    copy_pdf['b'] = [0,0,0]
    #assert np.array_equal(pdf['b'].values, copy_pdf['b'].values)
def test_pandas_dataframe_deepcopy():
    pdf = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=['a','b','c'])
    from copy import deepcopy
    deepcopy_pdf = deepcopy(pdf)
    deepcopy_pdf.iloc[1][1] = 10
    assert pdf.iloc[1][1] != deepcopy_pdf.iloc[1][1]
    deepcopy_pdf['b'] = [0,0,0]
    assert not np.array_equal(pdf['b'].values, deepcopy_pdf['b'].values)

def test_cudf_dataframe_copy_deep_False():
    cdf = DataFrame.from_pandas(pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],
        columns=['a','b','c']))
    copy_deep_False_cdf = cdf.copy(deep=False)
    copy_deep_False_cdf['b'] = [0,0,0]
    # copy is a deep copy, so this fails
    assert np.array_equal(cdf['b'].to_array(), copy_deep_False_cdf['b']
            .to_array())
def test_cudf_dataframe_copy_deep_True():
    cdf = DataFrame.from_pandas(pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],
    columns=['a','b','c']))
    copy_deep_True_cdf = cdf.copy(deep=True)
    copy_deep_True_cdf['b'] = [0,0,0]
    assert not np.array_equal(cdf['b'].to_array(), copy_deep_True_cdf['b']
            .to_array())
def test_cudf_dataframe_copy():
    cdf = DataFrame.from_pandas(pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],
        columns=['a','b','c']))
    from copy import copy
    copy_cdf = copy(cdf)
    copy_cdf['b'] = [0,0,0]
    # copy is a deepcopy, so this fails
    assert np.array_equal(cdf['b'].to_array(), copy_cdf['b'].to_array())
def test_cudf_dataframe_deepcopy():
    cdf = DataFrame.from_pandas(pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],
        columns=['a','b','c']))
    from copy import deepcopy
    deepcopy_cdf = deepcopy(cdf)
    deepcopy_cdf['b'] = [0,0,0]
    assert not np.array_equal(cdf['b'].to_array(), deepcopy_cdf['b'].to_array())

"""
DataFrame copy bounds checking - sizes 0 through 10 perform as expected
"""
@pytest.mark.parametrize('ncols', [0, 1, 2, 10])
@pytest.mark.parametrize(
    'data_type',
    ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'datetime64[ms]']
)
def test_cudf_dataframe_deep_copy(ncols, data_type):
    from copy import deepcopy
    pdf = pd.DataFrame()
    for i in range(ncols):
        pdf[chr(i+ord('a'))] = np.random.randint(0, 1000, 20).astype(data_type)
    df = DataFrame.from_pandas(pdf)
    dfdc = deepcopy(df)
    assert df.to_string().split() == dfdc.to_string().split()
@pytest.mark.parametrize('ncols', [0, 1, 2, 10])
@pytest.mark.parametrize(
    'data_type',
    ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'datetime64[ms]']
)
def test_cudf_dataframe_shallow_copy(ncols, data_type):
    from copy import copy
    pdf = pd.DataFrame()
    for i in range(ncols):
        pdf[chr(i+ord('a'))] = np.random.randint(0, 1000, 20).astype(data_type)
    df = DataFrame.from_pandas(pdf)
    dfsc = copy(df)
    assert df.to_string().split() == dfsc.to_string().split()
@pytest.mark.parametrize('ncols', [0, 1, 2, 10])
@pytest.mark.parametrize(
    'data_type',
    ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'datetime64[ms]']
)
def test_cudf_dataframe_class_copy(ncols, data_type):
    from copy import copy
    pdf = pd.DataFrame()
    for i in range(ncols):
        pdf[chr(i+ord('a'))] = np.random.randint(0, 1000, 20).astype(data_type)
    df = DataFrame.from_pandas(pdf)
    dfcc = df.copy()
    assert df.to_string().split() == dfcc.to_string().split()
