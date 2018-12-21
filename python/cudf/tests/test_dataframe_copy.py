# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd

from cudf.dataframe.dataframe import DataFrame
from .utils import assert_eq

"""
DataFrame copy expectations
* A shallow copy constructs a new compound object and then (to the extent
  possible) inserts references into it to the objects found in the original.
* A deep copy constructs a new compound object and then, recursively, inserts
  copies into it of the objects found in the original.

  A cuDF DataFrame is a compound object containing a few members:
  _index, _size, _cols, where _cols is an OrderedDict
"""
from copy import copy  # noqa:F401
from copy import deepcopy  # noqa:F401


def test_dataframe_copy_shallow():
    pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       columns=['a', 'b', 'c'])
    gdf = DataFrame.from_pandas(pdf)
    copy_pdf = pdf.copy(deep=False)
    copy_gdf = gdf.copy(deep=False)
    copy_pdf['b'] = [0, 0, 0]
    copy_gdf['b'] = [0, 0, 0]
    assert_eq(pdf['b'].values, copy_pdf['b'].values)
    assert_eq(gdf['b'].to_array(), copy_gdf['b'].to_array())


@pytest.mark.parametrize('copy_parameters', [
    {'fn': lambda x: x.copy(), 'expected': False},
    {'fn': lambda x: x.copy(deep=True), 'expected': False},
    {'fn': lambda x: copy(x), 'expected': False},
    {'fn': lambda x: deepcopy(x), 'expected': False},
    {'fn': lambda x: x.copy(deep=False), 'expected': True},
    ])
def test_dataframe_copy(copy_parameters):
    pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       columns=['a', 'b', 'c'])
    gdf = DataFrame.from_pandas(pdf)
    copy_pdf = copy_parameters['fn'](pdf)
    copy_gdf = copy_parameters['fn'](gdf)
    copy_pdf['b'] = [0, 0, 0]
    copy_gdf['b'] = [0, 0, 0]
    pdf_pass = np.array_equal(pdf['b'].values, copy_pdf['b'].values)
    gdf_pass = np.array_equal(gdf['b'].to_array(), copy_gdf['b'].to_array())
    assert gdf_pass == copy_parameters['expected']
    assert pdf_pass == copy_parameters['expected']


"""
DataFrame copy bounds checking - sizes 0 through 10 perform as expected
"""


# lambda x: x.copy(deep=True),
# lambda x: copy(x),
# lambda x: deepcopy(x),
# lambda x: x.copy(deep=False),

@pytest.mark.parametrize('copy_fn', [
    lambda x: x.copy(),
    ])
@pytest.mark.parametrize('ncols', [1])
@pytest.mark.parametrize(
    'data_type',
    ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'datetime64[ms]',
        'category', ]
    # ['datetime64[ms]']
    # ['int8']
)
def test_cudf_dataframe_copy(copy_fn, ncols, data_type):
    pdf = pd.DataFrame()
    for i in range(ncols):
        pdf[chr(i+ord('a'))] = pd.Series(np.random.randint(0, 1000, 20),
                                         dtype=data_type)
    df = DataFrame.from_pandas(pdf)
    copy_df = copy_fn(df)
    assert_eq(df, copy_df)
