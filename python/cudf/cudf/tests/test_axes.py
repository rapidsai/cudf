import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq

# **Series**
@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 4],
        ['a', 'b', 'c'],
        [1.2, 2.2, 4.5],
        [np.nan, np.nan],
        [None, None, None],
    ],
)

def test_axes_series(data):

    # print(data)
    psr = pd.Series(data)
    csr = cudf.Series(data)
    
    expected = psr.axes
    actual = csr.axes
    
    for i in range(len(actual)):
        assert_eq(expected[i], actual[i])
        

# ** DataFrame **
@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2]},
        {"a": [1, 2, 3], "b": [3, 4, 5]},
        {"a": [1, 2, 3, 4], "b": [3, 4, 5, 6], "c": [1, 3, 5, 7]},
        {"a": [np.nan, 2, 3, 4], "b": [3, 4, np.nan, 6], "c": [1, 3, 5, 7]},
        {1: [1, 2, 3], 2: [3, 4, 5]},
        {"a": [1, None, None], "b": [3, np.nan, np.nan]},
        {1: ['a', 'b', 'c'], 2: ['q', 'w', 'u']},
        {1: ['a', np.nan, 'c'], 2: ['q', None, 'u']},
    ],
)

def test_axes_dataframe(data):

    # print(data)
    psr = pd.DataFrame(data)
    csr = cudf.DataFrame(data)
    
    expected = psr.axes
    actual = csr.axes
    
    for i in range(len(actual)):
        assert_eq(expected[i], actual[i])  

## When the DataFrame is empty and when there are column names - the expected output is slight different compared to logic
#     return [Index([],dtype = 'object'), self.columns]
            
## When the DataFrame is not empty, and when the column names are not available - the expected output is slightly different compared to logic 
#     return [self.index, RangeIndex(0,1)]


