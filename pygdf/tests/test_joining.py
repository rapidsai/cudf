from itertools import product

import pytest

import numpy as np

from pygdf.dataframe import DataFrame, Series


# sort_nelem_args = [2, 257]
# sort_dtype_args = [np.int32, np.int64, np.float32, np.float64]

# @pytest.mark.parametrize('nelem,dtype',
#                          list(product(sort_nelem_args,
#                                       sort_dtype_args)))
def test_dataframe_join(nelem=5, dtype=np.int32):
    np.random.seed(0)
    df = DataFrame()
    df['a'] = [0, 0, 4, 5, 5]
    df['b'] = [0, 0, 2, 3, 5]

    def work(df):
        df1 = df.set_index('a')
        df2 = df.set_index('b')
        joined = df1.join(df2)
        return joined

    expect = work(df.to_pandas())
    got = work(df)

    print(expect)
    print(got.to_string(nrows=None))

    assert list(expect.columns) == list(got.columns)
    assert list(expect.index) == list(got.index)
    assert list(expect['b']) == list(got['b'])
    # not equal due to NaN vs None
    assert list(expect['a']) != list(got['a'])
    # equal after filling NaN/None with -1
    assert list(expect['a'].fillna(-1)) == list(got['a'].fillna(-1))
