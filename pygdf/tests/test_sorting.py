from itertools import product

import pytest

import numpy as np

from pygdf.dataframe import DataFrame, Series


sort_nelem_args = [2, 257]
sort_dtype_args = [np.int32, np.int64, np.float32, np.float64]

@pytest.mark.parametrize('nelem,dtype',
                         list(product(sort_nelem_args,
                                      sort_dtype_args)))
def test_dataframe_sort_values(nelem, dtype):
    np.random.seed(0)
    df = DataFrame()
    df['a'] = aa = np.random.random(nelem).astype(dtype)
    df['b'] = bb = np.random.random(nelem).astype(dtype)
    sorted_df = df.sort_values(by='a')
    # Check
    sorted_index = np.argsort(aa, kind='mergesort')
    np.testing.assert_array_equal(sorted_df.index.values, sorted_index)
    np.testing.assert_array_equal(sorted_df['a'], aa[sorted_index])
    np.testing.assert_array_equal(sorted_df['b'], bb[sorted_index])


@pytest.mark.parametrize('nelem,dtype,asc',
                         list(product(sort_nelem_args,
                                      sort_dtype_args,
                                      [True, False])))
def test_series_argsort(nelem, dtype, asc):
    np.random.seed(0)
    sr = Series(np.random.random(nelem).astype(dtype))
    res = sr.argsort(ascending=asc)

    if asc:
        expected = np.argsort(sr.to_array(), kind='mergesort')
    else:
        expected = np.argsort(-sr.to_array(), kind='mergesort')
    np.testing.assert_array_equal(expected, res.to_array())
