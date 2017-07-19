import numpy as np

from pygdf.dataframe import DataFrame


def test_dataframe_sort_values():
    np.random.seed(0)
    df = DataFrame()
    df['a'] = aa = np.random.random(100)
    df['b'] = bb = np.random.random(100)
    sorted_df = df.sort_values(by='a')
    # Check
    sorted_index = np.argsort(aa, kind='mergesort')
    np.testing.assert_array_equal(sorted_df.index.values, sorted_index)
    np.testing.assert_array_equal(sorted_df['a'], aa[sorted_index])
    np.testing.assert_array_equal(sorted_df['b'], bb[sorted_index])
