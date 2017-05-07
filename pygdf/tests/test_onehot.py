import pytest

import numpy as np

from numba import cuda

from pygdf.dataframe import DataFrame


def test_onehot_simple():
    np.random.seed(0)
    df = DataFrame()
    # Populate with data [0, 10)
    df['vals'] = np.arange(10, dtype=np.int32)
    # One Hot (Series)
    for i, col in enumerate(df['vals'].one_hot_encoding(list(range(10)))):
        arr = col.to_array()
        # Verify 1 in the right position
        np.testing.assert_equal(arr[i], 1)
        # Every other slots are 0s
        np.testing.assert_equal(arr[:i], 0)
        np.testing.assert_equal(arr[i + 1:], 0)
    # One Hot (DataFrame)
    df2 = df.one_hot_encoding(column='vals',
                              prefix='vals',
                              cats=list(range(10)))
    assert df2.columns[0] == 'vals'
    for i in range(1, len(df2.columns)):
        assert df2.columns[i] == 'vals_%s' % (i - 1)
    got = df2.as_matrix(columns=df2.columns[1:])
    expect = np.identity(got.shape[0])
    np.testing.assert_equal(got, expect)


def test_onehot_random():
    df = DataFrame()
    low = 10
    high = 17
    size = 10
    df['src'] = src = np.random.randint(low=low, high=high, size=size)
    df2 = df.one_hot_encoding(column='src', prefix='out_',
                              cats=tuple(range(10, 17)))
    mat = df2.as_matrix(columns=df2.columns[1:])

    for val in range(low, high):
        colidx = val - low
        arr = mat[:, colidx]
        mask = src == val
        np.testing.assert_equal(arr, mask)


if __name__ == '__main__':
    test_onehot_random()

