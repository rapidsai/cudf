import pytest
import pygdf
import numpy as np
from numba import cuda
from pygdf.dataframe import DataFrame, Series


def test_label_encode():
    df = DataFrame()
    np.random.seed(0)

    # initialize data frame
    df['cats'] = np.random.randint(7, size=10, dtype=np.int32)
    vals = df['cats'].unique_k(10)
    lab = dict(zip(vals, list(range(len(vals)))))

    # label encode series
    ncol = df['cats'].label_encoding(cats=vals, dtype='float32')
    arr = ncol.to_array()

    # verify labels of new column
    for i in range(arr.size):
        np.testing.assert_equal(arr[i], lab.get(arr[i], None))

    # label encode data frame
    df2 = df.label_encoding(column='cats', prefix='cats', cats=vals, dtype='float32')

    assert df2.columns[0] == 'cats'
    assert df2.columns[1] == 'cats_labels'


if __name__ == '__main__':
    test_label_encode()
