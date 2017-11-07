import pytest
import numpy as np
import random
from numba import cuda
import pygdf
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
        np.testing.assert_equal(arr[i], lab.get(df.cats[i], None))

    # label encode data frame
    df2 = df.label_encoding(column='cats', prefix='cats', cats=vals, dtype='float32')

    assert df2.columns[0] == 'cats'
    assert df2.columns[1] == 'cats_labels'


def test_label_encode_drop_one():
    random.seed(0)
    np.random.seed(0)

    df = DataFrame()

    # initialize data frame
    df['cats'] = np.random.randint(7, size=10, dtype=np.int32)
    vals = list(df['cats'].unique_k(10))
    # drop 1 randomly
    del vals[random.randrange(len(vals))]

    lab = dict(zip(vals, list(range(len(vals)))))

    # label encode series
    ncol = df['cats'].label_encoding(cats=vals, dtype='float32')
    arr = ncol.to_array()

    # verify labels of new column

    for i in range(arr.size):
        # assuming -1 is used for missing value
        np.testing.assert_equal(arr[i], lab.get(df.cats[i], -1))

    # label encode data frame
    df2 = df.label_encoding(column='cats', prefix='cats', cats=vals, dtype='float32')

    assert df2.columns[0] == 'cats'
    assert df2.columns[1] == 'cats_labels'


if __name__ == '__main__':
    test_label_encode()
    test_label_encode_drop_one()
