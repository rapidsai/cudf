import pytest
import pygdf
from numba import cuda
from pygdf.dataframe import DataFrame, Series

def test_label_encode():
    df = DataFrame()
    np.random.seed(0)

    # initialize data frame
    df['cats'] = np.arange(10, dtype=np.int32)
    lab = list(range(len(df['cats'])))

    # label encode series
    ncol = df['cats'].label_encoding(list(range(10)))
    arr = ncol.to_array()

    # verify labels of new column
    for i in list(range(10)):
        np.testing.assert_equal(arr[i], lab[i])

    # label encode data frame
    df2 = df.label_encoding(column='cats', prefix='cats', cats=list(range(10)))

    assert df2.columns[0] == 'cats'
    assert df2.columns[1] == 'cats_labels'


if __name__ == '__main__':
    test_label_encode()
