# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np

from cudf.dataframe import DataFrame, Series, GenericIndex
from cudf.tests import utils


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


def test_onehot_masked():
    np.random.seed(0)
    high = 5
    size = 100
    arr = np.random.randint(low=0, high=high, size=size)
    bitmask = utils.random_bitmask(size)
    bytemask = np.asarray(utils.expand_bits_to_bytes(bitmask)[:size],
                          dtype=np.bool_)
    arr[~bytemask] = -1

    df = DataFrame()
    df['a'] = Series(arr).set_mask(bitmask)

    out = df.one_hot_encoding('a', cats=list(range(high)),
                              prefix='a', dtype=np.int32)

    assert tuple(out.columns) == ('a', 'a_0', 'a_1', 'a_2', 'a_3', 'a_4')
    np.testing.assert_array_equal(out['a_0'] == 1, arr == 0)
    np.testing.assert_array_equal(out['a_1'] == 1, arr == 1)
    np.testing.assert_array_equal(out['a_2'] == 1, arr == 2)
    np.testing.assert_array_equal(out['a_3'] == 1, arr == 3)
    np.testing.assert_array_equal(out['a_4'] == 1, arr == 4)


def test_onehot_generic_index():
    np.random.seed(0)
    size = 33
    indices = np.random.randint(low=0, high=100, size=size)
    df = DataFrame()
    values = np.random.randint(low=0, high=4, size=size)
    df['fo'] = Series(values, index=GenericIndex(indices))
    out = df.one_hot_encoding('fo', cats=df.fo.unique(), prefix='fo',
                              dtype=np.int32)
    assert set(out.columns) == {'fo', 'fo_0', 'fo_1', 'fo_2', 'fo_3'}
    np.testing.assert_array_equal(values == 0, out.fo_0.to_array())
    np.testing.assert_array_equal(values == 1, out.fo_1.to_array())
    np.testing.assert_array_equal(values == 2, out.fo_2.to_array())
    np.testing.assert_array_equal(values == 3, out.fo_3.to_array())


if __name__ == '__main__':
    test_onehot_random()
