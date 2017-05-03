import pytest

import numpy as np

from numba import cuda

from pygdf.dataframe import Buffer, BufferSentryError, Series, DataFrame


def test_buffer_basic():
    n = 10
    buf = Buffer(np.arange(n, dtype=np.float64))
    assert buf.size == n
    assert buf.capacity == n
    np.testing.assert_equal(buf.mem.copy_to_host(),
                            np.arange(n, dtype=np.float64))


def test_buffer_append():
    n = 10
    expected = np.arange(n, dtype=np.float64)
    buf = Buffer(expected, size=n - 4, capacity=n)
    assert buf.size == n - 4
    assert buf.capacity == n
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    np.testing.assert_equal(buf.to_array(), np.arange(n - 4, dtype=np.float64))

    # Buffer.append
    buf.append(1.23)
    expected[n - 4] = 1.23
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    assert buf.size == n - 3
    assert buf.capacity == n

    # Buffer.extend
    buf.extend(np.asarray([2, 3]))
    expected[n - 3] = 2
    expected[n - 2] = 3
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    assert buf.size == n - 1
    assert buf.capacity == n

    # Test out-of-bound
    with pytest.raises(MemoryError):
        buf.extend(np.asarray([2, 3]))
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    assert buf.size == n - 1
    assert buf.capacity == n

    # Append to last slot
    buf.append(10.125)
    expected[n - 1] = 10.125
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    assert buf.size == n
    assert buf.capacity == n

    with pytest.raises(MemoryError):
        buf.append(987654)

    np.testing.assert_equal(buf.to_array(), expected)
    assert buf.size == n
    assert buf.capacity == n


def test_series_basic():
    # Make series from buffer
    a1 = np.arange(10, dtype=np.float64)
    series = Series.from_any(a1)
    assert len(series) == 10
    np.testing.assert_equal(series.to_array(), np.hstack([a1]))

    # Add new buffer
    a2 = np.arange(5)
    series = series.append(a2)
    assert len(series) == 15
    np.testing.assert_equal(series.to_array(), np.hstack([a1, a2]))

    # Ensure appending to previous buffer
    a3 = np.arange(3)
    series = series.append(a3)
    assert len(series) == 18
    a4 = np.hstack([a1, a2, a3])
    np.testing.assert_equal(series.to_array(), a4)


def test_series_indexing():
    a1 = np.arange(20)
    series = Series.from_any(a1)
    # Indexing
    sr1 = series[:12]
    assert not sr1.has_null_mask
    np.testing.assert_equal(sr1.to_array(), a1[:12])
    sr2 = sr1[3:]
    assert not sr2.has_null_mask
    np.testing.assert_equal(sr2.to_array(), a1[3:12])
    # Index with stride
    sr3 = sr2[::2]
    assert not sr3.has_null_mask
    np.testing.assert_equal(sr3.to_array(), a1[3:12:2])


def test_dataframe_basic():
    np.random.seed(0)
    df = DataFrame()

    # Populate with cuda memory
    df['keys'] = cuda.to_device(np.arange(10, dtype=np.float64))
    np.testing.assert_equal(df['keys'].to_array(), np.arange(10))
    assert len(df) == 10

    # Populate with numpy array
    rnd_vals = np.random.random(10)
    df['vals'] = rnd_vals
    np.testing.assert_equal(df['vals'].to_array(), rnd_vals)
    assert len(df) == 10
    assert df.columns == ('keys', 'vals')

    # Make another dataframe
    df2 = DataFrame()
    df2['keys'] = np.array([123], dtype=np.float64)
    df2['vals'] = np.array([321], dtype=np.float64)

    # Concat
    df = df.concat(df2)
    assert len(df) == 11

    hkeys = np.asarray(np.arange(10, dtype=np.float64).tolist() + [123])
    hvals = np.asarray(rnd_vals.tolist() + [321])

    np.testing.assert_equal(df['keys'].to_array(), hkeys)
    np.testing.assert_equal(df['vals'].to_array(), hvals)

    # As matrix
    mat = df.as_matrix()

    expect = np.vstack([hkeys, hvals]).T

    print(expect)
    print(mat)
    np.testing.assert_equal(mat, expect)


def test_dataframe_column_add_drop():
    df = DataFrame()
    data = np.asarray(range(10))
    df['a'] = data
    df['b'] = data
    assert df.columns == ('a', 'b')
    del df['a']
    assert df.columns == ('b',)
    df['c'] = data
    assert df.columns == ('b', 'c')
    df['a'] = data
    assert df.columns == ('b', 'c', 'a')


def test_dataframe_astype():
    df = DataFrame()
    data = np.asarray(range(10), dtype=np.int32)
    df['a'] = data
    assert df['a'].dtype == np.dtype(np.int32)
    df['b'] = df['a'].astype(np.float32)
    assert df['b'].dtype == np.dtype(np.float32)
    np.testing.assert_equal(df['a'].to_array(), df['b'].to_array())


def test_dataframe_loc():
    df = DataFrame()
    size = 123
    df['a'] = ha = np.random.randint(low=0, high=100, size=size).astype(np.int32)
    df['b'] = hb = np.random.random(size).astype(np.float32)
    df['c'] = hc = np.random.randint(low=0, high=100, size=size).astype(np.int64)
    df['d'] = hd = np.random.random(size).astype(np.float64)

    # Row slice first 10
    first_10 = df.loc[:10]
    assert len(first_10) == 10
    assert first_10.columns == tuple(['a', 'b', 'c', 'd'])
    np.testing.assert_equal(first_10['a'].to_array(), ha[:10])
    np.testing.assert_equal(first_10['b'].to_array(), hb[:10])
    np.testing.assert_equal(first_10['c'].to_array(), hc[:10])
    np.testing.assert_equal(first_10['d'].to_array(), hd[:10])
    del first_10

    # Row slice first 10
    last_10 = df.loc[-10:]
    assert len(last_10) == 10
    assert last_10.columns == tuple(['a', 'b', 'c', 'd'])
    np.testing.assert_equal(last_10['a'].to_array(), ha[-10:])
    np.testing.assert_equal(last_10['b'].to_array(), hb[-10:])
    np.testing.assert_equal(last_10['c'].to_array(), hc[-10:])
    np.testing.assert_equal(last_10['d'].to_array(), hd[-10:])
    del last_10

    # Row slice [begin:end]
    begin = 7
    end = 121
    subrange = df.loc[begin:end]
    assert len(subrange) == end - begin
    assert subrange.columns == tuple(['a', 'b', 'c', 'd'])
    np.testing.assert_equal(subrange['a'].to_array(), ha[begin:end])
    np.testing.assert_equal(subrange['b'].to_array(), hb[begin:end])
    np.testing.assert_equal(subrange['c'].to_array(), hc[begin:end])
    np.testing.assert_equal(subrange['d'].to_array(), hd[begin:end])
    del subrange

    # Row + Col slice
    begin = 117
    end = 123
    fewer = df.loc[begin:end, ['c', 'd', 'a']]
    assert len(fewer) == end - begin
    assert fewer.columns == tuple(['c', 'd', 'a'])
    np.testing.assert_equal(fewer['a'].to_array(), ha[begin:end])
    np.testing.assert_equal(fewer['c'].to_array(), hc[begin:end])
    np.testing.assert_equal(fewer['d'].to_array(), hd[begin:end])
    del fewer


def test_dataframe_to_string():
    # Test basic
    df = DataFrame([('a', [1, 2, 3, 4, 5, 6]),
                    ('b', [11, 12, 13, 14, 15, 16])])
    string = str(df)
    print(string)
    assert string.splitlines()[-1] == '[1 more rows]'

    # Test skipped columns
    df = DataFrame([('a', [1,  2,  3,  4,  5,  6]),
                    ('b', [11, 12, 13, 14, 15, 16]),
                    ('c', [11, 12, 13, 14, 15, 16]),
                    ('d', [11, 12, 13, 14, 15, 16])])
    string = df.to_string(ncols=3)
    print(string)
    assert string.splitlines()[-2] == '[1 more rows]'
    assert string.splitlines()[-1] == '[1 more columns]'

    # Test masked
    df = DataFrame([('a', [1, 2, 3, 4, 5, 6]),
                    ('b', [11, 12, 13, 14, 15, 16])])

    data = np.arange(6)
    mask = np.zeros(1, dtype=np.uint8)
    mask[0] = 0b00101101

    masked = Series.from_masked_array(data, mask)
    assert masked.null_count == 2
    df['c'] = masked

    # check data
    values = list(masked)
    validids = [0, 2, 3, 5]
    densearray = masked.to_array()
    np.testing.assert_equal(data[validids], densearray)
    # valid position is corret
    for i in validids:
        assert data[i] == values[i]
    # null position is correct
    for i in range(len(values)):
        if i not in validids:
            assert values[i] is None

    got = df.to_string(nrows=None)
    print(got)
    expect = '''
  a b  c
0 1 11 0
1 2 12
2 3 13 2
3 4 14 3
4 5 15
5 6 16 5
'''
    # values should match despite whitespace difference
    assert all(x == y for x, y in zip(got.split(), expect.split()))


if __name__ == '__main__':
    test_dataframe_to_string()
