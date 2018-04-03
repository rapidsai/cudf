import pytest

import numpy as np
import pandas as pd

from numba import cuda

import pygdf as gd
from pygdf.dataframe import Series, DataFrame
from pygdf.buffer import Buffer
from pygdf.settings import set_options

from . import utils


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
    series = Series(a1)
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
    series = Series(a1)
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
    df = gd.concat([df, df2])
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
    assert df['a'].dtype is np.dtype(np.int32)
    df['b'] = df['a'].astype(np.float32)
    assert df['b'].dtype is np.dtype(np.float32)
    np.testing.assert_equal(df['a'].to_array(), df['b'].to_array())


def test_dataframe_slicing():
    df = DataFrame()
    size = 123
    df['a'] = ha = np.random.randint(low=0, high=100, size=size).astype(np.int32)
    df['b'] = hb = np.random.random(size).astype(np.float32)
    df['c'] = hc = np.random.randint(low=0, high=100, size=size).astype(np.int64)
    df['d'] = hd = np.random.random(size).astype(np.float64)

    # Row slice first 10
    first_10 = df[:10]
    assert len(first_10) == 10
    assert first_10.columns == tuple(['a', 'b', 'c', 'd'])
    np.testing.assert_equal(first_10['a'].to_array(), ha[:10])
    np.testing.assert_equal(first_10['b'].to_array(), hb[:10])
    np.testing.assert_equal(first_10['c'].to_array(), hc[:10])
    np.testing.assert_equal(first_10['d'].to_array(), hd[:10])
    del first_10

    # Row slice last 10
    last_10 = df[-10:]
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
    subrange = df[begin:end]
    assert len(subrange) == end - begin
    assert subrange.columns == tuple(['a', 'b', 'c', 'd'])
    np.testing.assert_equal(subrange['a'].to_array(), ha[begin:end])
    np.testing.assert_equal(subrange['b'].to_array(), hb[begin:end])
    np.testing.assert_equal(subrange['c'].to_array(), hc[begin:end])
    np.testing.assert_equal(subrange['d'].to_array(), hd[begin:end])
    del subrange


def test_dataframe_loc():
    df = DataFrame()
    size = 123
    df['a'] = ha = np.random.randint(low=0, high=100, size=size).astype(np.int32)
    df['b'] = hb = np.random.random(size).astype(np.float32)
    df['c'] = hc = np.random.randint(low=0, high=100, size=size).astype(np.int64)
    df['d'] = hd = np.random.random(size).astype(np.float64)

    # Full slice
    full = df.loc[:, ['c']]
    assert full.columns == tuple(['c'])
    np.testing.assert_equal(full['c'].to_array(), hc)

    begin = 117
    end = 122
    fewer = df.loc[begin:end, ['c', 'd', 'a']]
    assert len(fewer) == end - begin + 1
    assert fewer.columns == tuple(['c', 'd', 'a'])
    np.testing.assert_equal(fewer['a'].to_array(), ha[begin:end + 1])
    np.testing.assert_equal(fewer['c'].to_array(), hc[begin:end + 1])
    np.testing.assert_equal(fewer['d'].to_array(), hd[begin:end + 1])
    del fewer

    # Make int64 index
    offset = 50
    df2 = df[offset:]
    begin = 117
    end = 122
    fewer = df2.loc[begin:end, ['c', 'd', 'a']]
    assert len(fewer) == end - begin + 1
    assert fewer.columns == tuple(['c', 'd', 'a'])
    np.testing.assert_equal(fewer['a'].to_array(), ha[begin:end + 1])
    np.testing.assert_equal(fewer['c'].to_array(), hc[begin:end + 1])
    np.testing.assert_equal(fewer['d'].to_array(), hd[begin:end + 1])


def test_dataframe_to_string():
    with set_options(formatting={'nrows': 5, 'ncols': 8}):
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
        assert got.split() == expect.split()


def test_dataframe_to_string_wide():
    # Test basic
    df = DataFrame()
    for i in range(100):
        df['a{}'.format(i)] = list(range(3))
    got = df.to_string(ncols=8)
    print(got)
    expect = '''
    a0   a1   a2   a3   a4   a5   a6 ...  a99
0    0    0    0    0    0    0    0 ...    0
1    1    1    1    1    1    1    1 ...    1
2    2    2    2    2    2    2    2 ...    2
[92 more columns]
'''
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_dtypes():
    dtypes = pd.Series([np.int32, np.float32, np.float64],
                       index=['c', 'a', 'b'])
    df = DataFrame([(k, np.ones(10, dtype=v))
                    for k, v in dtypes.iteritems()])
    assert df.dtypes.equals(dtypes)


def test_dataframe_dir_and_getattr():
    df = DataFrame([('a', np.ones(10)),
                    ('b', np.ones(10)),
                    ('not an id', np.ones(10)),
                    ('oop$', np.ones(10))])
    o = dir(df)
    assert {'a', 'b'}.issubset(o)
    assert 'not an id' not in o
    assert 'oop$' not in o

    # Getattr works
    assert df.a is df['a']
    assert df.b is df['b']
    with pytest.raises(AttributeError):
        df.not_a_column


def test_dataframe_as_gpu_matrix():
    df = DataFrame()

    nelem = 123
    for k in 'abcd':
        df[k] = np.random.random(nelem)

    # Check all columns
    mat = df.as_gpu_matrix().copy_to_host()
    assert mat.shape == (nelem, 4)
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(df[k].to_array(), mat[:, i])

    # Check column subset
    mat = df.as_gpu_matrix(columns=['a', 'c']).copy_to_host()
    assert mat.shape == (nelem, 2)

    for i, k in enumerate('ac'):
        np.testing.assert_array_equal(df[k].to_array(), mat[:, i])


def test_dataframe_as_gpu_matrix_null_values():
    df = DataFrame()

    nelem = 123
    na = -10000

    refvalues = {}
    for k in 'abcd':
        df[k] = data = np.random.random(nelem)
        bitmask = utils.random_bitmask(nelem)
        df[k] = df[k].set_mask(bitmask)
        boolmask = np.asarray(utils.expand_bits_to_bytes(bitmask)[:nelem],
                              dtype=np.bool_)
        data[~boolmask] = na
        refvalues[k] = data

    # Check null value causes error
    with pytest.raises(ValueError) as raises:
        df.as_gpu_matrix()
    raises.match("column 'a' has null values")

    for k in df.columns:
        df[k] = df[k].fillna(na)

    mat = df.as_gpu_matrix().copy_to_host()
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(refvalues[k], mat[:, i])


@pytest.mark.parametrize('ntake', [0, 1, 10, 123, 122, 200])
def test_dataframe_take(ntake):
    np.random.seed(0)
    df = DataFrame()

    nelem = 123
    df['ii'] = ii = np.random.randint(0, 20, nelem)
    df['ff'] = ff = np.random.random(nelem)

    take_indices = np.random.randint(0, len(df), ntake)

    def check(**kwargs):
        out = df.take(take_indices, **kwargs)
        assert len(out) == ntake
        np.testing.assert_array_equal(out.ii.to_array(), ii[take_indices])
        np.testing.assert_array_equal(out.ff.to_array(), ff[take_indices])
        if kwargs.get('ignore_index'):
            np.testing.assert_array_equal(out.index, np.arange(ntake))
        else:
            np.testing.assert_array_equal(out.index, take_indices)

    check()
    check(ignore_index=True)

def test_dataframe_append_empty():
    # Test issue 110
    pdf = pd.DataFrame({
        "key": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        })
    gdf = DataFrame.from_pandas(pdf)

    gdf['newcol'] = 100
    pdf['newcol'] = 100

    assert len(gdf['newcol']) == len(pdf)
    assert len(pdf['newcol']) == len(pdf)
    pd.testing.assert_frame_equal(gdf.to_pandas(), pdf)


def test_dataframe_append_to_empty():
    # Test issue 125
    pdf = pd.DataFrame()
    pdf['a'] = []
    pdf['b'] = [1, 2, 3]

    gdf = DataFrame()
    gdf['a'] = []
    gdf['b'] = [1, 2, 3]

    pd.testing.assert_frame_equal(gdf.to_pandas(), pdf)
