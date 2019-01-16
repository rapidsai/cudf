# Copyright (c) 2018, NVIDIA CORPORATION.

import operator
import pytest

import numpy as np
import pandas as pd
import pyarrow as pa
import array as arr

from librmm_cffi import librmm as rmm

import cudf as gd
from cudf.dataframe.dataframe import Series, DataFrame
from cudf.dataframe.buffer import Buffer
from cudf.settings import set_options

from itertools import combinations

from . import utils
from .utils import assert_eq


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


def test_series_append():
    a1 = np.arange(10, dtype=np.float64)
    series = Series(a1)
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

    # Appending different dtype
    a5 = np.array([1, 2, 3], dtype=np.int32)
    a6 = np.array([4.5, 5.5, 6.5], dtype=np.float64)
    series = Series(a5).append(a6)
    np.testing.assert_equal(series.to_array(), np.hstack([a5, a6]))
    series = Series(a6).append(a5)
    np.testing.assert_equal(series.to_array(), np.hstack([a6, a5]))


index_dtypes = [np.int64, np.int32, np.int16, np.int8,
                np.uint64, np.uint32, np.uint16, np.uint8]


@pytest.mark.parametrize(
    'i1, i2, i3',
    ([(slice(None, 12), slice(3, None), slice(None, None, 2)),
      (range(12), range(3, 12), range(0, 9, 2)),
      (np.arange(12), np.arange(3, 12), np.arange(0, 9, 2)),
      (list(range(12)), list(range(3, 12)), list(range(0, 9, 2))),
      (pd.Series(range(12)), pd.Series(range(3, 12)),
       pd.Series(range(0, 9, 2))),
      (Series(range(12)), Series(range(3, 12)), Series(range(0, 9, 2))),
      ([i in range(12) for i in range(20)],
       [i in range(3, 12) for i in range(12)],
       [i in range(0, 9, 2) for i in range(9)]),
      (np.array([i in range(12) for i in range(20)], dtype=bool),
       np.array([i in range(3, 12) for i in range(12)], dtype=bool),
       np.array([i in range(0, 9, 2) for i in range(9)], dtype=bool))]
     + [(np.arange(12, dtype=t), np.arange(3, 12, dtype=t),
         np.arange(0, 9, 2, dtype=t)) for t in index_dtypes]),
    ids=(['slice', 'range', 'numpy.array', 'list', 'pandas.Series',
          'Series', 'list[bool]', 'numpy.array[bool]']
         + ['numpy.array[%s]' % t.__name__ for t in index_dtypes]))
def test_series_indexing(i1, i2, i3):
    a1 = np.arange(20)
    series = Series(a1)
    # Indexing
    sr1 = series[i1]
    assert sr1.null_count == 0
    np.testing.assert_equal(sr1.to_array(), a1[:12])
    sr2 = sr1[i2]
    assert sr2.null_count == 0
    np.testing.assert_equal(sr2.to_array(), a1[3:12])
    # Index with stride
    sr3 = sr2[i3]
    assert sr3.null_count == 0
    np.testing.assert_equal(sr3.to_array(), a1[3:12:2])

    # Integer indexing
    if isinstance(i1, range):
        for i in i1:  # Python int-s
            assert series[i] == a1[i]
    if isinstance(i1, np.ndarray) and i1.dtype in index_dtypes:
        for i in i1:  # numpy integers
            assert series[i] == a1[i]


def test_dataframe_basic():
    np.random.seed(0)
    df = DataFrame()

    # Populate with cuda memory
    df['keys'] = rmm.to_device(np.arange(10, dtype=np.float64))
    np.testing.assert_equal(df['keys'].to_array(), np.arange(10))
    assert len(df) == 10

    # Populate with numpy array
    rnd_vals = np.random.random(10)
    df['vals'] = rnd_vals
    np.testing.assert_equal(df['vals'].to_array(), rnd_vals)
    assert len(df) == 10
    assert tuple(df.columns) == ('keys', 'vals')

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


def test_dataframe_column_name_indexing():
    df = DataFrame()
    data = np.asarray(range(10), dtype=np.int32)
    df['a'] = data
    df[1] = data
    np.testing.assert_equal(df['a'].to_array(),
                            np.asarray(range(10), dtype=np.int32))
    np.testing.assert_equal(df[1].to_array(),
                            np.asarray(range(10), dtype=np.int32))

    pdf = pd.DataFrame()
    nelem = 10
    pdf['key1'] = np.random.randint(0, 5, nelem)
    pdf['key2'] = np.random.randint(0, 3, nelem)
    pdf[1] = np.arange(1, 1 + nelem)
    pdf[2] = np.random.random(nelem)
    df = DataFrame.from_pandas(pdf)
    for i in range(1, len(pdf.columns)+1):
        for idx in combinations(pdf.columns, i):
            assert(pdf[list(idx)].equals(df[list(idx)].to_pandas()))

    # test for only numeric columns
    df = pd.DataFrame()
    for i in range(0, 10):
        df[i] = range(nelem)
    gdf = DataFrame.from_pandas(df)
    assert_eq(gdf, df)


def test_dataframe_drop_method():
    df = DataFrame()
    data = np.asarray(range(10))
    df['a'] = data
    df['b'] = data
    df['c'] = data

    assert tuple(df.columns) == ('a', 'b', 'c')
    assert tuple(df.drop('a').columns) == ('b', 'c')
    assert tuple(df.columns) == ('a', 'b', 'c')
    assert tuple(df.drop(['a', 'b']).columns) == ('c',)
    assert tuple(df.columns) == ('a', 'b', 'c')

    # Test drop error
    with pytest.raises(NameError) as raises:
        df.drop('d')
    raises.match("column 'd' does not exist")
    with pytest.raises(NameError) as raises:
        df.drop(['a', 'd', 'b'])
    raises.match("column 'd' does not exist")


def test_dataframe_column_add_drop():
    df = DataFrame()
    data = np.asarray(range(10))
    df['a'] = data
    df['b'] = data
    assert tuple(df.columns) == ('a', 'b')
    del df['a']
    assert tuple(df.columns) == ('b',)
    df['c'] = data
    assert tuple(df.columns) == ('b', 'c')
    df['a'] = data
    assert tuple(df.columns) == ('b', 'c', 'a')


@pytest.mark.parametrize('nelem', [0, 3, 100, 1000])
def test_dataframe_astype(nelem):
    df = DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df['a'] = data
    assert df['a'].dtype is np.dtype(np.int32)
    df['b'] = df['a'].astype(np.float32)
    assert df['b'].dtype is np.dtype(np.float32)
    np.testing.assert_equal(df['a'].to_array(), df['b'].to_array())


def test_dataframe_slicing():
    df = DataFrame()
    size = 123
    df['a'] = ha = np.random.randint(low=0, high=100, size=size)\
        .astype(np.int32)
    df['b'] = hb = np.random.random(size).astype(np.float32)
    df['c'] = hc = np.random.randint(low=0, high=100, size=size)\
        .astype(np.int64)
    df['d'] = hd = np.random.random(size).astype(np.float64)

    # Row slice first 10
    first_10 = df[:10]
    assert len(first_10) == 10
    assert tuple(first_10.columns) == ('a', 'b', 'c', 'd')
    np.testing.assert_equal(first_10['a'].to_array(), ha[:10])
    np.testing.assert_equal(first_10['b'].to_array(), hb[:10])
    np.testing.assert_equal(first_10['c'].to_array(), hc[:10])
    np.testing.assert_equal(first_10['d'].to_array(), hd[:10])
    del first_10

    # Row slice last 10
    last_10 = df[-10:]
    assert len(last_10) == 10
    assert tuple(last_10.columns) == ('a', 'b', 'c', 'd')
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
    assert tuple(subrange.columns) == ('a', 'b', 'c', 'd')
    np.testing.assert_equal(subrange['a'].to_array(), ha[begin:end])
    np.testing.assert_equal(subrange['b'].to_array(), hb[begin:end])
    np.testing.assert_equal(subrange['c'].to_array(), hc[begin:end])
    np.testing.assert_equal(subrange['d'].to_array(), hd[begin:end])
    del subrange


def test_dataframe_loc():
    df = DataFrame()
    size = 123
    df['a'] = ha = np.random.randint(low=0, high=100, size=size)\
        .astype(np.int32)
    df['b'] = hb = np.random.random(size).astype(np.float32)  # noqa: F841
    df['c'] = hc = np.random.randint(low=0, high=100, size=size)\
        .astype(np.int64)
    df['d'] = hd = np.random.random(size).astype(np.float64)

    # Full slice
    full = df.loc[:, ['c']]
    assert tuple(full.columns) == ('c',)
    np.testing.assert_equal(full['c'].to_array(), hc)

    begin = 117
    end = 122
    fewer = df.loc[begin:end, ['c', 'd', 'a']]
    assert len(fewer) == end - begin + 1
    assert tuple(fewer.columns) == ('c', 'd', 'a')
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
    assert tuple(fewer.columns) == ('c', 'd', 'a')
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


def test_dataframe_empty_to_string():
    # Test for printing empty dataframe
    df = DataFrame()
    got = df.to_string()
    print(got)
    expect = "Empty DataFrame\nColumns: []\nIndex: []\n"
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_emptycolumns_to_string():
    # Test for printing dataframe having empty columns
    df = DataFrame()
    df['a'] = []
    df['b'] = []
    got = df.to_string()
    print(got)
    expect = "Empty DataFrame\nColumns: ['a', 'b']\nIndex: []\n"
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_copy():
    # Test for copying the dataframe using python copy pkg
    from copy import copy
    df = DataFrame()
    df['a'] = [1, 2, 3]
    df2 = copy(df)
    df2['b'] = [4, 5, 6]
    got = df.to_string()
    print(got)
    expect = '''
     a
0    1
1    2
2    3
'''
    # values should match despite whitespace difference
    assert got.split() == expect.split()


def test_dataframe_copy_shallow():
    # Test for copy dataframe using class method
    df = DataFrame()
    df['a'] = [1, 2, 3]
    df2 = df.copy()
    df2['b'] = [4, 2, 3]
    got = df.to_string()
    print(got)
    expect = '''
     a
0    1
1    2
2    3
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


@pytest.mark.parametrize('order', ['C', 'F'])
def test_dataframe_as_gpu_matrix(order):
    df = DataFrame()

    nelem = 123
    for k in 'abcd':
        df[k] = np.random.random(nelem)

    # Check all columns
    mat = df.as_gpu_matrix(order=order).copy_to_host()
    assert mat.shape == (nelem, 4)
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(df[k].to_array(), mat[:, i])

    # Check column subset
    mat = df.as_gpu_matrix(order=order, columns=['a', 'c']).copy_to_host()
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


def test_dataframe_setitem_from_masked_object():
    ary = np.random.randn(100)
    mask = np.zeros(100, dtype=bool)
    mask[:20] = True
    np.random.shuffle(mask)
    ary[mask] = np.nan

    test1_null = Series(ary, nan_as_null=True)
    assert(test1_null.has_null_mask)
    assert(test1_null.null_count == 20)
    test1_nan = Series(ary, nan_as_null=False)
    assert(test1_nan.null_count == 0)

    test2_null = DataFrame.from_pandas(pd.DataFrame({'a': ary}),
                                       nan_as_null=True)
    assert(test2_null['a'].has_null_mask)
    assert(test2_null['a'].null_count == 20)
    test2_nan = DataFrame.from_pandas(pd.DataFrame({'a': ary}),
                                      nan_as_null=False)
    assert(test2_nan['a'].null_count == 0)

    gpu_ary = rmm.to_device(ary)
    test3_null = Series(gpu_ary, nan_as_null=True)
    assert(test3_null.has_null_mask)
    assert(test3_null.null_count == 20)
    test3_nan = Series(gpu_ary, nan_as_null=False)
    assert(test3_nan.null_count == 0)

    test4 = DataFrame()
    lst = [1, 2, None, 4, 5, 6, None, 8, 9]
    test4['lst'] = lst
    assert(test4['lst'].has_null_mask)
    assert(test4['lst'].null_count == 2)


def test_dataframe_append_to_empty():
    pdf = pd.DataFrame()
    pdf['a'] = []
    pdf['b'] = [1, 2, 3]

    gdf = DataFrame()
    gdf['a'] = []
    gdf['b'] = [1, 2, 3]

    pd.testing.assert_frame_equal(gdf.to_pandas(), pdf)


def test_dataframe_setitem_index_len1():
    gdf = DataFrame()
    gdf['a'] = [1]
    gdf['b'] = gdf.index.as_column()

    np.testing.assert_equal(gdf.b.to_array(), [0])


def test_assign():
    gdf = DataFrame({'x': [1, 2, 3]})
    gdf2 = gdf.assign(y=gdf.x + 1)
    assert list(gdf.columns) == ['x']
    assert list(gdf2.columns) == ['x', 'y']

    np.testing.assert_equal(gdf2.y.to_array(), [2, 3, 4])


@pytest.mark.parametrize('nrows', [1, 8, 100, 1000])
def test_dataframe_hash_columns(nrows):
    gdf = DataFrame()
    data = np.asarray(range(nrows))
    data[0] = data[-1]  # make first and last the same
    gdf['a'] = data
    gdf['b'] = gdf.a + 100
    out = gdf.hash_columns(['a', 'b'])
    assert isinstance(out, Series)
    assert len(out) == nrows
    assert out.dtype == np.int32

    # Check default
    out_all = gdf.hash_columns()
    np.testing.assert_array_equal(out.to_array(), out_all.to_array())

    # Check single column
    out_one = gdf.hash_columns(['a']).to_array()
    # First matches last
    assert out_one[0] == out_one[-1]
    # Equivalent to the Series.hash_values()
    np.testing.assert_array_equal(
        gdf.a.hash_values().to_array(),
        out_one,
        )


@pytest.mark.parametrize('nrows', [3, 10, 100, 1000])
@pytest.mark.parametrize('nparts', [1, 2, 8, 13])
@pytest.mark.parametrize('nkeys', [1, 2])
def test_dataframe_hash_partition(nrows, nparts, nkeys):
    np.random.seed(123)
    gdf = DataFrame()
    keycols = []
    for i in range(nkeys):
        keyname = 'key{}'.format(i)
        gdf[keyname] = np.random.randint(0, 7 - i, nrows)
        keycols.append(keyname)
    gdf['val1'] = np.random.randint(0, nrows * 2, nrows)

    got = gdf.partition_by_hash(keycols, nparts=nparts)
    # Must return a list
    assert isinstance(got, list)
    # Must have correct number of partitions
    assert len(got) == nparts
    # All partitions must be DataFrame type
    assert all(isinstance(p, DataFrame) for p in got)
    # Check that all partitions have unique keys
    part_unique_keys = set()
    for p in got:
        if len(p):
            # Take rows of the keycolums and build a set of the key-values
            unique_keys = set(map(tuple, p.as_matrix(columns=keycols)))
            # Ensure that none of the key-values have occurred in other groups
            assert not (unique_keys & part_unique_keys)
            part_unique_keys |= unique_keys
    assert len(part_unique_keys)


@pytest.mark.parametrize('nrows', [3, 10, 50])
def test_dataframe_hash_partition_masked_value(nrows):
    gdf = DataFrame()
    gdf['key'] = np.arange(nrows)
    gdf['val'] = np.arange(nrows) + 100
    bitmask = utils.random_bitmask(nrows)
    bytemask = utils.expand_bits_to_bytes(bitmask)
    gdf['val'] = gdf['val'].set_mask(bitmask)
    parted = gdf.partition_by_hash(['key'], nparts=3)
    # Verify that the valid mask is correct
    for p in parted:
        df = p.to_pandas()
        for row in df.itertuples():
            valid = bool(bytemask[row.key])
            expected_value = row.key + 100 if valid else -1
            got_value = row.val
            assert expected_value == got_value


@pytest.mark.parametrize('nrows', [3, 10, 50])
def test_dataframe_hash_partition_masked_keys(nrows):
    gdf = DataFrame()
    gdf['key'] = np.arange(nrows)
    gdf['val'] = np.arange(nrows) + 100
    bitmask = utils.random_bitmask(nrows)
    bytemask = utils.expand_bits_to_bytes(bitmask)
    gdf['key'] = gdf['key'].set_mask(bitmask)
    parted = gdf.partition_by_hash(['key'], nparts=3)
    # Verify that the valid mask is correct
    for p in parted:
        df = p.to_pandas()
        for row in df.itertuples():
            valid = bool(bytemask[row.val - 100])
            # val is key + 100
            expected_value = row.val - 100 if valid else -1
            got_value = row.key
            assert expected_value == got_value


def test_dataframe_empty_concat():
    gdf1 = DataFrame()
    gdf1['a'] = []
    gdf1['b'] = []

    gdf2 = gdf1.copy()

    gdf3 = gd.concat([gdf1, gdf2])
    assert len(gdf3) == 0
    assert len(gdf3.columns) == 2


@pytest.mark.parametrize('nrows', [0, 3, 10, 100, 1000])
def test_nonmatching_index_setitem(nrows):
    np.random.seed(0)

    gdf = DataFrame()
    gdf['a'] = np.random.randint(2147483647, size=nrows)
    gdf['b'] = np.random.randint(2147483647, size=nrows)
    gdf = gdf.set_index('b')

    test_values = np.random.randint(2147483647, size=nrows)
    gdf['c'] = test_values
    assert(len(test_values) == len(gdf['c']))
    assert(gdf['c'].to_pandas().equals(
        Series(test_values).set_index(gdf._index).to_pandas()))


@pytest.mark.parametrize('nelem', [0, 1, 5, 20, 100])
@pytest.mark.parametrize('slice_start', [None, 0, 1, 3, 10])
@pytest.mark.parametrize('slice_end', [None, 0, 1, 30, 50, -1])
def test_dataframe_masked_slicing(nelem, slice_start, slice_end):
    gdf = DataFrame()
    gdf['a'] = list(range(nelem))
    gdf['b'] = list(range(nelem, 2 * nelem))
    gdf['a'] = gdf['a'].set_mask(utils.random_bitmask(nelem))
    gdf['b'] = gdf['b'].set_mask(utils.random_bitmask(nelem))

    def do_slice(x):
        return x[slice_start: slice_end]

    expect = do_slice(gdf.to_pandas())
    got = do_slice(gdf).to_pandas()

    pd.testing.assert_frame_equal(expect, got)


def test_from_pandas():
    df = pd.DataFrame({'x': [1, 2, 3]}, index=[4., 5., 6.])
    gdf = gd.DataFrame.from_pandas(df)
    assert isinstance(gdf, gd.DataFrame)

    pd.testing.assert_frame_equal(df, gdf.to_pandas())

    s = df.x
    gs = gd.Series.from_pandas(s)
    assert isinstance(gs, gd.Series)

    pd.testing.assert_series_equal(s, gs.to_pandas())


@pytest.mark.xfail(reason="constructor does not coerce index inputs")
def test_index_in_dataframe_constructor():
    a = pd.DataFrame({'x': [1, 2, 3]}, index=[4., 5., 6.])
    b = gd.DataFrame({'x': [1, 2, 3]}, index=[4., 5., 6.])

    pd.testing.assert_frame_equal(a, b.to_pandas())
    assert pd.testing.assert_frame_equal(a.loc[4:], b.loc[4:].to_pandas())


@pytest.mark.parametrize('nelem', [0, 2, 3, 100, 1000])
@pytest.mark.parametrize(
    'data_type',
    ['bool', 'int8', 'int16', 'int32', 'int64',
     'float32', 'float64', 'datetime64[ms]']
)
def test_from_arrow(nelem, data_type):
    df = pd.DataFrame(
        {
            'a': np.random.randint(0, 1000, nelem).astype(data_type),
            'b': np.random.randint(0, 1000, nelem).astype(data_type)
        }
    )
    padf = pa.Table.from_pandas(df, preserve_index=False)\
        .replace_schema_metadata(None)
    gdf = gd.DataFrame.from_arrow(padf)
    assert isinstance(gdf, gd.DataFrame)

    pd.testing.assert_frame_equal(df, gdf.to_pandas())

    s = pa.Array.from_pandas(df.a)
    gs = gd.Series.from_arrow(s)
    assert isinstance(gs, gd.Series)

    # For some reason PyArrow to_pandas() converts to numpy array and has
    # better type compatibility
    np.testing.assert_array_equal(s.to_pandas(), gs.to_array())


@pytest.mark.parametrize('nelem', [0, 2, 3, 100, 1000])
@pytest.mark.parametrize(
    'data_type',
    ['bool', 'int8', 'int16', 'int32', 'int64',
     'float32', 'float64', 'datetime64[ms]']
)
def test_to_arrow(nelem, data_type):
    df = pd.DataFrame(
        {
            'a': np.random.randint(0, 1000, nelem).astype(data_type),
            'b': np.random.randint(0, 1000, nelem).astype(data_type)
        }
    )
    gdf = gd.DataFrame.from_pandas(df)

    pa_df = pa.Table.from_pandas(df, preserve_index=False)\
        .replace_schema_metadata(None)
    # Pandas uses ns so need to cast columns to ms
    if data_type == 'datetime64[ms]':
        pa_df = pa_df.add_column(
                    0,
                    pa_df.column(1)
                    .cast(pa.timestamp('ms'))
                    .cast(pa.int64())
                    .cast(pa.date64())
                ).add_column(
                    0,
                    pa_df.column(0)
                    .cast(pa.timestamp('ms'))
                    .cast(pa.int64())
                    .cast(pa.date64())
                ).remove_column(2).remove_column(2)
    pa_gdf = gdf.to_arrow(index=False)

    assert isinstance(pa_gdf, pa.Table)
    assert pa.Table.equals(pa_df, pa_gdf)

    pa_s = pa.Array.from_pandas(df.a)
    # Pandas uses ns so need to cast columns to ms
    if data_type == 'datetime64[ms]':
        pa_s = pa_s.cast(pa.timestamp('ms')).cast(pa.int64()).cast(pa.date64())
    pa_gs = gdf['a'].to_arrow()

    assert isinstance(pa_gs, pa.Array)
    assert pa.Array.equals(pa_s, pa_gs)

    pa_i = pa.Array.from_pandas(df.index)
    pa_gi = gdf.index.to_arrow()

    assert isinstance(pa_gi, pa.Array)
    assert pa.Array.equals(pa_i, pa_gi)


@pytest.mark.parametrize(
    'data_type',
    ['bool', 'int8', 'int16', 'int32', 'int64',
     'float32', 'float64', 'datetime64[ms]']
)
def test_to_from_arrow_nulls(data_type):
    if data_type == 'datetime64[ms]':
        data_type = pa.date64()
    if data_type == 'bool':
        s1 = pa.array([True, None, False, None, True], type=data_type)
    else:
        s1 = pa.array([1, None, 3, None, 5], type=data_type)
    gs1 = gd.Series.from_arrow(s1)
    assert isinstance(gs1, gd.Series)
    np.testing.assert_array_equal(
        np.array(s1.buffers()[0]),
        gs1.nullmask.to_array()
    )
    assert pa.Array.equals(s1, gs1.to_arrow())

    s2 = pa.array([None, None, None, None, None], type=data_type)
    gs2 = gd.Series.from_arrow(s2)
    assert isinstance(gs2, gd.Series)
    np.testing.assert_array_equal(
        np.array(s2.buffers()[0]),
        gs2.nullmask.to_array()
    )
    assert pa.Array.equals(s2, gs2.to_arrow())


def test_to_arrow_categorical():
    df = pd.DataFrame()
    df['a'] = pd.Series(['a', 'b', 'c'], dtype="category")
    gdf = gd.DataFrame.from_pandas(df)

    pa_df = pa.Table.from_pandas(df, preserve_index=False)\
        .replace_schema_metadata(None)
    pa_gdf = gdf.to_arrow(index=False)

    assert isinstance(pa_gdf, pa.Table)
    assert pa.Table.equals(pa_df, pa_gdf)

    pa_s = pa.Array.from_pandas(df.a)
    pa_gs = gdf['a'].to_arrow()

    assert isinstance(pa_gs, pa.Array)
    assert pa.Array.equals(pa_s, pa_gs)


def test_from_arrow_missing_categorical():
    pd_cat = pd.Categorical(['a', 'b', 'c'], categories=['a', 'b'])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = gd.Series(pa_cat)

    assert isinstance(gd_cat, gd.Series)
    pd.testing.assert_series_equal(
        pd.Series(pa_cat.to_pandas()),  # PyArrow returns a pd.Categorical
        gd_cat.to_pandas()
    )


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="PyArrow does not yet support validity masks in creating "
           "DictionaryArray objects"
)
def test_to_arrow_missing_categorical():
    pd_cat = pd.Categorical(['a', 'b', 'c'], categories=['a', 'b'])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = gd.Series(pa_cat)

    assert isinstance(gd_cat, gd.Series)
    assert pa.Array.equals(pa_cat, gd_cat.to_arrow())


@pytest.mark.parametrize(
    'data_type',
    ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'datetime64[ms]']
)
def test_from_scalar_typing(data_type):
    if data_type == 'datetime64[ms]':
        scalar = np.dtype('int64').type(np.random.randint(0, 5))\
            .astype('datetime64[ms]')
    else:
        scalar = np.dtype(data_type).type(np.random.randint(0, 5))

    gdf = gd.DataFrame()
    gdf['a'] = [1, 2, 3, 4, 5]
    gdf['b'] = scalar
    assert(gdf['b'].dtype == np.dtype(data_type))
    assert(len(gdf['b']) == len(gdf['a']))


@pytest.mark.parametrize(
    'data_type',
    ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']
)
def test_from_python_array(data_type):
    np_arr = np.random.randint(0, 100, 10).astype(data_type)
    data = memoryview(np_arr)
    data = arr.array(data.format, data)

    gs = gd.Series(data)

    np.testing.assert_equal(gs.to_array(), np_arr)


def test_series_shape():
    ps = pd.Series([1, 2, 3, 4])
    cs = Series([1, 2, 3, 4])

    assert ps.shape == cs.shape


def test_series_shape_empty():
    ps = pd.Series()
    cs = Series([])

    assert ps.shape == cs.shape


def test_dataframe_shape():
    pdf = pd.DataFrame({'a': [0, 1, 2, 3], 'b': [0.1, 0.2, None, 0.3]})
    gdf = DataFrame.from_pandas(pdf)

    assert pdf.shape == gdf.shape


def test_dataframe_shape_empty():
    pdf = pd.DataFrame()
    gdf = DataFrame()

    assert pdf.shape == gdf.shape


def test_generated_column():
    gdf = DataFrame({'a': (i for i in range(5))})
    assert len(gdf) == 5


@pytest.fixture
def pdf():
    return pd.DataFrame({'x': range(10),
                         'y': range(10)})


@pytest.fixture
def gdf(pdf):
    return gd.DataFrame.from_pandas(pdf)


@pytest.mark.parametrize('func', [
    lambda df: df.mean(),
    lambda df: df.sum(),
    lambda df: df.min(),
    lambda df: df.max(),
    lambda df: df.std(),
    lambda df: df.count(),
    pytest.param(lambda df: df.size, marks=pytest.mark.xfail()),
])
@pytest.mark.parametrize('accessor', [
    pytest.param(lambda df: df, marks=pytest.mark.xfail(
        reason="dataframe reductions not yet supported")),
    lambda df: df.x,
])
def test_reductions(pdf, gdf, accessor, func):
    assert_eq(func(accessor(pdf)), func(accessor(gdf)))


@pytest.mark.parametrize('left', [
    pytest.param(lambda df: df, marks=pytest.mark.xfail()),
    lambda df: df.x,
    lambda df: 3,
])
@pytest.mark.parametrize('right', [
    pytest.param(lambda df: df, marks=pytest.mark.xfail()),
    lambda df: df.x,
    lambda df: 3,
])
@pytest.mark.parametrize('binop', [
    operator.add,
    operator.mul,
    pytest.param(operator.floordiv, marks=pytest.mark.xfail()),
    pytest.param(operator.truediv, marks=pytest.mark.xfail()),
    pytest.param(operator.mod, marks=pytest.mark.xfail()),
    pytest.param(operator.pow, marks=pytest.mark.xfail()),
    operator.eq,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.ne,
])
def test_binops(pdf, gdf, left, right, binop):
    d = binop(left(pdf), right(pdf))
    g = binop(left(gdf), right(gdf))
    assert_eq(d, g)


@pytest.mark.xfail(reason="null is not supported in gpu yet")
def test_dataframe_boolean_mask_with_None():
    pdf = pd.DataFrame({'a': [0, 1, 2, 3], 'b': [0.1, 0.2, None, 0.3]})
    gdf = DataFrame.from_pandas(pdf)
    pdf_masked = pdf[[True, False, True, False]]
    gdf_masked = gdf[[True, False, True, False]]
    assert pdf_masked.to_string().split() == gdf_masked.to_string().split()


"""
This test compares cudf and Pandas dataframe boolean indexing.
"""


@pytest.mark.parametrize('mask_fn', [
    lambda x: x,
    lambda x: np.array(x),
    lambda x: pd.Series(x),
    ])
def test_dataframe_boolean_mask(pdf, gdf, mask_fn):
    mask_base = [True, False, True, False, True, False, True, False, True,
                 False]
    mask = mask_fn(mask_base)
    assert len(mask) == gdf.shape[0]
    pdf_masked = pdf[mask]
    gdf_masked = gdf[mask]
    assert pdf_masked.to_string().split() == gdf_masked.to_string().split()


"""
This test only tests boolean indexing of a cudf DataFrame with a cudf Series.
Pandas does not support cudf Series.  When masking with a Series, the length
is not required to match.
"""


def test_dataframe_boolean_mask_Series(gdf):
    mask = Series([True, False, True, False])
    mask2 = Series([True, True, True, True])
    mask3 = Series([True, True, True, True, True, True, True, True])
    gdf_masked = gdf[mask]
    gdf_masked2 = gdf[mask2]
    gdf_masked3 = gdf[mask3]
    assert gdf_masked.shape[0] == 2
    assert gdf_masked2.shape[0] == 4
    assert gdf_masked3.shape[0] == 8


def test_iter(pdf, gdf):
    assert list(pdf) == list(gdf)


def test_iteritems(gdf):
    for k, v in gdf.iteritems():
        assert k in gdf.columns
        assert isinstance(v, gd.Series)
        assert_eq(v, gdf[k])


@pytest.mark.xfail(reason="our quantile result is a DataFrame, not a Series")
def test_quantile(pdf, gdf):
    assert_eq(pdf.quantile(), gdf.quantile())


def test_from_pandas_function(pdf):
    gdf = gd.from_pandas(pdf)
    assert isinstance(gdf, gd.DataFrame)
    assert_eq(pdf, gdf)

    gdf = gd.from_pandas(pdf.x)
    assert isinstance(gdf, gd.Series)
    assert_eq(pdf.x, gdf)

    with pytest.raises(TypeError):
        gd.from_pandas(123)
