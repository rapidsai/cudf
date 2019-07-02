import pytest

import numpy as np

from cudf.dataframe import DataFrame


def test_to_records_noindex():
    df = DataFrame()
    df['a'] = aa = np.arange(10, dtype=np.int32)
    df['b'] = bb = np.arange(10, 20, dtype=np.float64)

    rec = df.to_records(index=False)
    assert rec.dtype.names == ('a', 'b')
    np.testing.assert_array_equal(rec['a'], aa)
    np.testing.assert_array_equal(rec['b'], bb)


def test_to_records_withindex():
    df = DataFrame()
    df['a'] = aa = np.arange(10, dtype=np.int32)
    df['b'] = bb = np.arange(10, 20, dtype=np.float64)

    rec_indexed = df.to_records(index=True)
    assert rec_indexed.size == len(aa)
    assert rec_indexed.dtype.names == ('index', 'a', 'b')
    np.testing.assert_array_equal(rec_indexed['a'], aa)
    np.testing.assert_array_equal(rec_indexed['b'], bb)
    np.testing.assert_array_equal(rec_indexed['index'], np.arange(10))


@pytest.mark.parametrize('columns', [None, ('a', 'b'), ('a',), ('b',)])
def test_from_records_noindex(columns):
    recdtype = np.dtype([
        ('a', np.int32),
        ('b', np.float64),
    ])
    rec = np.recarray(10, dtype=recdtype)
    rec.a = aa = np.arange(10, dtype=np.int32)
    rec.b = bb = np.arange(10, 20, dtype=np.float64)
    df = DataFrame.from_records(rec, columns=columns)

    if columns and 'a' in columns:
        np.testing.assert_array_equal(aa, df['a'])
    if columns and 'b' in columns:
        np.testing.assert_array_equal(bb, df['b'])
    np.testing.assert_array_equal(np.arange(10), df.index.values)


@pytest.mark.parametrize('columns', [None, ('a', 'b'), ('a',), ('b',)])
def test_from_records_withindex(columns):
    recdtype = np.dtype([
        ('index', np.int64),
        ('a', np.int32),
        ('b', np.float64),
    ])
    rec = np.recarray(10, dtype=recdtype)
    rec.index = ii = np.arange(30, 40)
    rec.a = aa = np.arange(10, dtype=np.int32)
    rec.b = bb = np.arange(10, 20, dtype=np.float64)
    df = DataFrame.from_records(rec, index='index')

    if columns and 'a' in columns:
        np.testing.assert_array_equal(aa, df['a'])
    if columns and 'b' in columns:
        np.testing.assert_array_equal(bb, df['b'])
    np.testing.assert_array_equal(ii, df.index.values)


def test_numpy_non_contiguious():
    recdtype = np.dtype([
        ('index', np.int64),
        ('a', np.int32),
    ])
    rec = np.recarray(10, dtype=recdtype)
    rec.index = np.arange(30, 40)
    rec.a = aa = np.arange(20, dtype=np.int32)[::2]
    assert rec.a.flags['C_CONTIGUOUS'] is False

    gdf = DataFrame.from_records(rec, index='index')
    np.testing.assert_array_equal(aa, gdf['a'].to_array())
