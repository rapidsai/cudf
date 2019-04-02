# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
try:
    from distributed.protocol import serialize, deserialize
    _have_distributed = True
except ImportError:
    _have_distributed = False
import pytest
import cudf
from cudf.tests import utils


require_distributed = pytest.mark.skipif(not _have_distributed,
                                         reason='no distributed')


@require_distributed
def test_serialize_dataframe():
    df = cudf.DataFrame()
    df['a'] = np.arange(100)
    df['b'] = np.arange(100, dtype=np.float32)
    df['c'] = pd.Categorical(['a', 'b', 'c', '_', '_'] * 20,
                             categories=['a', 'b', 'c'])
    outdf = deserialize(*serialize(df))
    pd.util.testing.assert_frame_equal(df.to_pandas(), outdf.to_pandas())


@require_distributed
def test_serialize_dataframe_with_index():
    df = cudf.DataFrame()
    df['a'] = np.arange(100)
    df['b'] = np.random.random(100)
    df['c'] = pd.Categorical(['a', 'b', 'c', '_', '_'] * 20,
                             categories=['a', 'b', 'c'])
    df = df.sort_values('b')
    outdf = deserialize(*serialize(df))
    pd.util.testing.assert_frame_equal(df.to_pandas(), outdf.to_pandas())


@require_distributed
def test_serialize_series():
    sr = cudf.Series(np.arange(100))
    outsr = deserialize(*serialize(sr))
    pd.util.testing.assert_series_equal(sr.to_pandas(), outsr.to_pandas())


@require_distributed
def test_serialize_range_index():
    index = cudf.dataframe.index.RangeIndex(10, 20)
    outindex = deserialize(*serialize(index))
    assert index == outindex


@require_distributed
def test_serialize_generic_index():
    index = cudf.dataframe.index.GenericIndex(cudf.Series(np.arange(10)))
    outindex = deserialize(*serialize(index))
    assert index == outindex


@require_distributed
def test_serialize_masked_series():
    nelem = 50
    data = np.random.random(nelem)
    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]
    null_count = utils.count_zero(bitmask)
    assert null_count >= 0
    sr = cudf.Series.from_masked_array(data, mask, null_count=null_count)
    outsr = deserialize(*serialize(sr))
    pd.util.testing.assert_series_equal(sr.to_pandas(), outsr.to_pandas())


@require_distributed
def test_serialize_groupby():
    df = cudf.DataFrame()
    df['key'] = np.random.randint(0, 20, 100)
    df['val'] = np.arange(100, dtype=np.float32)
    gb = df.groupby('key')
    outgb = deserialize(*serialize(gb))

    got = gb.mean()
    expect = outgb.mean()
    pd.util.testing.assert_frame_equal(got.to_pandas(), expect.to_pandas())


@require_distributed
def test_serialize_datetime():
    # Make frame with datetime column
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=20),
                       'y': np.random.normal(size=20)})
    ts = np.arange(0, len(df), dtype=np.dtype('datetime64[ms]'))
    df['timestamp'] = ts
    gdf = cudf.DataFrame.from_pandas(df)
    # (De)serialize roundtrip
    recreated = deserialize(*serialize(gdf))
    # Check
    pd.util.testing.assert_frame_equal(recreated.to_pandas(), df)


@require_distributed
def test_serialize_string():
    # Make frame with string column
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=5),
                       'y': np.random.normal(size=5)})
    str_data = ['a', 'bc', 'def', 'ghij', 'klmno']
    df['timestamp'] = str_data
    gdf = cudf.DataFrame.from_pandas(df)
    # (De)serialize roundtrip
    recreated = deserialize(*serialize(gdf))
    # Check
    pd.util.testing.assert_frame_equal(recreated.to_pandas(), df)


@require_distributed
def test_serialize_empty_string():
    pd_series = pd.Series([], dtype='str')
    gd_series = cudf.Series([], dtype='str')

    recreated = deserialize(*serialize(gd_series))
    pd.util.testing.assert_series_equal(recreated.to_pandas(), pd_series)


@require_distributed
def test_serialize_all_null_string():
    data = [None, None, None, None, None]
    pd_series = pd.Series(data, dtype='str')
    gd_series = cudf.Series(data, dtype='str')

    recreated = deserialize(*serialize(gd_series))
    pd.util.testing.assert_series_equal(recreated.to_pandas(), pd_series)
