import numpy as np
import pandas as pd
from distributed.protocol import serialize, deserialize

import pygdf
from . import utils


def test_serialize_dataframe():
    df = pygdf.DataFrame()
    df['a'] = np.arange(100)
    df['b'] = np.arange(100, dtype=np.float32)
    df['c'] = pd.Categorical(['a', 'b', 'c', '_', '_'] * 20,
                             categories=['a', 'b', 'c'])
    outdf = deserialize(*serialize(df))
    pd.util.testing.assert_frame_equal(df.to_pandas(), outdf.to_pandas())


def test_serialize_series():
    sr = pygdf.Series(np.arange(100))
    outsr = deserialize(*serialize(sr))
    pd.util.testing.assert_series_equal(sr.to_pandas(), outsr.to_pandas())


def test_serialize_range_index():
    index = pygdf.index.RangeIndex(10, 20)
    outindex = deserialize(*serialize(index))
    assert index == outindex


def test_serialize_generic_index():
    index = pygdf.index.GenericIndex(pygdf.Series(np.arange(10)))
    outindex = deserialize(*serialize(index))
    assert index == outindex


def test_serialize_masked_series():
    nelem = 50
    data = np.random.random(nelem)
    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]
    null_count = utils.count_zero(bitmask)
    assert null_count >= 0
    sr = pygdf.Series.from_masked_array(data, mask, null_count=null_count)
    outsr = deserialize(*serialize(sr))
    pd.util.testing.assert_series_equal(sr.to_pandas(), outsr.to_pandas())


def test_serialize_groupby():
    df = pygdf.DataFrame()
    df['key'] = np.random.randint(0, 20, 100)
    df['val'] = np.arange(100, dtype=np.float32)
    gb = df.groupby('key')
    outgb = deserialize(*serialize(gb))

    got = gb.mean()
    expect = outgb.mean()
    pd.util.testing.assert_frame_equal(got.to_pandas(), expect.to_pandas())
