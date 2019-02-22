# Copyright (c) 2018, NVIDIA CORPORATION.

import atexit
import multiprocessing as mp

import numpy as np
import pandas as pd
from numba import cuda
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
support_ipc = False
require_ipc = pytest.mark.skipIf(
    support_ipc,
    reason='only on linux and multiprocess has .get_context',
    )


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


def serialize_ipc(sr):
    # Non-IPC
    header, frames = serialize(sr)
    assert header['column']['data_buffer']['kind'] == 'normal'
    # IPC
    hostport = 'tcp://0.0.0.0:8888'
    fake_context = {
        'recipient': hostport,
        'sender': hostport,
    }

    assert sr._column.data._cached_ipch is None
    header, frames = serialize(sr, context=fake_context)
    assert header['column']['data_buffer']['kind'] == 'ipc'
    # Check that _cached_ipch is set on the buffer
    assert isinstance(sr._column.data._cached_ipch,
                      cuda.cudadrv.devicearray.IpcArrayHandle)

    # Spawn a new process to test the IPC handle deserialization
    mpctx = mp.get_context('spawn')
    result_queue = mpctx.Queue()

    proc = mpctx.Process(target=_load_ipc, args=(header, frames, result_queue))
    proc.start()
    out = result_queue.get()
    proc.join(3)
    return out


@require_distributed
@require_ipc
def test_serialize_ipc():
    sr = cudf.Series(np.arange(10))
    out = serialize_ipc(sr)
    # Verify that the output array matches the source
    np.testing.assert_array_equal(out.to_array(), sr.to_array())


@require_distributed
@require_ipc
def test_serialize_ipc_slice():
    sr = cudf.Series(np.arange(10))
    # test with a slice to verify internal offset calculations work
    out = serialize_ipc(sr[1:7])
    # Verify that the output array matches the source
    np.testing.assert_array_equal(out.to_array(), sr[1:7].to_array())


def _load_ipc(header, frames, result_queue):
    cudf._gdf.rmm_initialize()

    try:
        out = deserialize(header, frames)
        result_queue.put(out)
    except Exception as e:
        result_queue.put(e)

    atexit.register(cudf._gdf.rmm_finalize)


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
