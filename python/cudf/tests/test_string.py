# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda

from cudf.dataframe import Series
from cudf.tests.utils import assert_eq
from librmm_cffi import librmm as rmm


@pytest.mark.parametrize('construct', [list, np.array, pd.Series, pa.array])
def test_string_ingest(construct):
    expect = ['a', 'a', 'b', 'c', 'a']
    data = construct(expect)
    got = Series(data)
    assert got.dtype == np.dtype('str')
    assert len(got) == 5


def test_string_export():
    data = ['a', 'a', 'b', 'c', 'a']
    ps = pd.Series(data)
    gs = Series(data)

    expect = ps
    got = gs.to_pandas()
    pd.testing.assert_series_equal(expect, got)

    expect = np.array(ps)
    got = gs.to_array()
    np.testing.assert_array_equal(expect, got)

    expect = pa.Array.from_pandas(ps)
    print(expect)
    got = gs.to_arrow()
    print(got)
    assert pa.Array.equals(expect, got)


@pytest.mark.parametrize(
    'item',
    [
        0,
        2,
        4,
        slice(1, 3),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        np.array([0, 1, 2, 3, 4]),
        rmm.to_device(np.array([0, 1, 2, 3, 4])),
        [True] * 5,
        [False] * 5,
        np.array([True] * 5),
        np.array([False] * 5),
        rmm.to_device(np.array([True] * 5)),
        rmm.to_device(np.array([False] * 5)),
        list(np.random.randint(0, 2, 5).astype('bool')),
        np.random.randint(0, 2, 5).astype('bool'),
        rmm.to_device(np.random.randint(0, 2, 5).astype('bool'))
    ]
)
def test_string_get_item(item):
    data = ['a', 'b', 'c', 'd', 'e']
    ps = pd.Series(data)
    gs = Series(data)

    got = gs[item]
    if isinstance(got, Series):
        got = got.to_arrow()

    if isinstance(item, cuda.devicearray.DeviceNDArray):
        item = item.copy_to_host()

    expect = ps[item]
    if isinstance(expect, pd.Series):
        expect = pa.Array.from_pandas(expect)
        pa.Array.equals(expect, got)
    else:
        assert expect == got


@pytest.mark.parametrize('item', [0, slice(1, 3), slice(5)])
def test_string_repr(item):
    data = ['a', 'b', 'c', 'd', 'e']
    ps = pd.Series(data)
    gs = Series(data)

    got_out = gs[item]
    expect_out = ps[item]

    expect = str(expect_out)
    got = str(got_out)

    if isinstance(expect_out, pd.Series):
        expect = expect.replace("object", "str")

    assert expect == got


@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64',
                                   'float32', 'float64'])
def test_string_astype(dtype):
    if dtype.startswith('int'):
        data = ["1", "2", "3", "4", "5"]
    elif dtype.startswith('float'):
        data = ["1.0", "2.0", "3.0", "4.0", "5.0"]
    ps = pd.Series(data)
    gs = Series(data)

    expect = ps.astype(dtype)
    got = gs.astype(dtype)

    assert_eq(expect, got)
