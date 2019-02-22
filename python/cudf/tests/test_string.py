# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda

from cudf.dataframe import Series
# from cudf.tests.utils import assert_eq
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


@pytest.mark.parametrize('item', [0, 2, 4, slice(1, 3),
                                  np.array([0, 1, 2, 3, 4]),
                                  rmm.to_device(np.array([0, 1, 2, 3, 4]))])
def test_string_get_item(item):
    data = ['a', 'b', 'c', 'd', 'e']
    ps = pd.Series(data)
    gs = Series(data)

    got = gs[item].to_arrow()
    if isinstance(item, cuda.devicearray.DeviceNDArray):
        item = item.copy_to_host()
    expect = pa.Array.from_pandas(ps[item])

    pa.Array.equals(expect, got)


@pytest.mark.parametrize('item', [0, 2, 4, slice(1, 3, 1)])
def test_string_repr(item):
    data = ['a', 'b', 'c', 'd', 'e']
    ps = pd.Series(data)
    gs = Series(data)

    expect = str(ps[item])
    got = str(gs[item])

    assert expect == got
