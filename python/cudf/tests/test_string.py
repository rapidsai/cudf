# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda

from cudf import concat
from cudf.dataframe import Series
from cudf.tests.utils import assert_eq
from librmm_cffi import librmm as rmm


data_list = [
    ['a', 'b', 'c', 'd', 'e'],
    ['a', None, 'c', None, 'e'],
    [None, None, None, None, None]
]

data_id_list = [
    "no_nulls",
    "some_nulls",
    "all_nulls"
]

idx_list = [
    None,
    [10, 11, 12, 13, 14]
]

idx_id_list = [
    "None_index",
    "Set_index"
]


@pytest.fixture(params=data_list, ids=data_id_list)
def data(request):
    return request.param


@pytest.fixture(params=idx_list, ids=idx_id_list)
def index(request):
    return request.param


@pytest.fixture
def ps_gs(data, index):
    ps = pd.Series(data, index=index)
    gs = Series(data, index=index)
    return (ps, gs)


@pytest.mark.parametrize('construct', [list, np.array, pd.Series, pa.array])
def test_string_ingest(construct):
    expect = ['a', 'a', 'b', 'c', 'a']
    data = construct(expect)
    got = Series(data)
    assert got.dtype == np.dtype('str')
    assert len(got) == 5
    for idx, val in enumerate(expect):
        assert expect[idx] == got[idx]


def test_string_export(ps_gs):
    ps, gs = ps_gs

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
def test_string_get_item(ps_gs, item):
    ps, gs = ps_gs

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
def test_string_repr(ps_gs, item):
    ps, gs = ps_gs

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


def test_string_concat():
    data1 = ['a', 'b', 'c', 'd', 'e']
    data2 = ['f', 'g', 'h', 'i', 'j']

    ps1 = pd.Series(data1)
    ps2 = pd.Series(data2)
    gs1 = Series(data1)
    gs2 = Series(data2)

    expect = pd.concat([ps1, ps2])
    got = concat([gs1, gs2])

    assert_eq(expect, got)


def test_string_len(ps_gs):
    ps, gs = ps_gs

    expect = ps.str.len()
    got = gs.str.len()

    assert_eq(expect, got)


@pytest.mark.parametrize('others', [
    None,
    ['f', 'g', 'h', 'i', 'j'],
    pd.Series(['f', 'g', 'h', 'i', 'j'])
])
@pytest.mark.parametrize('sep', [None, '', ' ', '|', ',', '|||'])
@pytest.mark.parametrize('na_rep', [None, '', 'null', 'a'])
def test_string_cat(ps_gs, others, sep, na_rep):
    ps, gs = ps_gs

    expect = ps.str.cat(others=others, sep=sep, na_rep=na_rep)
    if isinstance(others, pd.Series):
        others = Series(others)
    got = gs.str.cat(others=others, sep=sep, na_rep=na_rep)

    assert_eq(expect, got)


@pytest.mark.parametrize('sep', [None, '', ' ', '|', ',', '|||'])
def test_string_join(data, sep):
    ps, gs = ps_gs

    expect = ps.str.join(sep)
    got = gs.str.join(sep)

    assert_eq(expect, got)
