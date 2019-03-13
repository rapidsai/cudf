# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
import pytest
import itertools

import numpy as np

from cudf.tests.utils import assert_eq

try:
    import cupy
    _have_cupy = True
except ImportError:
    _have_cupy = False

require_cupy = pytest.mark.skipif(not _have_cupy, reason='no cupy')

nelems = [0, 1, 10]
data = [np.nan, 10, 10.0]
params_1d = itertools.product(nelems, data)

ncols = [0, 1, 2]
params_2d = itertools.product(ncols, nelems, data)


@pytest.fixture(params=params_1d)
def data_1d(request):
    return np.array([request.param[1]] * request.param[0])


@pytest.fixture(params=params_2d)
def data_2d(request):
    a = np.ascontiguousarray(np.array([[request.param[2]] * request.param[0]] *
                             request.param[1]))
    return a

def test_to_dlpack_dataframe(data_2d):
    if (data_2d.size > 0):
        gdf = cudf.DataFrame.from_records(data_2d)
        dlt = gdf.to_dlpack()

        # PyCapsules are a C-API thing so couldn't come up with a better way
        assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_series(data_1d):
    gs = cudf.Series(data)
    dlt = gs.to_dlpack()

    # PyCapsules are a C-API thing so couldn't come up with a better way
    assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_index(data_1d):
    gi = cudf.dataframe.index.as_index(data)
    dlt = gi.to_dlpack()

    # PyCapsules are a C-API thing so couldn't come up with a better way
    assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_column(data_1d):
    gs = cudf.Series(data)
    dlt = gs._column.to_dlpack()

    # PyCapsules are a C-API thing so couldn't come up with a better way
    assert str(type(dlt)) == "<class 'PyCapsule'>"


@require_cupy
def test_to_dlpack_cupy_1d(data_1d):
    gs = cudf.Series(data)
    cudf_host_array = gs.to_array()
    dlt = gs._column.to_dlpack()

    cupy_array = cupy.fromDlpack(dlt)
    cupy_host_array = cupy_array.get()

    assert_eq(cudf_host_array, cupy_host_array)


@require_cupy
def test_to_dlpack_cupy_2d(data_2d):
    gdf = cudf.DataFrame.from_records(data_2d)
    cudf_host_array = np.array(gdf.to_pandas())
    dlt = gdf.to_dlpack()

    cupy_array = cupy.fromDlpack(dlt)
    cupy_host_array = cupy_array.get()

    assert_eq(cudf_host_array, cupy_host_array)


@require_cupy
def test_from_dlpack_cupy_1d(data_1d):
    cupy_array = cupy.array(data)
    cupy_host_array = cupy_array.get()
    dlt = cupy_array.toDlpack()

    gs = cudf.from_dlpack(dlt)
    cudf_host_array = gs.to_array()

    assert_eq(cudf_host_array, cupy_host_array)


@require_cupy
def test_from_dlpack_cupy_2d(data_2d):
    cupy_array = cupy.array(data_2d)
    cupy_host_array = cupy_array.get()
    dlt = cupy_array.toDlpack()

    gdf = cudf.from_dlpack(dlt)
    cudf_host_array = np.array(gdf.to_pandas())

    assert_eq(cudf_host_array, cupy_host_array)
