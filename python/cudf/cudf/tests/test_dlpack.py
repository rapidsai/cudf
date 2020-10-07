# Copyright (c) 2019, NVIDIA CORPORATION.

import itertools
from contextlib import ExitStack as does_not_raise

import cupy
import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

nelems = [0, 3, 10]
dtype = [np.uint16, np.int32, np.float64]
nulls = ["some", "none"]
params_1d = itertools.product(nelems, dtype, nulls)

ncols = [0, 1, 2]
params_2d = itertools.product(ncols, nelems, dtype, nulls)


def data_size_expectation_builder(data, nan_null_param=False):
    if nan_null_param and np.isnan(data).any():
        return pytest.raises((ValueError,))

    if data.size > 0:
        return does_not_raise()
    else:
        return pytest.raises((ValueError, IndexError))


@pytest.fixture(params=params_1d)
def data_1d(request):
    nelems = request.param[0]
    dtype = request.param[1]
    nulls = request.param[2]
    a = np.random.randint(10, size=nelems).astype(dtype)
    if nulls == "some" and a.size != 0 and np.issubdtype(dtype, np.floating):
        idx = np.random.choice(a.size, size=int(a.size * 0.2), replace=False)
        a[idx] = np.nan
    return a


@pytest.fixture(params=params_2d)
def data_2d(request):
    ncols = request.param[0]
    nrows = request.param[1]
    dtype = request.param[2]
    nulls = request.param[3]
    a = np.random.randint(10, size=(nrows, ncols)).astype(dtype)
    if nulls == "some" and a.size != 0 and np.issubdtype(dtype, np.floating):
        idx = np.random.choice(a.size, size=int(a.size * 0.2), replace=False)
        a.ravel()[idx] = np.nan
    return np.ascontiguousarray(a)


def test_to_dlpack_dataframe(data_2d):
    expectation = data_size_expectation_builder(data_2d)

    with expectation:
        gdf = cudf.DataFrame.from_records(data_2d)
        dlt = gdf.to_dlpack()

        # PyCapsules are a C-API thing so couldn't come up with a better way
        assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_series(data_1d):
    expectation = data_size_expectation_builder(data_1d, nan_null_param=False)

    with expectation:
        gs = cudf.Series(data_1d, nan_as_null=False)
        dlt = gs.to_dlpack()

        # PyCapsules are a C-API thing so couldn't come up with a better way
        assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_series_null(data_1d):
    expectation = data_size_expectation_builder(data_1d, nan_null_param=True)

    with expectation:
        gs = cudf.Series(data_1d, nan_as_null=True)
        dlt = gs.to_dlpack()

        # PyCapsules are a C-API thing so couldn't come up with a better way
        assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_index(data_1d):
    expectation = data_size_expectation_builder(data_1d)

    with expectation:
        if np.isnan(data_1d).any():
            pytest.skip("Nulls not allowed in Index")
        gi = cudf.core.index.as_index(data_1d)
        dlt = gi.to_dlpack()

        # PyCapsules are a C-API thing so couldn't come up with a better way
        assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_column(data_1d):
    expectation = data_size_expectation_builder(data_1d)

    with expectation:
        gs = cudf.Series(data_1d, nan_as_null=False)
        dlt = gs._column.to_dlpack()

        # PyCapsules are a C-API thing so couldn't come up with a better way
        assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_column_null(data_1d):
    expectation = data_size_expectation_builder(data_1d, nan_null_param=True)

    with expectation:
        gs = cudf.Series(data_1d, nan_as_null=True)
        dlt = gs._column.to_dlpack()

        # PyCapsules are a C-API thing so couldn't come up with a better way
        assert str(type(dlt)) == "<class 'PyCapsule'>"


def test_to_dlpack_cupy_1d(data_1d):
    expectation = data_size_expectation_builder(data_1d, False)
    with expectation:
        gs = cudf.Series(data_1d, nan_as_null=False)
        cudf_host_array = gs.to_array(fillna="pandas")
        dlt = gs._column.to_dlpack()

        cupy_array = cupy.fromDlpack(dlt)
        cupy_host_array = cupy_array.get()

        assert_eq(cudf_host_array, cupy_host_array)


def test_to_dlpack_cupy_2d(data_2d):
    expectation = data_size_expectation_builder(data_2d)

    with expectation:
        gdf = cudf.DataFrame.from_records(data_2d)
        cudf_host_array = np.array(gdf.to_pandas()).flatten()
        dlt = gdf.to_dlpack()

        cupy_array = cupy.fromDlpack(dlt)
        cupy_host_array = cupy_array.get().flatten()

        assert_eq(cudf_host_array, cupy_host_array)


def test_from_dlpack_cupy_1d(data_1d):
    cupy_array = cupy.array(data_1d)
    cupy_host_array = cupy_array.get()
    dlt = cupy_array.toDlpack()

    gs = cudf.from_dlpack(dlt)
    cudf_host_array = gs.to_array(fillna="pandas")

    assert_eq(cudf_host_array, cupy_host_array)


def test_from_dlpack_cupy_2d(data_2d):
    cupy_array = cupy.array(data_2d, order="F")
    cupy_host_array = cupy_array.get().flatten()
    dlt = cupy_array.toDlpack()

    gdf = cudf.from_dlpack(dlt)
    cudf_host_array = np.array(gdf.to_pandas()).flatten()

    assert_eq(cudf_host_array, cupy_host_array)


def test_to_dlpack_cupy_2d_null(data_2d):
    expectation = data_size_expectation_builder(data_2d, nan_null_param=True)

    with expectation:
        gdf = cudf.DataFrame.from_records(data_2d, nan_as_null=True)
        cudf_host_array = np.array(gdf.to_pandas()).flatten()
        dlt = gdf.to_dlpack()

        cupy_array = cupy.fromDlpack(dlt)
        cupy_host_array = cupy_array.get().flatten()

        assert_eq(cudf_host_array, cupy_host_array)


def test_to_dlpack_cupy_1d_null(data_1d):
    expectation = data_size_expectation_builder(data_1d, nan_null_param=True)

    with expectation:
        gs = cudf.Series(data_1d)
        cudf_host_array = gs.to_array(fillna="pandas")
        dlt = gs._column.to_dlpack()

        cupy_array = cupy.fromDlpack(dlt)
        cupy_host_array = cupy_array.get()

        assert_eq(cudf_host_array, cupy_host_array)
