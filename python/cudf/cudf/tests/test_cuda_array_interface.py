# Copyright (c) 2019, NVIDIA CORPORATION.

import types
from contextlib import ExitStack as does_not_raise

import numpy as np
import pandas as pd
import pytest
from numba import cuda

import cudf
from cudf.tests.utils import assert_eq

try:
    import cupy

    _have_cupy = True
except ImportError:
    _have_cupy = False

basic_dtypes = [
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("float32"),
    np.dtype("float64"),
]
string_dtypes = [np.dtype("object")]
datetime_dtypes = [
    # np.dtype("datetime64[ns]"),
    # np.dtype("datetime64[us]"),
    np.dtype("datetime64[ms]"),
    # np.dtype("datetime64[s]"),
]


def data_type_expectation_builder(data):
    if data.size > 0:
        return does_not_raise()
    else:
        return pytest.raises((ValueError, IndexError))


@pytest.mark.parametrize("dtype", basic_dtypes + datetime_dtypes)
@pytest.mark.parametrize("module", ["cupy", "numba"])
def test_cuda_array_interface_interop_in(dtype, module):
    np_data = np.arange(10).astype(dtype)

    expectation = does_not_raise()
    if module == "cupy":
        if not _have_cupy:
            pytest.skip("no cupy")
        if dtype in datetime_dtypes:
            expectation = pytest.raises([KeyError])
        with expectation:
            module_data = cupy.array(np_data)
    elif module == "numba":
        module_data = cuda.to_device(np_data)

    pd_data = pd.Series(np_data)

    # Test using a specific function for __cuda_array_interface__ here
    cudf_data = cudf.Series(module_data)

    assert_eq(pd_data, cudf_data)

    gdf = cudf.DataFrame()
    gdf["test"] = module_data
    pd_data.name = "test"
    assert_eq(pd_data, gdf["test"])


@pytest.mark.parametrize(
    "dtype", basic_dtypes + datetime_dtypes + string_dtypes
)
@pytest.mark.parametrize("module", ["cupy", "numba"])
def test_cuda_array_interface_interop_out(dtype, module):
    np_data = np.arange(10).astype(dtype)
    cudf_data = cudf.Series(np_data)

    expectation = does_not_raise()
    if dtype in string_dtypes:
        expectation = pytest.raises(NotImplementedError)
    if module == "cupy":
        if not _have_cupy:
            pytest.skip("no cupy")
        if dtype in datetime_dtypes:
            expectation = pytest.raises([KeyError])
        with expectation:
            module_data = cupy.asarray(cudf_data)
            got = cupy.asnumpy(module_data)
    elif module == "numba":
        with expectation:
            module_data = cuda.as_cuda_array(cudf_data)
            got = module_data.copy_to_host()

    expect = np_data
    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", basic_dtypes + datetime_dtypes)
@pytest.mark.parametrize("nulls", ["all", "some", "none"])
def test_cuda_array_interface_as_column(dtype, nulls):
    sr = cudf.Series(np.arange(10))

    if nulls == "some":
        sr[[1, 3, 4, 7]] = None
    elif nulls == "all":
        sr[:] = None

    sr = sr.astype(dtype)

    obj = types.SimpleNamespace(
        __cuda_array_interface__=sr.__cuda_array_interface__
    )

    expect = sr
    got = cudf.Series(obj)

    assert_eq(expect, got)
