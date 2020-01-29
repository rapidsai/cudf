# Copyright (c) 2019, NVIDIA CORPORATION.

import types
from contextlib import ExitStack as does_not_raise

import cupy
import numpy as np
import pandas as pd
import pytest
from numba import cuda

import cudf
from cudf.tests.utils import assert_eq

basic_dtypes = [
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("float32"),
    np.dtype("float64"),
]
string_dtypes = [np.dtype("str")]
datetime_dtypes = [
    np.dtype("datetime64[ns]"),
    np.dtype("datetime64[us]"),
    np.dtype("datetime64[ms]"),
    np.dtype("datetime64[s]"),
]


@pytest.mark.parametrize("dtype", basic_dtypes + datetime_dtypes)
@pytest.mark.parametrize("module", ["cupy", "numba"])
def test_cuda_array_interface_interop_in(dtype, module):
    np_data = np.arange(10).astype(dtype)

    expectation = does_not_raise()
    if module == "cupy":
        module_constructor = cupy.array
        if dtype in datetime_dtypes:
            expectation = pytest.raises(ValueError)
    elif module == "numba":
        module_constructor = cuda.to_device

    with expectation:
        module_data = module_constructor(np_data)

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
    expectation = does_not_raise()
    if dtype in string_dtypes:
        expectation = pytest.raises(NotImplementedError)
    if module == "cupy":
        module_constructor = cupy.asarray

        def to_host_function(x):
            return cupy.asnumpy(x)

    elif module == "numba":
        module_constructor = cuda.as_cuda_array

        def to_host_function(x):
            return x.copy_to_host()

    with expectation:
        np_data = np.arange(10).astype(dtype)
        cudf_data = cudf.Series(np_data)
        assert isinstance(cudf_data.__cuda_array_interface__, dict)

        module_data = module_constructor(cudf_data)
        got = to_host_function(module_data)

        expect = np_data

        assert_eq(expect, got)


@pytest.mark.parametrize("dtype", basic_dtypes + datetime_dtypes)
@pytest.mark.parametrize("module", ["cupy", "numba"])
def test_cuda_array_interface_interop_out_masked(dtype, module):
    expectation = does_not_raise()
    if module == "cupy":
        pytest.skip(
            "cupy doesn't support version 1 of "
            "`__cuda_array_interface__` yet"
        )
        module_constructor = cupy.asarray

        def to_host_function(x):
            return cupy.asnumpy(x)

    elif module == "numba":
        expectation = pytest.raises(NotImplementedError)
        module_constructor = cuda.as_cuda_array

        def to_host_function(x):
            return x.copy_to_host()

    np_data = np.arange(10).astype("float64")
    np_data[[0, 2, 4, 6, 8]] = np.nan

    with expectation:
        cudf_data = cudf.Series(np_data).astype(dtype)
        assert isinstance(cudf_data.__cuda_array_interface__, dict)

        module_data = module_constructor(cudf_data)  # noqa: F841


@pytest.mark.parametrize("dtype", basic_dtypes + datetime_dtypes)
@pytest.mark.parametrize("nulls", ["all", "some", "bools", "none"])
@pytest.mark.parametrize("mask_type", ["bits", "bools"])
def test_cuda_array_interface_as_column(dtype, nulls, mask_type):
    sr = cudf.Series(np.arange(10))

    if nulls == "some":
        sr[[1, 3, 4, 7]] = None
    elif nulls == "all":
        sr[:] = None

    sr = sr.astype(dtype)

    obj = types.SimpleNamespace(
        __cuda_array_interface__=sr.__cuda_array_interface__
    )

    if mask_type == "bools":
        if nulls == "some":
            obj.__cuda_array_interface__["mask"] = cuda.to_device(
                [
                    True,
                    False,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                ]
            )
        elif nulls == "all":
            obj.__cuda_array_interface__["mask"] = cuda.to_device([False] * 10)

    expect = sr
    got = cudf.Series(obj)

    assert_eq(expect, got)


def test_column_from_ephemeral_cupy():
    # Test that we keep a reference to the ephemeral
    # CuPy array. If we didn't, then `a` would end
    # up referring to the same memory as `b` due to
    # CuPy's caching allocator
    a = cudf.Series(cupy.asarray([1, 2, 3]))
    b = cudf.Series(cupy.asarray([1, 1, 1]))
    assert_eq(pd.Series([1, 2, 3]), a)
    assert_eq(pd.Series([1, 1, 1]), b)


def test_column_from_ephemeral_cupy_try_lose_reference():
    # Try to lose the reference we keep to the ephermal
    # CuPy array
    a = cudf.Series(cupy.asarray([1, 2, 3]))._column
    a = cudf.core.column.as_column(a)
    b = cupy.asarray([1, 1, 1])  # noqa: F841
    assert_eq(pd.Series([1, 2, 3]), a.to_pandas())

    a = cudf.Series(cupy.asarray([1, 2, 3]))._column
    a.name = "b"
    b = cupy.asarray([1, 1, 1])  # noqa: F841
    assert_eq(pd.Series([1, 2, 3]), a.to_pandas())
