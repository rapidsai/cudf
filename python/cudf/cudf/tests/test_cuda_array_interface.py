# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import types
from contextlib import ExitStack as does_not_raise

import cupy
import numba.cuda
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.testing import assert_eq
from cudf.testing._utils import DATETIME_TYPES, NUMERIC_TYPES, TIMEDELTA_TYPES


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + DATETIME_TYPES)
@pytest.mark.parametrize("module", ["cupy", "numba"])
def test_cuda_array_interface_interop_in(dtype, module):
    np_data = np.arange(10).astype(dtype)

    expectation = does_not_raise()
    if module == "cupy":
        module_constructor = cupy.array
        if dtype in DATETIME_TYPES:
            expectation = pytest.raises(ValueError)
    elif module == "numba":
        module_constructor = numba.cuda.to_device

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
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES + ["str"]
)
@pytest.mark.parametrize("module", ["cupy", "numba"])
def test_cuda_array_interface_interop_out(dtype, module):
    expectation = does_not_raise()
    if dtype == "str":
        expectation = pytest.raises(AttributeError)
    if module == "cupy":
        module_constructor = cupy.asarray

        def to_host_function(x):
            return cupy.asnumpy(x)

    elif module == "numba":
        module_constructor = numba.cuda.as_cuda_array

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


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES
)
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
        module_constructor = numba.cuda.as_cuda_array

        def to_host_function(x):
            return x.copy_to_host()

    np_data = np.arange(10).astype("float64")
    np_data[[0, 2, 4, 6, 8]] = np.nan

    with expectation:
        cudf_data = cudf.Series(np_data).astype(dtype)
        assert isinstance(cudf_data.__cuda_array_interface__, dict)

        module_data = module_constructor(cudf_data)  # noqa: F841


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES
)
@pytest.mark.parametrize("nulls", ["all", "some", "bools", "none"])
@pytest.mark.parametrize("mask_type", ["bits", "bools"])
def test_cuda_array_interface_as_column(dtype, nulls, mask_type):
    sr = cudf.Series(np.arange(10))

    if nulls == "some":
        mask = [
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
        sr[sr[~np.asarray(mask)]] = None
    elif nulls == "all":
        sr[:] = None

    sr = sr.astype(dtype)

    obj = types.SimpleNamespace(
        __cuda_array_interface__=sr.__cuda_array_interface__
    )

    if mask_type == "bools":
        if nulls == "some":
            obj.__cuda_array_interface__["mask"] = numba.cuda.to_device(mask)
        elif nulls == "all":
            obj.__cuda_array_interface__["mask"] = numba.cuda.to_device(
                [False] * 10
            )

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
    # Try to lose the reference we keep to the ephemeral
    # CuPy array
    a = cudf.Series(cupy.asarray([1, 2, 3]))._column
    a = cudf.core.column.as_column(a)
    b = cupy.asarray([1, 1, 1])
    assert_eq(pd.Index([1, 2, 3]), a.to_pandas())

    a = cudf.Series(cupy.asarray([1, 2, 3]))._column
    a.name = "b"
    b = cupy.asarray([1, 1, 1])  # noqa: F841
    assert_eq(pd.Index([1, 2, 3]), a.to_pandas())


@pytest.mark.xfail(
    get_global_manager() is not None,
    reason=(
        "spilling doesn't support PyTorch, see "
        "`cudf.core.buffer.spillable_buffer.DelayedPointerTuple`"
    ),
)
def test_cuda_array_interface_pytorch():
    torch = pytest.importorskip("torch", minversion="2.4.0")
    if not torch.cuda.is_available():
        pytest.skip("need gpu version of pytorch to be installed")

    series = cudf.Series([1, -1, 10, -56])
    tensor = torch.tensor(series)
    got = cudf.Series(tensor)

    assert_eq(got, series)
    buffer = cudf.core.buffer.as_buffer(cupy.ones(10, dtype=np.bool_))
    tensor = torch.tensor(buffer)
    got = cudf.Series(tensor, dtype=np.bool_)

    assert_eq(got, cudf.Series(buffer, dtype=np.bool_))

    index = cudf.Index([], dtype="float64")
    tensor = torch.tensor(index)
    got = cudf.Index(tensor)
    assert_eq(got, index)

    index = cudf.core.index.RangeIndex(start=0, stop=100)
    tensor = torch.tensor(index)
    got = cudf.Series(tensor)

    assert_eq(got, cudf.Series(index))

    index = cudf.Index([1, 2, 8, 6])
    tensor = torch.tensor(index)
    got = cudf.Index(tensor)

    assert_eq(got, index)

    str_series = cudf.Series(["a", "g"])

    with pytest.raises(AttributeError):
        str_series.__cuda_array_interface__

    cat_series = str_series.astype("category")

    with pytest.raises(TypeError):
        cat_series.__cuda_array_interface__


def test_cai_after_indexing():
    df = cudf.DataFrame({"a": [1, 2, 3]})
    cai1 = df["a"].__cuda_array_interface__
    df[["a"]]
    cai2 = df["a"].__cuda_array_interface__
    assert cai1 == cai2
