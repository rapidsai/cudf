# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.dtypes import CategoricalDtype, ListDtype
from cudf.tests.utils import assert_eq


def test_cdt_basic():
    psr = pd.Series(["a", "b", "a", "c"], dtype="category")
    sr = cudf.Series(["a", "b", "a", "c"], dtype="category")
    assert isinstance(sr.dtype, CategoricalDtype)
    assert_eq(sr.dtype.categories, psr.dtype.categories)


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
@pytest.mark.parametrize("ordered", [None, False, True])
def test_cdt_eq(data, ordered):
    dt = cudf.CategoricalDtype(categories=data, ordered=ordered)
    assert dt == "category"
    assert dt == dt
    assert dt == cudf.CategoricalDtype(categories=None, ordered=ordered)
    assert dt == cudf.CategoricalDtype(categories=data, ordered=ordered)
    assert not dt == cudf.CategoricalDtype(
        categories=data, ordered=not ordered
    )


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
@pytest.mark.parametrize("ordered", [None, False, True])
def test_cdf_to_pandas(data, ordered):
    assert (
        pd.CategoricalDtype(data, ordered)
        == cudf.CategoricalDtype(categories=data, ordered=ordered).to_pandas()
    )


@pytest.mark.parametrize(
    "value_type",
    [
        int,
        "int32",
        np.int32,
        "datetime64[ms]",
        "datetime64[ns]",
        "str",
        "object",
    ],
)
def test_list_dtype_pyarrow_round_trip(value_type):
    pa_type = pa.list_(cudf.utils.dtypes.np_to_pa_dtype(np.dtype(value_type)))
    expect = pa_type
    got = ListDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_dtype_eq():
    lhs = ListDtype("int32")
    rhs = ListDtype("int32")
    assert lhs == rhs
    rhs = ListDtype("int64")
    assert lhs != rhs


def test_nested_dtype():
    dt = ListDtype(ListDtype("int32"))
    expect = ListDtype("int32")
    got = dt.element_type
    assert expect == got


@pytest.mark.parametrize(
    "string,dtype",
    [
        ("uint8", cudf.UInt8Dtype),
        ("uint16", cudf.UInt16Dtype),
        ("uint32", cudf.UInt32Dtype),
        ("uint64", cudf.UInt64Dtype),
        ("UInt8", cudf.UInt8Dtype),
        ("UInt16", cudf.UInt16Dtype),
        ("UInt32", cudf.UInt32Dtype),
        ("UInt64", cudf.UInt64Dtype),
        ("int8", cudf.Int8Dtype),
        ("int16", cudf.Int16Dtype),
        ("int32", cudf.Int32Dtype),
        ("int64", cudf.Int64Dtype),
        ("Int8", cudf.Int8Dtype),
        ("Int16", cudf.Int16Dtype),
        ("Int32", cudf.Int32Dtype),
        ("Int64", cudf.Int64Dtype),
        ("int", cudf.Int64Dtype),
        ("float32", cudf.Float32Dtype),
        ("float64", cudf.Float64Dtype),
        ("Float32", cudf.Float32Dtype),
        ("Float64", cudf.Float64Dtype),
        ("float", cudf.Float64Dtype),
        ("bool", cudf.BooleanDtype),
        ("Boolean", cudf.BooleanDtype),
        ("string", cudf.StringDtype),
        ("String", cudf.StringDtype),
        ("object", cudf.StringDtype),
        ("datetime64[ns]", cudf.Datetime64NSDtype),
        ("datetime64[us]", cudf.Datetime64USDtype),
        ("datetime64[ms]", cudf.Datetime64MSDtype),
        ("datetime64[s]", cudf.Datetime64SDtype),
        ("Datetime64NS", cudf.Datetime64NSDtype),
        ("Datetime64US", cudf.Datetime64USDtype),
        ("Datetime64MS", cudf.Datetime64MSDtype),
        ("Datetime64S", cudf.Datetime64SDtype),
        ("timedelta64[ns]", cudf.Timedelta64NSDtype),
        ("timedelta64[us]", cudf.Timedelta64USDtype),
        ("timedelta64[ms]", cudf.Timedelta64MSDtype),
        ("timedelta64[s]", cudf.Timedelta64SDtype),
        ("Timedelta64NS", cudf.Timedelta64NSDtype),
        ("Timedelta64US", cudf.Timedelta64USDtype),
        ("Timedelta64MS", cudf.Timedelta64MSDtype),
        ("Timedelta64S", cudf.Timedelta64SDtype),
    ],
)
def test_cudf_dtype_string_construction(string, dtype):
    assert type(cudf.dtype(string) == dtype)
