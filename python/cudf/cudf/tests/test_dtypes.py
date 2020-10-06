# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.dtypes import CategoricalDtype, ListDtype, StructDtype
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


def test_list_dtype_eq():
    lhs = ListDtype("int32")
    rhs = ListDtype("int32")
    assert lhs == rhs
    rhs = ListDtype("int64")
    assert lhs != rhs


def test_list_nested_dtype():
    dt = ListDtype(ListDtype("int32"))
    expect = ListDtype("int32")
    got = dt.element_type
    assert expect == got


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"a": "int64"},
        {"a": "datetime64[ms]"},
        {"a": "int32", "b": "int64"},
    ],
)
def test_struct_dtype_pyarrow_round_trip(fields):
    pa_type = pa.struct(
        {
            k: cudf.utils.dtypes.np_to_pa_dtype(np.dtype(v))
            for k, v in fields.items()
        }
    )
    expect = pa_type
    got = StructDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_struct_dtype_eq():
    lhs = StructDtype(
        {"a": "int32", "b": StructDtype({"c": "int64", "ab": "int32"})}
    )
    rhs = StructDtype(
        {"a": "int32", "b": StructDtype({"c": "int64", "ab": "int32"})}
    )
    assert lhs == rhs
    rhs = StructDtype({"a": "int32", "b": "int64"})
    assert lhs != rhs
    lhs = StructDtype({"b": "int64", "a": "int32"})
    assert lhs != rhs


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"a": "int32"},
        {"a": "object"},
        {"a": "str"},
        {"a": "datetime64[D]"},
        {"a": "int32", "b": "int64"},
        {"a": "int32", "b": StructDtype({"a": "int32", "b": "int64"})},
    ],
)
def test_struct_dtype_fields(fields):
    fields = {"a": "int32", "b": StructDtype({"c": "int64", "d": "int32"})}
    dt = StructDtype(fields)
    assert_eq(dt.fields, fields)
