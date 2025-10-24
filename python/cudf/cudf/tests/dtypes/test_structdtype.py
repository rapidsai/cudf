# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pytest

import cudf


@pytest.mark.parametrize(
    "fields",
    [
        {"a": np.dtype(np.int64)},
        {"a": np.dtype(np.int64), "b": None},
        {
            "a": cudf.ListDtype(np.dtype(np.int64)),
            "b": cudf.Decimal64Dtype(1, 0),
        },
        {
            "a": cudf.ListDtype(cudf.StructDtype({"b": np.dtype(np.int64)})),
            "b": cudf.ListDtype(cudf.ListDtype(np.dtype(np.int64))),
        },
    ],
)
def test_serialize_struct_dtype(fields):
    dtype = cudf.StructDtype(fields)
    recreated = dtype.__class__.device_deserialize(*dtype.device_serialize())
    assert recreated == dtype


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
        {k: pa.from_numpy_dtype(np.dtype(v)) for k, v in fields.items()}
    )
    expect = pa_type
    got = cudf.StructDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_struct_dtype_eq():
    lhs = cudf.StructDtype(
        {"a": "int32", "b": cudf.StructDtype({"c": "int64", "ab": "int32"})}
    )
    rhs = cudf.StructDtype(
        {"a": "int32", "b": cudf.StructDtype({"c": "int64", "ab": "int32"})}
    )
    assert lhs == rhs
    rhs = cudf.StructDtype({"a": "int32", "b": "int64"})
    assert lhs != rhs
    lhs = cudf.StructDtype({"b": "int64", "a": "int32"})
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
        {"a": "int32", "b": cudf.StructDtype({"a": "int32", "b": "int64"})},
    ],
)
def test_struct_dtype_fields(fields):
    fields = {
        "a": "int32",
        "b": cudf.StructDtype({"c": "int64", "d": "int32"}),
    }
    dt = cudf.StructDtype(fields)
    assert dt.fields == fields
