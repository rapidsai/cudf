# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
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
