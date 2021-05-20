# Copyright (c) 2018-2021, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.api import types as types


@pytest.mark.parametrize(
    "obj, expect",
    (
        (True, False),
        (np.bool_, False),
        (np.int_, False),
        (np.float64, False),
        (complex, False),
        (np.complex128, False),
        (np.bool_(1), False),
        (np.int_(1), False),
        (np.float64(1), False),
        (complex(1), False),
        (np.complex128(1), False),
        (np.dtype("int"), False),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.datetime64("2005-02-25T03:30"), False),
        (cudf.Series(["2005-02-25T03:30"], dtype="datetime64[s]"), False),
        (cudf.Series([1000], dtype="timedelta64[s]"), False),
        (pd.Series(["2005-02-25T03:30"], dtype="datetime64[s]"), False),
        (1, False),
        (1.0, False),
        ("hello", False),
        (cudf.Series("hello"), False),
        (cudf.Series("hello"), False),
        (cudf.CategoricalDtype, True),
        (cudf.CategoricalDtype("a"), True),
        (cudf.Series(["a"], dtype="category"), True),
        (cudf.Series([1, 2], dtype=cudf.Decimal64Dtype(5, 2)), False),
        (cudf.Decimal64Dtype(5, 2), False),
        (cudf.Decimal64Dtype, False),
        (pd.core.dtypes.dtypes.CategoricalDtypeType, True),
        (pd.CategoricalDtype, True),
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.StructDtype, False),
        (cudf.ListDtype, False),
    ),
)
def test_is_categorical_dtype(obj, expect):
    assert types.is_categorical_dtype(obj) == expect
