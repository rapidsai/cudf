# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data,expected_dtype",
    [
        ([10, 11, 12], pd.Int64Dtype()),
        ([0.1, 10.2, 12.3], pd.Float64Dtype()),
        (["abc", None, "def"], pd.StringDtype()),
    ],
)
def test_index_to_pandas_nullable(data, expected_dtype):
    gi = cudf.Index(data)
    pi = gi.to_pandas(nullable=True)
    expected = pd.Index(data, dtype=expected_dtype)

    assert_eq(pi, expected)
