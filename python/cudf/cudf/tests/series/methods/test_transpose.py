# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3],
        ["abc", "a", None, "hello world", "foo buzz", "", None, "rapids ai"],
    ],
)
def test_series_transpose(data):
    psr = pd.Series(data=data)
    csr = cudf.Series(data=data)

    cudf_transposed = csr.transpose()
    pd_transposed = psr.transpose()
    cudf_property = csr.T
    pd_property = psr.T

    assert_eq(pd_transposed, cudf_transposed)
    assert_eq(pd_property, cudf_property)
    assert_eq(cudf_transposed, csr)
