# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ["to_pydatetime", {}],
        ["to_period", {"freq": "D"}],
        ["strftime", {"date_format": "%Y-%m-%d"}],
    ],
)
def test_dti_methods(method, kwargs):
    pd_dti = pd.DatetimeIndex(["2020-01-01", "2020-12-31"], name="foo")
    cudf_dti = cudf.from_pandas(pd_dti)

    result = getattr(cudf_dti, method)(**kwargs)
    expected = getattr(pd_dti, method)(**kwargs)
    assert_eq(result, expected)
