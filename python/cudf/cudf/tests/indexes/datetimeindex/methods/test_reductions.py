# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "method, kwargs",
    [["mean", {}], ["std", {}], ["std", {"ddof": 0}]],
)
def test_dti_reduction(method, kwargs):
    pd_dti = pd.DatetimeIndex(["2020-01-01", "2020-12-31"], name="foo")
    cudf_dti = cudf.from_pandas(pd_dti)

    result = getattr(cudf_dti, method)(**kwargs)
    expected = getattr(pd_dti, method)(**kwargs)
    assert result == expected
