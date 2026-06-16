# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ["sum", {}],
        ["mean", {}],
        ["median", {}],
        ["std", {}],
        ["std", {"ddof": 0}],
    ],
)
def test_tdi_reductions(method, kwargs):
    pd_tdi = pd.TimedeltaIndex(["1 day", "2 days", "3 days"])
    cudf_tdi = cudf.from_pandas(pd_tdi)

    result = getattr(pd_tdi, method)(**kwargs)
    expected = getattr(cudf_tdi, method)(**kwargs)
    assert result == expected
