# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [],
        [None, None],
        [
            "2020-05-31 08:00:00",
            "1999-12-31 18:40:00",
            "2000-12-31 04:00:00",
        ],
        ["2100-03-14 07:30:00"],
    ],
)
def test_isocalendar_index(data):
    ps = pd.DatetimeIndex(data, dtype="datetime64[ns]")
    gs = cudf.from_pandas(ps)

    expect = ps.isocalendar()
    got = gs.isocalendar()

    assert_eq(expect, got, check_dtype=False)
