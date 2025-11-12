# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_dropna_bad_how():
    with pytest.raises(ValueError):
        cudf.Index([1]).dropna(how="foo")


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, float("nan"), 2], "float64"),
        (["x", None, "y"], "str"),
        (["x", None, "y"], "category"),
        (["2020-01-20", pd.NaT, "2020-03-15"], "datetime64[ns]"),
        (["1s", pd.NaT, "3d"], "timedelta64[ns]"),
    ],
)
def test_dropna_index(data, dtype):
    pi = pd.Index(data, dtype=dtype)
    gi = cudf.from_pandas(pi)

    expect = pi.dropna()
    got = gi.dropna()

    assert_eq(expect, got)
