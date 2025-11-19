# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data", [["z", "1", "a"], ["c", None, "b"], [None] * 3]
)
def test_string_sort(data, ascending):
    ps = pd.Series(data, dtype="str", name="nice name")
    gs = cudf.Series(data, dtype="str", name="nice name")

    expect = ps.sort_values(ascending=ascending)
    got = gs.sort_values(ascending=ascending)

    assert_eq(expect, got)


def test_series_sort_values_ignore_index(ignore_index):
    gsr = cudf.Series([1, 3, 5, 2, 4])
    psr = gsr.to_pandas()

    expect = psr.sort_values(ignore_index=ignore_index)
    got = gsr.sort_values(ignore_index=ignore_index)
    assert_eq(expect, got)
