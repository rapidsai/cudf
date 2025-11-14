# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf
from cudf.testing import assert_groupby_results_equal


def test_groupby_select_then_ffill():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [1, None, None, 2, None],
            "c": [3, None, None, 4, None],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].ffill()
    actual = gdf.groupby("a")["c"].ffill()

    assert_groupby_results_equal(expected, actual)
