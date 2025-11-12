# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd

import cudf
from cudf.testing import assert_groupby_results_equal


def test_groupby_pipe():
    pdf = pd.DataFrame({"A": "a b a b".split(), "B": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("A").pipe(lambda x: x.max() - x.min())
    actual = gdf.groupby("A").pipe(lambda x: x.max() - x.min())

    assert_groupby_results_equal(expected, actual)
