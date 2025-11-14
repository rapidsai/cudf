# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_multiindex_take():
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex.take([0]), gdfIndex.take([0]))
    assert_eq(pdfIndex.take(np.array([0])), gdfIndex.take(np.array([0])))
    assert_eq(pdfIndex.take(pd.Series([0])), gdfIndex.take(cudf.Series([0])))
    assert_eq(pdfIndex.take([0, 1]), gdfIndex.take([0, 1]))
    assert_eq(pdfIndex.take(np.array([0, 1])), gdfIndex.take(np.array([0, 1])))
    assert_eq(
        pdfIndex.take(pd.Series([0, 1])), gdfIndex.take(cudf.Series([0, 1]))
    )
