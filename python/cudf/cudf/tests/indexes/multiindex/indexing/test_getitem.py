# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_multiindex_getitem():
    pidx = pd.MultiIndex(
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
    pidx.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gidx = cudf.from_pandas(pidx)
    assert_eq(pidx[0], gidx[0])


@pytest.mark.parametrize(
    "key",
    [0, 1, [], [0, 1], slice(None), slice(0, 0), slice(0, 1), slice(0, 2)],
)
def test_multiindex_indexing(key):
    gi = cudf.MultiIndex.from_frame(
        cudf.DataFrame({"a": [1, 2, 3], "b": [True, False, False]})
    )
    pi = gi.to_pandas()

    assert_eq(gi[key], pi[key], exact=False)
