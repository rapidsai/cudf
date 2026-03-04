# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data,index",
    [
        ([1, 2, 3], [10, 11, 12]),
        ([1, 2, 3, 1, 1, 2, 3, 2], [10, 20, 23, 24, 25, 26, 27, 28]),
        ([1, None, 2, None, 3, None, 3, 1], [5, 6, 7, 8, 9, 10, 11, 12]),
        ([np.nan, 1.0, np.nan, 5.4, 5.4, 1.0], ["a", "b", "c", "d", "e", "f"]),
        (
            ["lama", "cow", "lama", None, "beetle", "lama", None, None],
            [1, 4, 10, 11, 2, 100, 200, 400],
        ),
    ],
)
@pytest.mark.parametrize("keep", ["first", "last", False])
@pytest.mark.parametrize("name", [None, "a"])
def test_series_duplicated(data, index, keep, name):
    gs = cudf.Series(data, index=index, name=name)
    ps = gs.to_pandas()

    assert_eq(gs.duplicated(keep=keep), ps.duplicated(keep=keep))
