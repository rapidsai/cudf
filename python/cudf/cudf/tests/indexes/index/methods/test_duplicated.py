# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 1, 1, 3, 2, 3],
        [np.nan, 10, 15, 16, np.nan, 10, 16],
        range(0, 10),
        ["ab", "zx", None, "pq", "ab", None, "zx", None],
    ],
)
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_index_duplicated(data, keep):
    gs = cudf.Index(data)
    ps = gs.to_pandas()

    expected = ps.duplicated(keep=keep)
    actual = gs.duplicated(keep=keep)
    assert_eq(expected, actual)
