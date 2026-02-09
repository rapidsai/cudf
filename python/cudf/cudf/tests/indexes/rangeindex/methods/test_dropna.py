# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_rangeindex_dropna():
    ri = cudf.RangeIndex(range(2))
    result = ri.dropna()
    expected = ri.copy()
    assert_eq(result, expected)
