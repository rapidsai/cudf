# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_rangeindex_append_return_rangeindex():
    idx = cudf.RangeIndex(0, 10)
    result = idx.append([])
    assert_eq(idx, result)

    result = idx.append(cudf.Index([10]))
    expected = cudf.RangeIndex(0, 11)
    assert_eq(result, expected)
