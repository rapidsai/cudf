# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("axis", [None, 0, "index", 1, "columns"])
@pytest.mark.parametrize("data", [[[1, 2], [2, 3]], [1, 2], [1]])
def test_squeeze(axis, data):
    df = cudf.DataFrame(data)
    result = df.squeeze(axis=axis)
    expected = df.to_pandas().squeeze(axis=axis)
    assert_eq(result, expected)
