# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "ascending,expected_data",
    [
        (True, [1, 2, 0]),
        (False, [0, 2, 1]),
    ],
)
def test_dataframe_argsort(ascending, expected_data):
    actual = cudf.DataFrame({"a": [10, 0, 2], "b": [-10, 10, 1]}).argsort(
        ascending=ascending
    )
    expected = cp.array(expected_data, dtype="int32")

    assert_eq(actual, expected)
