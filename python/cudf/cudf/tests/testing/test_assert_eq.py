# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from cudf.testing.testing import assert_eq


@pytest.mark.parametrize(
    "left, right",
    [
        (1493282, 1493282),
        (1493282.0, 1493282.0 + 1e-8),
        ("abc", "abc"),
        (0, np.array(0)),
        (
            np.datetime64(123456, "ns"),
            pd.Timestamp(np.datetime64(123456, "ns")),
        ),
        ("int64", np.dtype("int64")),
        (np.nan, np.nan),
    ],
)
def test_basic_scalar_equality(left, right):
    assert_eq(left, right)


@pytest.mark.parametrize(
    "left, right",
    [
        (1493282, 1493274),
        (1493282.0, 1493282.0 + 1e-6),
        ("abc", "abd"),
        (0, np.array(1)),
        (
            np.datetime64(123456, "ns"),
            pd.Timestamp(np.datetime64(123457, "ns")),
        ),
        ("int64", np.dtype("int32")),
    ],
)
def test_basic_scalar_inequality(left, right):
    with pytest.raises(AssertionError, match=r".*not (almost )?equal.*"):
        assert_eq(left, right)
