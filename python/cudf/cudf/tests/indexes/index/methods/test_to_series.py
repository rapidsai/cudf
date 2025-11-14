# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, 10, 2, 100, -10],
        ["z", "x", "a", "c", "b"],
        [-10.2, 100.1, -100.2, 0.0, 0.23],
    ],
)
def test_index_to_series(data):
    pdi = pd.Index(data)
    gdi = cudf.from_pandas(pdi)

    assert_eq(pdi.to_series(), gdi.to_series())
