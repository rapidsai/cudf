# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 1, None, None],
        [None, None, 3.2, 1, None, None],
        [None, "a", "3.2", "z", None, None],
        pd.Series(["a", "b", None], dtype="category"),
        np.array([1, 2, 3, None], dtype="datetime64[s]"),
    ],
)
def test_index_to_arrow(data):
    pdi = pd.Index(data)
    gdi = cudf.Index(data)

    expected_arrow_array = pa.Array.from_pandas(pdi)
    got_arrow_array = gdi.to_arrow()

    assert_eq(expected_arrow_array, got_arrow_array)
