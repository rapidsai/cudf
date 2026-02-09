# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
from cudf.testing import assert_eq


def test_multiindex_to_arrow():
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 1, 2, 3],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "c": np.array([1, 2, 3, None, 5], dtype="datetime64[s]"),
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    pdf["a"] = pdf["a"].astype("category")
    df = cudf.from_pandas(pdf)
    gdi = cudf.MultiIndex.from_frame(df)

    expected = pa.Table.from_pandas(pdf)
    got = gdi.to_arrow()

    assert_eq(expected, got)
