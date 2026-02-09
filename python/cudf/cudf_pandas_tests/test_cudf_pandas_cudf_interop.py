# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
from cudf.pandas import LOADED

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import pandas as pd


def test_cudf_pandas_loaded_to_cudf():
    hybrid_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cudf_df = cudf.from_pandas(hybrid_df)
    pd.testing.assert_frame_equal(hybrid_df, cudf_df.to_pandas())
