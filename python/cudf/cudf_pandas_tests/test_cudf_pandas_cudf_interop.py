# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import cudf
from cudf.pandas import LOADED

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import pandas as pd


def test_cudf_pandas_loaded_to_cudf():
    hybrid_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cudf_df = cudf.from_pandas(hybrid_df)
    pd.testing.assert_frame_equal(hybrid_df, cudf_df.to_pandas())
