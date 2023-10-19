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

from cudf.pandas import LOADED, Profiler

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import numpy as np
import pandas as pd


def test_profiler():
    np.random.seed(42)
    with Profiler() as profiler:
        df = pd.DataFrame(
            {
                "idx": np.random.randint(0, 10, 1000),
                "data": np.random.rand(1000),
            }
        )
        sums = df.groupby("idx").sum()
        total = df.sum()["data"]
        assert np.isclose(total, sums.sum()["data"])

    per_function_stats = profiler.per_function_stats
    assert set(per_function_stats) == {
        "DataFrame",
        "DataFrame.groupby",
        "DataFrameGroupBy.sum",
        "DataFrame.sum",
        "Series.__getitem__",
    }
    assert all(len(func["cpu"]) == 0 for func in per_function_stats.values())

    per_line_stats = profiler.per_line_stats
    calls = [
        "pd.DataFrame",
        "np.random.randint",
        "np.random.rand",
        'df.groupby("idx").sum',
        'df.sum()["data"]',
        "np.isclose",
    ]
    for line_stats, call in zip(per_line_stats, calls):
        # Check that the expected function calls were recorded.
        assert call in line_stats[1]
        # No CPU time
        assert line_stats[3] == 0
