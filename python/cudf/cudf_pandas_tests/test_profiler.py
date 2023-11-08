# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        _ = pd.Timestamp(2020, 1, 1) + pd.Timedelta(1)

    per_function_stats = profiler.per_function_stats
    assert set(per_function_stats) == {
        "DataFrame",
        "DataFrame.groupby",
        "DataFrameGroupBy.sum",
        "DataFrame.sum",
        "Series.__getitem__",
    }
    for name, func in per_function_stats.items():
        assert (
            len(func["cpu"]) == 0
            if "Time" not in name
            else len(func["gpu"]) == 0
        )

    per_line_stats = profiler.per_line_stats
    calls = [
        "pd.DataFrame",
        "",
        "np.random.randint",
        "np.random.rand",
        'df.groupby("idx").sum',
        'df.sum()["data"]',
        "np.isclose",
        "pd.Timestamp",
    ]
    for line_stats, call in zip(per_line_stats, calls):
        # Check that the expected function calls were recorded.
        assert call in line_stats[1]
        # No CPU time
        assert line_stats[3] == 0 if "Time" not in call else line_stats[2] == 0


def test_profiler_hasattr_exception():
    with Profiler():
        df = pd.DataFrame({"data": [1, 2, 3]})
        hasattr(df, "this_does_not_exist")


def test_profiler_fast_slow_name_mismatch():
    with Profiler():
        df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
        df.iloc[0, 1] = "foo"
