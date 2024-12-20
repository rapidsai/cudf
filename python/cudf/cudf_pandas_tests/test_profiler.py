# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

import pytest

from cudf.pandas import LOADED, Profiler

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import numpy as np
import pandas as pd

from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="function names change across versions of pandas, so making sure it only runs on latest version of pandas",
)
def test_profiler():
    rng = np.random.default_rng(seed=42)
    with Profiler() as profiler:
        df = pd.DataFrame(
            {
                "idx": rng.integers(0, 10, 1000),
                "data": rng.random(1000),
            }
        )
        sums = df.groupby("idx").sum()
        total = df.sum()["data"]
        assert np.isclose(total, sums.sum()["data"])
        _ = pd.Timestamp(2020, 1, 1) + pd.Timedelta(1)

    per_function_stats = profiler.per_function_stats
    assert set(per_function_stats) == {
        "Timestamp",
        "DataFrame",
        "DataFrame.groupby",
        "GroupBy.sum",
        "DataFrame.sum",
        "Series.__getitem__",
        "Timedelta",
        "_Timestamp.__add__",
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
        "rng.integers",
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


def test_profiler_commandline():
    data_directory = os.path.dirname(os.path.abspath(__file__))
    # Create a copy of the current environment variables
    env = os.environ.copy()
    # Setting the 'COLUMNS' environment variable to a large number
    # because the terminal output shouldn't be compressed for
    # text validations below.
    env["COLUMNS"] = "10000"

    sp_completed = subprocess.run(
        [
            "python",
            "-m",
            "cudf.pandas",
            "--profile",
            data_directory + "/data/profile_basic.py",
        ],
        capture_output=True,
        text=True,
        env=env,
        encoding="utf-8",
    )
    assert sp_completed.returncode == 0
    output = sp_completed.stdout

    for string in [
        "Total time",
        "Stats",
        "Function",
        "GPU ncalls",
        "GPU cumtime",
        "GPU percall",
        "CPU ncalls",
        "CPU cumtime",
        "CPU percall",
    ]:
        assert string in output
