# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from config import NUM_COLS, NUM_ROWS, cudf


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("num_cols", NUM_COLS)
@pytest.mark.parametrize(
    "index, window",
    [
        [lambda num_rows: pd.RangeIndex(num_rows), 2],
        [
            lambda num_rows: pd.date_range(
                start="1800-01-01", periods=num_rows, freq="s"
            ),
            "2s",
        ],
    ],
    ids=["int-window", "frequency-window"],
)
def bench_rolling(benchmark, num_rows, num_cols, index, window):
    def bench_func(df, window):
        return df.rolling(window=window).sum()

    df = cudf.DataFrame(
        {str(i): range(num_rows) for i in range(num_cols)},
        index=index(num_rows),
    )
    benchmark(bench_func, df, window)


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("num_cols", NUM_COLS)
@pytest.mark.parametrize("cardinality", [0.25, 0.5, 0.75])
@pytest.mark.parametrize(
    "index, window",
    [
        [lambda num_rows: pd.RangeIndex(num_rows), 2],
        [
            lambda num_rows: pd.date_range(
                start="1800-01-01", periods=num_rows, freq="s"
            ),
            "2s",
        ],
    ],
    ids=["int-window", "frequency-window"],
)
def bench_groupby_rolling(
    benchmark, num_rows, num_cols, index, window, cardinality
):
    def bench_func(df, window):
        return df.groupby("a").rolling(window=window).sum()

    df = cudf.DataFrame(
        {str(i): range(num_rows) for i in range(num_cols)},
        index=index(num_rows),
    )
    rng = np.random.default_rng(0)
    df["a"] = rng.choice(
        np.arange(int(num_rows * cardinality)), size=num_rows, replace=True
    )
    benchmark(bench_func, df, window)
