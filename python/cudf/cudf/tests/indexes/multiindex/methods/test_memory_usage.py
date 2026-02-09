# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf


def test_memory_usage_multi():
    # We need to sample without replacement to guarantee that the size of the
    # levels are always the same.
    rng = np.random.default_rng(seed=0)
    rows = 10
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": rng.choice(
                np.arange(rows, dtype="int64"), rows, replace=False
            ),
            "C": rng.choice(
                np.arange(rows, dtype="float64"), rows, replace=False
            ),
        }
    ).set_index(["B", "C"])
    gdf = cudf.from_pandas(df)
    # Assume MultiIndex memory footprint is just that
    # of the underlying columns, levels, and codes
    expect = rows * 16  # Source Columns
    expect += rows * 16  # Codes
    expect += rows * 8  # Level 0
    expect += rows * 8  # Level 1

    assert expect == gdf.index.memory_usage(deep=True)
