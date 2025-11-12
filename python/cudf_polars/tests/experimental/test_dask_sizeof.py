# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from dask.sizeof import sizeof

import polars as pl

from cudf_polars.containers import DataFrame
from cudf_polars.experimental.dask_registers import register
from cudf_polars.utils.cuda_stream import get_dask_cuda_stream

# Must register sizeof dispatch before running tests
register()


@pytest.mark.parametrize(
    "polars_tbl, size",
    [
        (pl.DataFrame(), 0),
        (pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}), 9 * 8),
        (pl.DataFrame({"a": [1, 2, 3]}), 3 * 8),
        (pl.DataFrame({"a": ["a"], "b": ["bc"]}), 2 * 8 + 3),
        (pl.DataFrame({"a": [1, 2, None]}), 88),
    ],
)
def test_dask_sizeof(polars_tbl, size):
    df = DataFrame.from_polars(polars_tbl, stream=get_dask_cuda_stream())
    assert sizeof(df) == size
    assert sum(sizeof(c) for c in df.columns) == size
