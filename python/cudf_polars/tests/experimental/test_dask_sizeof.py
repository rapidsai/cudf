# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
import pytest
from dask.sizeof import sizeof

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.experimental.dask_registers import register

# Must register sizeof dispatch before running tests
register()


@pytest.mark.parametrize(
    "arrow_tbl, size",
    [
        (pa.table([]), 0),
        (pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}), 9 * 8),
        (pa.table({"a": [1, 2, 3]}), 3 * 8),
        (pa.table({"a": ["a"], "b": ["bc"]}), 2 * 8 + 3),
        (pa.table({"a": [1, 2, None]}), 88),
    ],
)
def test_dask_sizeof(arrow_tbl, size):
    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    df = DataFrame.from_table(plc_tbl, names=arrow_tbl.column_names)
    assert sizeof(df) == size
    assert sum(sizeof(c) for c in df.columns) == size
