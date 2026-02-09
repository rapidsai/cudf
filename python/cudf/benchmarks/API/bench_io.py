# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for IO operations."""

import os
from tempfile import TemporaryDirectory

import pytest
from config import NUM_ROWS

import cudf


@pytest.mark.parametrize("num_rows", NUM_ROWS)
def bench_read_parquet_with_filters(benchmark, num_rows):
    df = cudf.DataFrame(
        {
            "x": cudf.Series(range(num_rows), dtype="int32"),
            "y": cudf.Series(range(num_rows, 2 * num_rows), dtype="float64"),
            "z": cudf.Series(
                ["a", "b", "c", "d"] * (num_rows // 4), dtype="str"
            ),
        }
    )

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "filtered.parquet")
        df.to_parquet(path)

        threshold = num_rows // 2

        filters = [
            [("x", ">", threshold), ("z", "in", ["a", "b"])],
            [("y", "<", threshold), ("z", "not in", ["c"])],
        ]

        benchmark(cudf.read_parquet, path, filters=filters)
