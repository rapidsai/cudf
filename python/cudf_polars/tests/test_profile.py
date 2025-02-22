# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl


def test_profile_columns() -> None:
    df = pl.LazyFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    q = df.group_by("a", maintain_order=True).agg(pl.all().sum()).sort("a")

    profiling_info = q.profile(engine="gpu")
    # ┌────────────────┬───────┬──────┐
    # │ node           ┆ start ┆ end  │
    # │ ---            ┆ ---   ┆ ---  │
    # │ str            ┆ u64   ┆ u64  │
    # ╞════════════════╪═══════╪══════╡
    # │ optimization   ┆ 0     ┆ 147  │
    # │ dataframe_scan ┆ 147   ┆ 2881 │
    # │ group_by       ┆ 2907  ┆ 6907 │
    # │ sort           ┆ 6951  ┆ 7621 │
    # └────────────────┴───────┴──────┘

    assert len(profiling_info) == 2
    assert profiling_info[1].columns == ["node", "start", "end"]
    assert profiling_info[1].select(pl.col("node")).to_arrow().to_pydict()["node"] == [
        "optimization",
        "dataframe_scan",
        "group_by",
        "sort",
    ]
