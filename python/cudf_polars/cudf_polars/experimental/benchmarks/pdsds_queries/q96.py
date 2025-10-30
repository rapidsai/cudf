# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 96."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 96."""
    return """
    SELECT Count(*)
    FROM   store_sales,
           household_demographics,
           time_dim,
           store
    WHERE  ss_sold_time_sk = time_dim.t_time_sk
           AND ss_hdemo_sk = household_demographics.hd_demo_sk
           AND ss_store_sk = s_store_sk
           AND time_dim.t_hour = 15
           AND time_dim.t_minute >= 30
           AND household_demographics.hd_dep_count = 7
           AND store.s_store_name = 'ese'
    ORDER  BY Count(*)
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 96."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    return (
        store_sales.join(
            time_dim, left_on="ss_sold_time_sk", right_on="t_time_sk", how="inner"
        )
        .join(
            household_demographics,
            left_on="ss_hdemo_sk",
            right_on="hd_demo_sk",
            how="inner",
        )
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .filter(
            (pl.col("t_hour") == 15)
            & (pl.col("t_minute") >= 30)
            & (pl.col("hd_dep_count") == 7)
            & (pl.col("s_store_name") == "ese")
        )
        .select([pl.len().cast(pl.Int64).alias("count_star()")])
        .sort("count_star()", nulls_last=True)
        .limit(100)
    )
