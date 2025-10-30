# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 88."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 88."""
    return """
    select  *
    from
     (select count(*) h8_30_to_9
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 8
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = -1 and household_demographics.hd_vehicle_count<=-1+2) or
              (household_demographics.hd_dep_count = 2 and household_demographics.hd_vehicle_count<=2+2) or
              (household_demographics.hd_dep_count = 3 and household_demographics.hd_vehicle_count<=3+2))
         and store.s_store_name = 'ese') s1,
     (select count(*) h9_to_9_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 9
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = -1 and household_demographics.hd_vehicle_count<=-1+2) or
              (household_demographics.hd_dep_count = 2 and household_demographics.hd_vehicle_count<=2+2) or
              (household_demographics.hd_dep_count = 3 and household_demographics.hd_vehicle_count<=3+2))
         and store.s_store_name = 'ese') s2,
     (select count(*) h9_30_to_10
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 9
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = -1 and household_demographics.hd_vehicle_count<=-1+2) or
              (household_demographics.hd_dep_count = 2 and household_demographics.hd_vehicle_count<=2+2) or
              (household_demographics.hd_dep_count = 3 and household_demographics.hd_vehicle_count<=3+2))
         and store.s_store_name = 'ese') s3,
     (select count(*) h10_to_10_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 10
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = -1 and household_demographics.hd_vehicle_count<=-1+2) or
              (household_demographics.hd_dep_count = 2 and household_demographics.hd_vehicle_count<=2+2) or
              (household_demographics.hd_dep_count = 3 and household_demographics.hd_vehicle_count<=3+2))
         and store.s_store_name = 'ese') s4,
     (select count(*) h10_30_to_11
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 10
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = -1 and household_demographics.hd_vehicle_count<=-1+2) or
              (household_demographics.hd_dep_count = 2 and household_demographics.hd_vehicle_count<=2+2) or
              (household_demographics.hd_dep_count = 3 and household_demographics.hd_vehicle_count<=3+2))
         and store.s_store_name = 'ese') s5,
     (select count(*) h11_to_11_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 11
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = -1 and household_demographics.hd_vehicle_count<=-1+2) or
              (household_demographics.hd_dep_count = 2 and household_demographics.hd_vehicle_count<=2+2) or
              (household_demographics.hd_dep_count = 3 and household_demographics.hd_vehicle_count<=3+2))
         and store.s_store_name = 'ese') s6,
     (select count(*) h11_30_to_12
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 11
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = -1 and household_demographics.hd_vehicle_count<=-1+2) or
              (household_demographics.hd_dep_count = 2 and household_demographics.hd_vehicle_count<=2+2) or
              (household_demographics.hd_dep_count = 3 and household_demographics.hd_vehicle_count<=3+2))
         and store.s_store_name = 'ese') s7,
     (select count(*) h12_to_12_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 12
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = -1 and household_demographics.hd_vehicle_count<=-1+2) or
              (household_demographics.hd_dep_count = 2 and household_demographics.hd_vehicle_count<=2+2) or
              (household_demographics.hd_dep_count = 3 and household_demographics.hd_vehicle_count<=3+2))
         and store.s_store_name = 'ese') s8;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 88."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    hd_filter = (
        ((pl.col("hd_dep_count") == -1) & (pl.col("hd_vehicle_count") <= 1))
        | ((pl.col("hd_dep_count") == 2) & (pl.col("hd_vehicle_count") <= 4))
        | ((pl.col("hd_dep_count") == 3) & (pl.col("hd_vehicle_count") <= 5))
    )
    base_query = (
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
            hd_filter
            & (pl.col("s_store_name").is_not_null() & (pl.col("s_store_name") == "ese"))
        )
    )
    return base_query.select(
        [
            pl.when((pl.col("t_hour") == 8) & (pl.col("t_minute") >= 30))
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias("h8_30_to_9"),
            pl.when((pl.col("t_hour") == 9) & (pl.col("t_minute") < 30))
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias("h9_to_9_30"),
            pl.when((pl.col("t_hour") == 9) & (pl.col("t_minute") >= 30))
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias("h9_30_to_10"),
            pl.when((pl.col("t_hour") == 10) & (pl.col("t_minute") < 30))
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias("h10_to_10_30"),
            pl.when((pl.col("t_hour") == 10) & (pl.col("t_minute") >= 30))
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias("h10_30_to_11"),
            pl.when((pl.col("t_hour") == 11) & (pl.col("t_minute") < 30))
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias("h11_to_11_30"),
            pl.when((pl.col("t_hour") == 11) & (pl.col("t_minute") >= 30))
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias("h11_30_to_12"),
            pl.when((pl.col("t_hour") == 12) & (pl.col("t_minute") < 30))
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias("h12_to_12_30"),
        ]
    )
