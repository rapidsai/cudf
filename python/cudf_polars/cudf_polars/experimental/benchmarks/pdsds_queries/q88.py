# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 88."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 88."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=88,
        qualification=run_config.qualification,
    )

    s_store_name = params["s_store_name"]
    hd1 = params["hd_dep_count1"]
    hd2 = params["hd_dep_count2"]
    hd3 = params["hd_dep_count3"]

    return f"""
    select  *
    from
     (select count(*) h8_30_to_9
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 8
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s1,
     (select count(*) h9_to_9_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 9
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s2,
     (select count(*) h9_30_to_10
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 9
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s3,
     (select count(*) h10_to_10_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 10
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s4,
     (select count(*) h10_30_to_11
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 10
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s5,
     (select count(*) h11_to_11_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 11
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s6,
     (select count(*) h11_30_to_12
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 11
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s7,
     (select count(*) h12_to_12_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 12
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s8;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 88."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=88,
        qualification=run_config.qualification,
    )

    s_store_name = params["s_store_name"]
    hd1 = params["hd_dep_count1"]
    hd2 = params["hd_dep_count2"]
    hd3 = params["hd_dep_count3"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    hd_filter = (
        ((pl.col("hd_dep_count") == hd1) & (pl.col("hd_vehicle_count") <= hd1 + 2))
        | ((pl.col("hd_dep_count") == hd2) & (pl.col("hd_vehicle_count") <= hd2 + 2))
        | ((pl.col("hd_dep_count") == hd3) & (pl.col("hd_vehicle_count") <= hd3 + 2))
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
            & (
                pl.col("s_store_name").is_not_null()
                & (pl.col("s_store_name") == s_store_name)
            )
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
