# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 43."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 43."""
    return """
    SELECT s_store_name,
                   s_store_id,
                   Sum(CASE
                         WHEN ( d_day_name = 'Sunday' ) THEN ss_sales_price
                         ELSE NULL
                       END) sun_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Monday' ) THEN ss_sales_price
                         ELSE NULL
                       END) mon_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Tuesday' ) THEN ss_sales_price
                         ELSE NULL
                       END) tue_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Wednesday' ) THEN ss_sales_price
                         ELSE NULL
                       END) wed_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Thursday' ) THEN ss_sales_price
                         ELSE NULL
                       END) thu_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Friday' ) THEN ss_sales_price
                         ELSE NULL
                       END) fri_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Saturday' ) THEN ss_sales_price
                         ELSE NULL
                       END) sat_sales
    FROM   date_dim,
           store_sales,
           store
    WHERE  d_date_sk = ss_sold_date_sk
           AND s_store_sk = ss_store_sk
           AND s_gmt_offset = -5
           AND d_year = 2002
    GROUP  BY s_store_name,
              s_store_id
    ORDER  BY s_store_name,
              s_store_id,
              sun_sales,
              mon_sales,
              tue_sales,
              wed_sales,
              thu_sales,
              fri_sales,
              sat_sales
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 43."""
    # Load tables
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    # Main query with joins and conditional aggregations
    return (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter((pl.col("s_gmt_offset") == -5) & (pl.col("d_year") == 2002))
        .with_columns(
            [
                # Pre-compute conditional sales amounts for each day
                pl.when(pl.col("d_day_name") == "Sunday")
                .then(pl.col("ss_sales_price"))
                .otherwise(0)
                .alias("sun_sales_amount"),
                pl.when(pl.col("d_day_name") == "Monday")
                .then(pl.col("ss_sales_price"))
                .otherwise(0)
                .alias("mon_sales_amount"),
                pl.when(pl.col("d_day_name") == "Tuesday")
                .then(pl.col("ss_sales_price"))
                .otherwise(0)
                .alias("tue_sales_amount"),
                pl.when(pl.col("d_day_name") == "Wednesday")
                .then(pl.col("ss_sales_price"))
                .otherwise(0)
                .alias("wed_sales_amount"),
                pl.when(pl.col("d_day_name") == "Thursday")
                .then(pl.col("ss_sales_price"))
                .otherwise(0)
                .alias("thu_sales_amount"),
                pl.when(pl.col("d_day_name") == "Friday")
                .then(pl.col("ss_sales_price"))
                .otherwise(0)
                .alias("fri_sales_amount"),
                pl.when(pl.col("d_day_name") == "Saturday")
                .then(pl.col("ss_sales_price"))
                .otherwise(0)
                .alias("sat_sales_amount"),
            ]
        )
        .group_by(["s_store_name", "s_store_id"])
        .agg(
            [
                pl.col("sun_sales_amount").sum().alias("sun_sales"),
                pl.col("mon_sales_amount").sum().alias("mon_sales"),
                pl.col("tue_sales_amount").sum().alias("tue_sales"),
                pl.col("wed_sales_amount").sum().alias("wed_sales"),
                pl.col("thu_sales_amount").sum().alias("thu_sales"),
                pl.col("fri_sales_amount").sum().alias("fri_sales"),
                pl.col("sat_sales_amount").sum().alias("sat_sales"),
            ]
        )
        .select(
            [
                "s_store_name",
                "s_store_id",
                "sun_sales",
                "mon_sales",
                "tue_sales",
                "wed_sales",
                "thu_sales",
                "fri_sales",
                "sat_sales",
            ]
        )
        .sort(
            [
                "s_store_name",
                "s_store_id",
                "sun_sales",
                "mon_sales",
                "tue_sales",
                "wed_sales",
                "thu_sales",
                "fri_sales",
                "sat_sales",
            ]
        )
        .limit(100)
    )
