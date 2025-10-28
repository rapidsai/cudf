# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 79."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 79."""
    return """
    SELECT c_last_name,
                   c_first_name,
                   Substr(s_city, 1, 30),
                   ss_ticket_number,
                   amt,
                   profit
    FROM   (SELECT ss_ticket_number,
                   ss_customer_sk,
                   store.s_city,
                   Sum(ss_coupon_amt) amt,
                   Sum(ss_net_profit) profit
            FROM   store_sales,
                   date_dim,
                   store,
                   household_demographics
            WHERE  store_sales.ss_sold_date_sk = date_dim.d_date_sk
                   AND store_sales.ss_store_sk = store.s_store_sk
                   AND store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk
                   AND ( household_demographics.hd_dep_count = 8
                          OR household_demographics.hd_vehicle_count > 4 )
                   AND date_dim.d_dow = 1
                   AND date_dim.d_year IN ( 2000, 2000 + 1, 2000 + 2 )
                   AND store.s_number_employees BETWEEN 200 AND 295
            GROUP  BY ss_ticket_number,
                      ss_customer_sk,
                      ss_addr_sk,
                      store.s_city) ms,
           customer
    WHERE  ss_customer_sk = c_customer_sk
    ORDER  BY c_last_name,
              c_first_name,
              Substr(s_city, 1, 30),
              profit
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 79."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    ms = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk")
        .filter(
            ((pl.col("hd_dep_count") == 8) | (pl.col("hd_vehicle_count") > 4))
            & (pl.col("d_dow") == 1)
            & (pl.col("d_year").is_in([2000, 2001, 2002]))
            & (pl.col("s_number_employees").is_between(200, 295))
        )
        .group_by(["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "s_city"])
        .agg(
            [
                pl.col("ss_coupon_amt").sum().alias("amt_sum"),
                pl.col("ss_coupon_amt").count().alias("amt_count"),
                pl.col("ss_net_profit").sum().alias("profit_sum"),
                pl.col("ss_net_profit").count().alias("profit_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("amt_count") > 0)
                .then(pl.col("amt_sum"))
                .otherwise(None)
                .alias("amt"),
                pl.when(pl.col("profit_count") > 0)
                .then(pl.col("profit_sum"))
                .otherwise(None)
                .alias("profit"),
            ]
        )
        .drop(["amt_sum", "amt_count", "profit_sum", "profit_count"])
    )
    return (
        ms.join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .select(
            [
                "c_last_name",
                "c_first_name",
                pl.col("s_city").str.slice(0, 30).alias("substr(s_city, 1, 30)"),
                "ss_ticket_number",
                "amt",
                "profit",
            ]
        )
        .sort(
            ["c_last_name", "c_first_name", "substr(s_city, 1, 30)", "profit"],
            nulls_last=True,
        )
        .limit(100)
    )
