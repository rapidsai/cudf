# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 46."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 46."""
    return """
    SELECT c_last_name,
                   c_first_name,
                   ca_city,
                   bought_city,
                   ss_ticket_number,
                   amt,
                   profit
    FROM   (SELECT ss_ticket_number,
                   ss_customer_sk,
                   ca_city            bought_city,
                   Sum(ss_coupon_amt) amt,
                   Sum(ss_net_profit) profit
            FROM   store_sales,
                   date_dim,
                   store,
                   household_demographics,
                   customer_address
            WHERE  store_sales.ss_sold_date_sk = date_dim.d_date_sk
                   AND store_sales.ss_store_sk = store.s_store_sk
                   AND store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk
                   AND store_sales.ss_addr_sk = customer_address.ca_address_sk
                   AND ( household_demographics.hd_dep_count = 6
                          OR household_demographics.hd_vehicle_count = 0 )
                   AND date_dim.d_dow IN ( 6, 0 )
                   AND date_dim.d_year IN ( 2000, 2000 + 1, 2000 + 2 )
                   AND store.s_city IN ( 'Midway', 'Fairview', 'Fairview',
                                         'Fairview',
                                         'Fairview' )
            GROUP  BY ss_ticket_number,
                      ss_customer_sk,
                      ss_addr_sk,
                      ca_city) dn,
           customer,
           customer_address current_addr
    WHERE  ss_customer_sk = c_customer_sk
           AND customer.c_current_addr_sk = current_addr.ca_address_sk
           AND current_addr.ca_city <> bought_city
    ORDER  BY c_last_name,
              c_first_name,
              ca_city,
              bought_city,
              ss_ticket_number
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 46."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    # Step 1: Create the subquery (dn) equivalent
    subquery_dn = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
        .filter(
            # Demographics filter (OR condition)
            ((pl.col("hd_dep_count") == 6) | (pl.col("hd_vehicle_count") == 0))
            &
            # Weekend filter (Saturday=6, Sunday=0)
            (pl.col("d_dow").is_in([6, 0]))
            &
            # Year filter
            (pl.col("d_year").is_in([2000, 2001, 2002]))
            &
            # City filter (Fairview appears multiple times in SQL, but unique values)
            (pl.col("s_city").is_in(["Midway", "Fairview"]))
        )
        .group_by(["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "ca_city"])
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
        .with_columns([pl.col("ca_city").alias("bought_city")])
        .select(["ss_ticket_number", "ss_customer_sk", "bought_city", "amt", "profit"])
    )
    # Step 2: Join with customer and current address
    return (
        subquery_dn.join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(
            customer_address,
            left_on="c_current_addr_sk",
            right_on="ca_address_sk",
            suffix="_current",
        )
        .filter(
            # Current city != bought city (people who traveled to shop)
            pl.col("ca_city") != pl.col("bought_city")
        )
        .select(
            [
                "c_last_name",
                "c_first_name",
                "ca_city",
                "bought_city",
                "ss_ticket_number",
                "amt",
                "profit",
            ]
        )
        .sort(
            [
                "c_last_name",
                "c_first_name",
                "ca_city",
                "bought_city",
                "ss_ticket_number",
            ],
            nulls_last=True,
            descending=[False, False, False, False, False],
        )
        .limit(100)
    )
