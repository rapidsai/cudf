# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 87."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 87."""
    return """
    select count(*)
    from ((select distinct c_last_name, c_first_name, d_date
           from store_sales, date_dim, customer
           where store_sales.ss_sold_date_sk = date_dim.d_date_sk
             and store_sales.ss_customer_sk = customer.c_customer_sk
             and d_month_seq between 1188 and 1188+11)
           except
          (select distinct c_last_name, c_first_name, d_date
           from catalog_sales, date_dim, customer
           where catalog_sales.cs_sold_date_sk = date_dim.d_date_sk
             and catalog_sales.cs_bill_customer_sk = customer.c_customer_sk
             and d_month_seq between 1188 and 1188+11)
           except
          (select distinct c_last_name, c_first_name, d_date
           from web_sales, date_dim, customer
           where web_sales.ws_sold_date_sk = date_dim.d_date_sk
             and web_sales.ws_bill_customer_sk = customer.c_customer_sk
             and d_month_seq between 1188 and 1188+11)
    ) cool_cust;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 87."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_customers = (
        store_sales.filter(
            pl.col("ss_sold_date_sk").is_not_null()
            & pl.col("ss_customer_sk").is_not_null()
        )
        .join(
            date_dim.filter(
                (pl.col("d_month_seq") >= 1188) & (pl.col("d_month_seq") <= 1188 + 11)
            ),
            left_on="ss_sold_date_sk",
            right_on="d_date_sk",
            how="inner",
        )
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk", how="inner")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    catalog_customers = (
        catalog_sales.filter(
            pl.col("cs_sold_date_sk").is_not_null()
            & pl.col("cs_bill_customer_sk").is_not_null()
        )
        .join(
            date_dim.filter(
                (pl.col("d_month_seq") >= 1188) & (pl.col("d_month_seq") <= 1188 + 11)
            ),
            left_on="cs_sold_date_sk",
            right_on="d_date_sk",
            how="inner",
        )
        .join(
            customer,
            left_on="cs_bill_customer_sk",
            right_on="c_customer_sk",
            how="inner",
        )
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    web_customers = (
        web_sales.filter(
            pl.col("ws_sold_date_sk").is_not_null()
            & pl.col("ws_bill_customer_sk").is_not_null()
        )
        .join(
            date_dim.filter(
                (pl.col("d_month_seq") >= 1188) & (pl.col("d_month_seq") <= 1188 + 11)
            ),
            left_on="ws_sold_date_sk",
            right_on="d_date_sk",
            how="inner",
        )
        .join(
            customer,
            left_on="ws_bill_customer_sk",
            right_on="c_customer_sk",
            how="inner",
        )
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    store_customers_sentinel = store_customers.with_columns(
        [
            pl.col("c_last_name").fill_null("NULL_SENTINEL_LAST"),
            pl.col("c_first_name").fill_null("NULL_SENTINEL_FIRST"),
        ]
    )
    catalog_customers_sentinel = catalog_customers.with_columns(
        [
            pl.col("c_last_name").fill_null("NULL_SENTINEL_LAST"),
            pl.col("c_first_name").fill_null("NULL_SENTINEL_FIRST"),
        ]
    )
    web_customers_sentinel = web_customers.with_columns(
        [
            pl.col("c_last_name").fill_null("NULL_SENTINEL_LAST"),
            pl.col("c_first_name").fill_null("NULL_SENTINEL_FIRST"),
        ]
    )
    result_after_first_except = store_customers_sentinel.join(
        catalog_customers_sentinel,
        on=["c_last_name", "c_first_name", "d_date"],
        how="anti",
    ).unique()
    result_after_second_except = (
        result_after_first_except.join(
            web_customers_sentinel,
            on=["c_last_name", "c_first_name", "d_date"],
            how="anti",
        )
        .with_columns(
            [
                pl.when(pl.col("c_last_name") == "NULL_SENTINEL_LAST")
                .then(None)
                .otherwise(pl.col("c_last_name"))
                .alias("c_last_name"),
                pl.when(pl.col("c_first_name") == "NULL_SENTINEL_FIRST")
                .then(None)
                .otherwise(pl.col("c_first_name"))
                .alias("c_first_name"),
            ]
        )
        .unique()
    )
    return result_after_second_except.select(
        [pl.len().cast(pl.Int64).alias("count_star()")]
    )
