# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 38."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 38."""
    return """
    SELECT Count(*)
    FROM   (SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   store_sales,
                   date_dim,
                   customer
            WHERE  store_sales.ss_sold_date_sk = date_dim.d_date_sk
                   AND store_sales.ss_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN 1188 AND 1188 + 11
            INTERSECT
            SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   catalog_sales,
                   date_dim,
                   customer
            WHERE  catalog_sales.cs_sold_date_sk = date_dim.d_date_sk
                   AND catalog_sales.cs_bill_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN 1188 AND 1188 + 11
            INTERSECT
            SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   web_sales,
                   date_dim,
                   customer
            WHERE  web_sales.ws_sold_date_sk = date_dim.d_date_sk
                   AND web_sales.ws_bill_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN 1188 AND 1188 + 11) hot_cust
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 38."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    # Filter date_dim for the specified month sequence range
    date_filter = date_dim.filter(pl.col("d_month_seq").is_between(1188, 1188 + 11))
    # Store sales customers with names and dates
    store_customers = (
        store_sales.join(date_filter, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    # Catalog sales customers with names and dates
    catalog_customers = (
        catalog_sales.join(date_filter, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    # Web sales customers with names and dates
    web_customers = (
        web_sales.join(date_filter, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    # Find INTERSECT of all three using a different approach
    # Combine all three and find tuples that appear exactly 3 times
    all_customers = pl.concat(
        [
            store_customers.with_columns(pl.lit("store").alias("source")),
            catalog_customers.with_columns(pl.lit("catalog").alias("source")),
            web_customers.with_columns(pl.lit("web").alias("source")),
        ]
    )
    # Find combinations that appear in all three sources
    intersect_final = (
        all_customers.group_by(["c_last_name", "c_first_name", "d_date"])
        .agg(pl.col("source").n_unique().alias("source_count"))
        .filter(pl.col("source_count") == 3)
        .select(["c_last_name", "c_first_name", "d_date"])
    )
    # Count the final result
    return (
        intersect_final
        # Cast -> Int64 to match DuckDB
        .select([pl.len().cast(pl.Int64).alias("count_star()")]).limit(100)
    )
