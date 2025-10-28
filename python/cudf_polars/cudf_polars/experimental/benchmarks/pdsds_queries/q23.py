# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 23."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 23."""
    return """
    WITH frequent_ss_items
         AS (SELECT Substr(i_item_desc, 1, 30) itemdesc,
                    i_item_sk                  item_sk,
                    d_date                     solddate,
                    Count(*)                   cnt
             FROM   store_sales,
                    date_dim,
                    item
             WHERE  ss_sold_date_sk = d_date_sk
                    AND ss_item_sk = i_item_sk
                    AND d_year IN ( 1998, 1998 + 1, 1998 + 2, 1998 + 3 )
             GROUP  BY Substr(i_item_desc, 1, 30),
                       i_item_sk,
                       d_date
             HAVING Count(*) > 4),
         max_store_sales
         AS (SELECT Max(csales) tpcds_cmax
             FROM   (SELECT c_customer_sk,
                            Sum(ss_quantity * ss_sales_price) csales
                     FROM   store_sales,
                            customer,
                            date_dim
                     WHERE  ss_customer_sk = c_customer_sk
                            AND ss_sold_date_sk = d_date_sk
                            AND d_year IN ( 1998, 1998 + 1, 1998 + 2, 1998 + 3 )
                     GROUP  BY c_customer_sk)),
         best_ss_customer
         AS (SELECT c_customer_sk,
                    Sum(ss_quantity * ss_sales_price) ssales
             FROM   store_sales,
                    customer
             WHERE  ss_customer_sk = c_customer_sk
             GROUP  BY c_customer_sk
             HAVING Sum(ss_quantity * ss_sales_price) >
                    ( 95 / 100.0 ) * (SELECT *
                                      FROM   max_store_sales))
    SELECT Sum(sales)
    FROM   (SELECT cs_quantity * cs_list_price sales
            FROM   catalog_sales,
                   date_dim
            WHERE  d_year = 1998
                   AND d_moy = 6
                   AND cs_sold_date_sk = d_date_sk
                   AND cs_item_sk IN (SELECT item_sk
                                      FROM   frequent_ss_items)
                   AND cs_bill_customer_sk IN (SELECT c_customer_sk
                                               FROM   best_ss_customer)
            UNION ALL
            SELECT ws_quantity * ws_list_price sales
            FROM   web_sales,
                   date_dim
            WHERE  d_year = 1998
                   AND d_moy = 6
                   AND ws_sold_date_sk = d_date_sk
                   AND ws_item_sk IN (SELECT item_sk
                                      FROM   frequent_ss_items)
                   AND ws_bill_customer_sk IN (SELECT c_customer_sk
                                               FROM   best_ss_customer)) LIMIT 100;

    WITH frequent_ss_items
         AS (SELECT Substr(i_item_desc, 1, 30) itemdesc,
                    i_item_sk                  item_sk,
                    d_date                     solddate,
                    Count(*)                   cnt
             FROM   store_sales,
                    date_dim,
                    item
             WHERE  ss_sold_date_sk = d_date_sk
                    AND ss_item_sk = i_item_sk
                    AND d_year IN ( 1998, 1998 + 1, 1998 + 2, 1998 + 3 )
             GROUP  BY Substr(i_item_desc, 1, 30),
                       i_item_sk,
                       d_date
             HAVING Count(*) > 4),
         max_store_sales
         AS (SELECT Max(csales) tpcds_cmax
             FROM   (SELECT c_customer_sk,
                            Sum(ss_quantity * ss_sales_price) csales
                     FROM   store_sales,
                            customer,
                            date_dim
                     WHERE  ss_customer_sk = c_customer_sk
                            AND ss_sold_date_sk = d_date_sk
                            AND d_year IN ( 1998, 1998 + 1, 1998 + 2, 1998 + 3 )
                     GROUP  BY c_customer_sk)),
         best_ss_customer
         AS (SELECT c_customer_sk,
                    Sum(ss_quantity * ss_sales_price) ssales
             FROM   store_sales,
                    customer
             WHERE  ss_customer_sk = c_customer_sk
             GROUP  BY c_customer_sk
             HAVING Sum(ss_quantity * ss_sales_price) >
                    ( 95 / 100.0 ) * (SELECT *
                                      FROM   max_store_sales))
    SELECT c_last_name,
                   c_first_name,
                   sales
    FROM   (SELECT c_last_name,
                   c_first_name,
                   Sum(cs_quantity * cs_list_price) sales
            FROM   catalog_sales,
                   customer,
                   date_dim
            WHERE  d_year = 1998
                   AND d_moy = 6
                   AND cs_sold_date_sk = d_date_sk
                   AND cs_item_sk IN (SELECT item_sk
                                      FROM   frequent_ss_items)
                   AND cs_bill_customer_sk IN (SELECT c_customer_sk
                                               FROM   best_ss_customer)
                   AND cs_bill_customer_sk = c_customer_sk
            GROUP  BY c_last_name,
                      c_first_name
            UNION ALL
            SELECT c_last_name,
                   c_first_name,
                   Sum(ws_quantity * ws_list_price) sales
            FROM   web_sales,
                   customer,
                   date_dim
            WHERE  d_year = 1998
                   AND d_moy = 6
                   AND ws_sold_date_sk = d_date_sk
                   AND ws_item_sk IN (SELECT item_sk
                                      FROM   frequent_ss_items)
                   AND ws_bill_customer_sk IN (SELECT c_customer_sk
                                               FROM   best_ss_customer)
                   AND ws_bill_customer_sk = c_customer_sk
            GROUP  BY c_last_name,
                      c_first_name)
    ORDER  BY c_last_name,
              c_first_name,
              sales
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 23."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)

    # Step 1: Build frequent_ss_items (items sold frequently in store sales)
    frequent_ss_items = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter(pl.col("d_year").is_in([1998, 1999, 2000, 2001]))
        .with_columns([pl.col("i_item_desc").str.slice(0, 30).alias("itemdesc")])
        .group_by(["itemdesc", "ss_item_sk", "d_date"])
        .agg([pl.len().alias("cnt")])
        .filter(pl.col("cnt") > 4)
        .select("ss_item_sk")
        .unique()
    )

    # Step 2: Build best_ss_customer (high-value store sales customers)
    # First calculate customer sales totals
    customer_sales = (
        store_sales.join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_in([1998, 1999, 2000, 2001]))
        .group_by("ss_customer_sk")
        .agg([(pl.col("ss_quantity") * pl.col("ss_sales_price")).sum().alias("csales")])
    )

    # Calculate threshold (95% of max customer sales)
    max_sales_table = customer_sales.select(pl.col("csales").max().alias("max_sales"))
    threshold_table = max_sales_table.with_columns(
        (pl.col("max_sales") * 0.95).alias("threshold")
    ).select("threshold")

    # Get customers above threshold
    best_customers = (
        store_sales.join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .group_by("ss_customer_sk")
        .agg([(pl.col("ss_quantity") * pl.col("ss_sales_price")).sum().alias("ssales")])
        .join(threshold_table, how="cross")
        .filter(pl.col("ssales") > pl.col("threshold"))
        .select("ss_customer_sk")
        .unique()
    )

    # Step 3: Main query - Catalog sales part
    catalog_part = (
        catalog_sales.join(
            customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk"
        )
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(
            frequent_ss_items, left_on="cs_item_sk", right_on="ss_item_sk", how="semi"
        )
        .join(
            best_customers,
            left_on="cs_bill_customer_sk",
            right_on="ss_customer_sk",
            how="semi",
        )
        .filter((pl.col("d_year") == 1998) & (pl.col("d_moy") == 6))
        .group_by(["c_last_name", "c_first_name"])
        .agg([(pl.col("cs_quantity") * pl.col("cs_list_price")).sum().alias("sales")])
    )

    # Step 4: Main query - Web sales part
    web_part = (
        web_sales.join(
            customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk"
        )
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(
            frequent_ss_items, left_on="ws_item_sk", right_on="ss_item_sk", how="semi"
        )
        .join(
            best_customers,
            left_on="ws_bill_customer_sk",
            right_on="ss_customer_sk",
            how="semi",
        )
        .filter((pl.col("d_year") == 1998) & (pl.col("d_moy") == 6))
        .group_by(["c_last_name", "c_first_name"])
        .agg([(pl.col("ws_quantity") * pl.col("ws_list_price")).sum().alias("sales")])
    )

    # Step 5: Combine results
    return (
        pl.concat([catalog_part, web_part])
        .sort(["c_last_name", "c_first_name", "sales"])
        .limit(100)
    )
