# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 54."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 54."""
    return """
    WITH my_customers
         AS (SELECT DISTINCT c_customer_sk,
                             c_current_addr_sk
             FROM   (SELECT cs_sold_date_sk     sold_date_sk,
                            cs_bill_customer_sk customer_sk,
                            cs_item_sk          item_sk
                     FROM   catalog_sales
                     UNION ALL
                     SELECT ws_sold_date_sk     sold_date_sk,
                            ws_bill_customer_sk customer_sk,
                            ws_item_sk          item_sk
                     FROM   web_sales) cs_or_ws_sales,
                    item,
                    date_dim,
                    customer
             WHERE  sold_date_sk = d_date_sk
                    AND item_sk = i_item_sk
                    AND i_category = 'Sports'
                    AND i_class = 'fitness'
                    AND c_customer_sk = cs_or_ws_sales.customer_sk
                    AND d_moy = 5
                    AND d_year = 2000),
         my_revenue
         AS (SELECT c_customer_sk,
                    Sum(ss_ext_sales_price) AS revenue
             FROM   my_customers,
                    store_sales,
                    customer_address,
                    store,
                    date_dim
             WHERE  c_current_addr_sk = ca_address_sk
                    AND ca_county = s_county
                    AND ca_state = s_state
                    AND ss_sold_date_sk = d_date_sk
                    AND c_customer_sk = ss_customer_sk
                    AND d_month_seq BETWEEN (SELECT DISTINCT d_month_seq + 1
                                             FROM   date_dim
                                             WHERE  d_year = 2000
                                                    AND d_moy = 5) AND
                                            (SELECT DISTINCT
                                            d_month_seq + 3
                                             FROM   date_dim
                                             WHERE  d_year = 2000
                                                    AND d_moy = 5)
             GROUP  BY c_customer_sk),
         segments
         AS (SELECT Cast(( revenue / 50 ) AS INT) AS segment
             FROM   my_revenue)
    SELECT segment,
                   Count(*)     AS num_customers,
                   segment * 50 AS segment_base
    FROM   segments
    GROUP  BY segment
    ORDER  BY segment,
              num_customers
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 54."""
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    cs_sales = catalog_sales.select(
        [
            pl.col("cs_sold_date_sk").alias("sold_date_sk"),
            pl.col("cs_bill_customer_sk").alias("customer_sk"),
            pl.col("cs_item_sk").alias("item_sk"),
        ]
    )
    ws_sales = web_sales.select(
        [
            pl.col("ws_sold_date_sk").alias("sold_date_sk"),
            pl.col("ws_bill_customer_sk").alias("customer_sk"),
            pl.col("ws_item_sk").alias("item_sk"),
        ]
    )
    cs_or_ws_sales = pl.concat([cs_sales, ws_sales])
    my_customers = (
        cs_or_ws_sales.join(date_dim, left_on="sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="item_sk", right_on="i_item_sk")
        .join(customer, left_on="customer_sk", right_on="c_customer_sk")
        .filter(
            (pl.col("i_category") == "Sports")
            & (pl.col("i_class") == "fitness")
            & (pl.col("d_moy") == 5)
            & (pl.col("d_year") == 2000)
        )
        .select([pl.col("customer_sk").alias("c_customer_sk"), "c_current_addr_sk"])
        .unique()
    )

    my_revenue = (
        my_customers.join(
            customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
        )
        .join(
            store, left_on=["ca_county", "ca_state"], right_on=["s_county", "s_state"]
        )
        .join(store_sales, left_on="c_customer_sk", right_on="ss_customer_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_month_seq").is_between(1206, 1208))
        .group_by("c_customer_sk")
        .agg([pl.col("ss_ext_sales_price").sum().alias("revenue")])
    )

    segments = my_revenue.with_columns(
        (pl.col("revenue") / 50.0).round(0).cast(pl.Int32).alias("segment")
    ).select("segment")

    return (
        segments.group_by("segment")
        .agg([pl.len().alias("num_customers")])
        .with_columns((pl.col("segment") * 50).alias("segment_base"))
        .select(
            [
                "segment",
                pl.col("num_customers").cast(pl.Int64),
                "segment_base",
            ]
        )
        .sort(
            ["segment", "num_customers"],
            nulls_last=True,
            descending=[False, False],
        )
        .limit(100)
    )
