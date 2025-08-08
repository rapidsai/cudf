# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 5."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 5."""
    return """
    WITH ssr AS
    (
             SELECT   s_store_id,
                      Sum(sales_price) AS sales,
                      Sum(profit)      AS profit,
                      Sum(return_amt)  AS returns1,
                      Sum(net_loss)    AS profit_loss
             FROM     (
                             SELECT ss_store_sk             AS store_sk,
                                    ss_sold_date_sk         AS date_sk,
                                    ss_ext_sales_price      AS sales_price,
                                    ss_net_profit           AS profit,
                                    Cast(0 AS DECIMAL(7,2)) AS return_amt,
                                    Cast(0 AS DECIMAL(7,2)) AS net_loss
                             FROM   store_sales
                             UNION ALL
                             SELECT sr_store_sk             AS store_sk,
                                    sr_returned_date_sk     AS date_sk,
                                    Cast(0 AS DECIMAL(7,2)) AS sales_price,
                                    Cast(0 AS DECIMAL(7,2)) AS profit,
                                    sr_return_amt           AS return_amt,
                                    sr_net_loss             AS net_loss
                             FROM   store_returns ) salesreturns,
                      date_dim,
                      store
             WHERE    date_sk = d_date_sk
             AND      d_date BETWEEN Cast('2002-08-22' AS DATE) AND      (
                               Cast('2002-08-22' AS DATE) + INTERVAL '14' day)
             AND      store_sk = s_store_sk
             GROUP BY s_store_id) , csr AS
    (
             SELECT   cp_catalog_page_id,
                      sum(sales_price) AS sales,
                      sum(profit)      AS profit,
                      sum(return_amt)  AS returns1,
                      sum(net_loss)    AS profit_loss
             FROM     (
                             SELECT cs_catalog_page_sk      AS page_sk,
                                    cs_sold_date_sk         AS date_sk,
                                    cs_ext_sales_price      AS sales_price,
                                    cs_net_profit           AS profit,
                                    cast(0 AS decimal(7,2)) AS return_amt,
                                    cast(0 AS decimal(7,2)) AS net_loss
                             FROM   catalog_sales
                             UNION ALL
                             SELECT cr_catalog_page_sk      AS page_sk,
                                    cr_returned_date_sk     AS date_sk,
                                    cast(0 AS decimal(7,2)) AS sales_price,
                                    cast(0 AS decimal(7,2)) AS profit,
                                    cr_return_amount        AS return_amt,
                                    cr_net_loss             AS net_loss
                             FROM   catalog_returns ) salesreturns,
                      date_dim,
                      catalog_page
             WHERE    date_sk = d_date_sk
             AND      d_date BETWEEN cast('2002-08-22' AS date) AND      (
                               cast('2002-08-22' AS date) + INTERVAL '14' day)
             AND      page_sk = cp_catalog_page_sk
             GROUP BY cp_catalog_page_id) , wsr AS
    (
             SELECT   web_site_id,
                      sum(sales_price) AS sales,
                      sum(profit)      AS profit,
                      sum(return_amt)  AS returns1,
                      sum(net_loss)    AS profit_loss
             FROM     (
                             SELECT ws_web_site_sk          AS wsr_web_site_sk,
                                    ws_sold_date_sk         AS date_sk,
                                    ws_ext_sales_price      AS sales_price,
                                    ws_net_profit           AS profit,
                                    cast(0 AS decimal(7,2)) AS return_amt,
                                    cast(0 AS decimal(7,2)) AS net_loss
                             FROM   web_sales
                             UNION ALL
                             SELECT          ws_web_site_sk          AS wsr_web_site_sk,
                                             wr_returned_date_sk     AS date_sk,
                                             cast(0 AS decimal(7,2)) AS sales_price,
                                             cast(0 AS decimal(7,2)) AS profit,
                                             wr_return_amt           AS return_amt,
                                             wr_net_loss             AS net_loss
                             FROM            web_returns
                             LEFT OUTER JOIN web_sales
                             ON              (
                                                             wr_item_sk = ws_item_sk
                                             AND             wr_order_number = ws_order_number) ) salesreturns,
                      date_dim,
                      web_site
             WHERE    date_sk = d_date_sk
             AND      d_date BETWEEN cast('2002-08-22' AS date) AND      (
                               cast('2002-08-22' AS date) + INTERVAL '14' day)
             AND      wsr_web_site_sk = web_site_sk
             GROUP BY web_site_id)
    SELECT
             channel ,
             id ,
             sum(sales)   AS sales ,
             sum(returns1) AS returns1 ,
             sum(profit)  AS profit
    FROM     (
                    SELECT 'store channel' AS channel ,
                           'store'
                                  || s_store_id AS id ,
                           sales ,
                           returns1 ,
                           (profit - profit_loss) AS profit
                    FROM   ssr
                    UNION ALL
                    SELECT 'catalog channel' AS channel ,
                           'catalog_page'
                                  || cp_catalog_page_id AS id ,
                           sales ,
                           returns1 ,
                           (profit - profit_loss) AS profit
                    FROM   csr
                    UNION ALL
                    SELECT 'web channel' AS channel ,
                           'web_site'
                                  || web_site_id AS id ,
                           sales ,
                           returns1 ,
                           (profit - profit_loss) AS profit
                    FROM   wsr ) x
    GROUP BY rollup (channel, id)
    ORDER BY channel ,
             id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 5."""
    # Load required tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    catalog_page = get_data(run_config.dataset_path, "catalog_page", run_config.suffix)
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)

    # Date range filter - use actual date values
    start_date = date(2002, 8, 22)
    end_date = start_date + timedelta(days=14)

    # Step 1: Create ssr CTE (Store Sales and Returns)
    # Filter sales and returns by date first, then transform
    store_sales_data = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_date").is_between(start_date, end_date, closed="both"))
        .select(
            [
                pl.col("ss_store_sk").alias("store_sk"),
                pl.col("ss_sold_date_sk").alias("date_sk"),
                pl.col("ss_ext_sales_price").alias("sales_price"),
                pl.col("ss_net_profit").alias("profit"),
                pl.lit(0.0).alias("return_amt"),
                pl.lit(0.0).alias("net_loss"),
            ]
        )
    )
    store_returns_data = (
        store_returns.join(
            date_dim, left_on="sr_returned_date_sk", right_on="d_date_sk"
        )
        .filter(pl.col("d_date").is_between(start_date, end_date, closed="both"))
        .select(
            [
                pl.col("sr_store_sk").alias("store_sk"),
                pl.col("sr_returned_date_sk").alias("date_sk"),
                pl.lit(0.0).alias("sales_price"),
                pl.lit(0.0).alias("profit"),
                pl.col("sr_return_amt").alias("return_amt"),
                pl.col("sr_net_loss").alias("net_loss"),
            ]
        )
    )
    store_salesreturns = pl.concat([store_sales_data, store_returns_data])
    ssr = (
        store_salesreturns.join(store, left_on="store_sk", right_on="s_store_sk")
        .group_by("s_store_id")
        .agg(
            [
                pl.col("sales_price").sum().alias("sales"),
                pl.col("sales_price").count().alias("sales_count"),
                pl.col("profit").sum().alias("profit"),
                pl.col("profit").count().alias("profit_count"),
                pl.col("return_amt").sum().alias("returns1"),
                pl.col("return_amt").count().alias("returns1_count"),
                pl.col("net_loss").sum().alias("profit_loss"),
                pl.col("net_loss").count().alias("profit_loss_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") > 0)
                .then(pl.col("sales"))
                .otherwise(None)
                .alias("sales"),
                pl.when(pl.col("profit_count") > 0)
                .then(pl.col("profit"))
                .otherwise(None)
                .alias("profit"),
                pl.when(pl.col("returns1_count") > 0)
                .then(pl.col("returns1"))
                .otherwise(None)
                .alias("returns1"),
                pl.when(pl.col("profit_loss_count") > 0)
                .then(pl.col("profit_loss"))
                .otherwise(None)
                .alias("profit_loss"),
            ]
        )
        .drop(["sales_count", "profit_count", "returns1_count", "profit_loss_count"])
    )

    # Step 2: Create csr CTE (Catalog Sales and Returns)
    # Filter sales and returns by date first, then transform
    catalog_sales_data = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_date").is_between(start_date, end_date, closed="both"))
        .select(
            [
                pl.col("cs_catalog_page_sk").alias("page_sk"),
                pl.col("cs_sold_date_sk").alias("date_sk"),
                pl.col("cs_ext_sales_price").alias("sales_price"),
                pl.col("cs_net_profit").alias("profit"),
                pl.lit(0.0).alias("return_amt"),
                pl.lit(0.0).alias("net_loss"),
            ]
        )
    )
    catalog_returns_data = (
        catalog_returns.join(
            date_dim, left_on="cr_returned_date_sk", right_on="d_date_sk"
        )
        .filter(pl.col("d_date").is_between(start_date, end_date, closed="both"))
        .select(
            [
                pl.col("cr_catalog_page_sk").alias("page_sk"),
                pl.col("cr_returned_date_sk").alias("date_sk"),
                pl.lit(0.0).alias("sales_price"),
                pl.lit(0.0).alias("profit"),
                pl.col("cr_return_amount").alias("return_amt"),
                pl.col("cr_net_loss").alias("net_loss"),
            ]
        )
    )
    catalog_salesreturns = pl.concat([catalog_sales_data, catalog_returns_data])
    csr = (
        catalog_salesreturns.join(
            catalog_page, left_on="page_sk", right_on="cp_catalog_page_sk"
        )
        .group_by("cp_catalog_page_id")
        .agg(
            [
                pl.col("sales_price").sum().alias("sales"),
                pl.col("sales_price").count().alias("sales_count"),
                pl.col("profit").sum().alias("profit"),
                pl.col("profit").count().alias("profit_count"),
                pl.col("return_amt").sum().alias("returns1"),
                pl.col("return_amt").count().alias("returns1_count"),
                pl.col("net_loss").sum().alias("profit_loss"),
                pl.col("net_loss").count().alias("profit_loss_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") > 0)
                .then(pl.col("sales"))
                .otherwise(None)
                .alias("sales"),
                pl.when(pl.col("profit_count") > 0)
                .then(pl.col("profit"))
                .otherwise(None)
                .alias("profit"),
                pl.when(pl.col("returns1_count") > 0)
                .then(pl.col("returns1"))
                .otherwise(None)
                .alias("returns1"),
                pl.when(pl.col("profit_loss_count") > 0)
                .then(pl.col("profit_loss"))
                .otherwise(None)
                .alias("profit_loss"),
            ]
        )
        .drop(["sales_count", "profit_count", "returns1_count", "profit_loss_count"])
    )

    # Step 3: Create wsr CTE (Web Sales and Returns)
    # Filter sales and returns by date first, then transform
    web_sales_data = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_date").is_between(start_date, end_date, closed="both"))
        .select(
            [
                pl.col("ws_web_site_sk").alias("wsr_web_site_sk"),
                pl.col("ws_sold_date_sk").alias("date_sk"),
                pl.col("ws_ext_sales_price").alias("sales_price"),
                pl.col("ws_net_profit").alias("profit"),
                pl.lit(0.0).alias("return_amt"),
                pl.lit(0.0).alias("net_loss"),
            ]
        )
    )
    # For web returns, we need the LEFT OUTER JOIN with web_sales, then filter by date
    web_returns_data = (
        web_returns.join(date_dim, left_on="wr_returned_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_date").is_between(start_date, end_date, closed="both"))
        .join(
            web_sales.select(["ws_item_sk", "ws_order_number", "ws_web_site_sk"]),
            left_on=["wr_item_sk", "wr_order_number"],
            right_on=["ws_item_sk", "ws_order_number"],
            how="left",
        )
        .select(
            [
                pl.col("ws_web_site_sk").alias("wsr_web_site_sk"),
                pl.col("wr_returned_date_sk").alias("date_sk"),
                pl.lit(0.0).alias("sales_price"),
                pl.lit(0.0).alias("profit"),
                pl.col("wr_return_amt").alias("return_amt"),
                pl.col("wr_net_loss").alias("net_loss"),
            ]
        )
    )
    web_salesreturns = pl.concat([web_sales_data, web_returns_data])
    wsr = (
        web_salesreturns.join(
            web_site, left_on="wsr_web_site_sk", right_on="web_site_sk"
        )
        .group_by("web_site_id")
        .agg(
            [
                pl.col("sales_price").sum().alias("sales"),
                pl.col("sales_price").count().alias("sales_count"),
                pl.col("profit").sum().alias("profit"),
                pl.col("profit").count().alias("profit_count"),
                pl.col("return_amt").sum().alias("returns1"),
                pl.col("return_amt").count().alias("returns1_count"),
                pl.col("net_loss").sum().alias("profit_loss"),
                pl.col("net_loss").count().alias("profit_loss_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") > 0)
                .then(pl.col("sales"))
                .otherwise(None)
                .alias("sales"),
                pl.when(pl.col("profit_count") > 0)
                .then(pl.col("profit"))
                .otherwise(None)
                .alias("profit"),
                pl.when(pl.col("returns1_count") > 0)
                .then(pl.col("returns1"))
                .otherwise(None)
                .alias("returns1"),
                pl.when(pl.col("profit_loss_count") > 0)
                .then(pl.col("profit_loss"))
                .otherwise(None)
                .alias("profit_loss"),
            ]
        )
        .drop(["sales_count", "profit_count", "returns1_count", "profit_loss_count"])
    )

    # Step 4: Create the union of all channels
    store_channel = ssr.select(
        [
            pl.lit("store channel").alias("channel"),
            (pl.lit("store") + pl.col("s_store_id").cast(pl.Utf8)).alias("id"),
            pl.col("sales"),
            pl.col("returns1"),
            (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
        ]
    )
    catalog_channel = csr.select(
        [
            pl.lit("catalog channel").alias("channel"),
            (pl.lit("catalog_page") + pl.col("cp_catalog_page_id").cast(pl.Utf8)).alias(
                "id"
            ),
            pl.col("sales"),
            pl.col("returns1"),
            (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
        ]
    )
    web_channel = wsr.select(
        [
            pl.lit("web channel").alias("channel"),
            (pl.lit("web_site") + pl.col("web_site_id").cast(pl.Utf8)).alias("id"),
            pl.col("sales"),
            pl.col("returns1"),
            (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
        ]
    )
    all_channels = pl.concat([store_channel, catalog_channel, web_channel])

    # Step 5: Group by channel and id (filter out NULL rollup rows)
    return (
        all_channels.group_by(["channel", "id"])
        .agg(
            [
                pl.col("sales").sum().alias("sales"),
                pl.col("sales").count().alias("sales_count"),
                pl.col("returns1").sum().alias("returns1"),
                pl.col("returns1").count().alias("returns1_count"),
                pl.col("profit").sum().alias("profit"),
                pl.col("profit").count().alias("profit_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") > 0)
                .then(pl.col("sales"))
                .otherwise(None)
                .alias("sales"),
                pl.when(pl.col("returns1_count") > 0)
                .then(pl.col("returns1"))
                .otherwise(None)
                .alias("returns1"),
                pl.when(pl.col("profit_count") > 0)
                .then(pl.col("profit"))
                .otherwise(None)
                .alias("profit"),
            ]
        )
        .drop(["sales_count", "returns1_count", "profit_count"])
        .filter(pl.col("channel").is_not_null() & pl.col("id").is_not_null())
        .sort(["channel", "id"])
        .limit(100)
    )
