# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 76."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 76."""
    return """
    SELECT channel,
                   col_name,
                   d_year,
                   d_qoy,
                   i_category,
                   Count(*)             sales_cnt,
                   Sum(ext_sales_price) sales_amt
    FROM   (SELECT 'store'            AS channel,
                   'ss_hdemo_sk'      col_name,
                   d_year,
                   d_qoy,
                   i_category,
                   ss_ext_sales_price ext_sales_price
            FROM   store_sales,
                   item,
                   date_dim
            WHERE  ss_hdemo_sk IS NULL
                   AND ss_sold_date_sk = d_date_sk
                   AND ss_item_sk = i_item_sk
            UNION ALL
            SELECT 'web'              AS channel,
                   'ws_ship_hdemo_sk' col_name,
                   d_year,
                   d_qoy,
                   i_category,
                   ws_ext_sales_price ext_sales_price
            FROM   web_sales,
                   item,
                   date_dim
            WHERE  ws_ship_hdemo_sk IS NULL
                   AND ws_sold_date_sk = d_date_sk
                   AND ws_item_sk = i_item_sk
            UNION ALL
            SELECT 'catalog'          AS channel,
                   'cs_warehouse_sk'  col_name,
                   d_year,
                   d_qoy,
                   i_category,
                   cs_ext_sales_price ext_sales_price
            FROM   catalog_sales,
                   item,
                   date_dim
            WHERE  cs_warehouse_sk IS NULL
                   AND cs_sold_date_sk = d_date_sk
                   AND cs_item_sk = i_item_sk) foo
    GROUP  BY channel,
              col_name,
              d_year,
              d_qoy,
              i_category
    ORDER  BY channel,
              col_name,
              d_year,
              d_qoy,
              i_category
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 76."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_component = (
        store_sales.filter(pl.col("ss_hdemo_sk").is_null())
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .select(
            [
                pl.lit("store").alias("channel"),
                pl.lit("ss_hdemo_sk").alias("col_name"),
                "d_year",
                "d_qoy",
                "i_category",
                pl.col("ss_ext_sales_price").alias("ext_sales_price"),
            ]
        )
    )
    web_component = (
        web_sales.filter(pl.col("ws_ship_hdemo_sk").is_null())
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .select(
            [
                pl.lit("web").alias("channel"),
                pl.lit("ws_ship_hdemo_sk").alias("col_name"),
                "d_year",
                "d_qoy",
                "i_category",
                pl.col("ws_ext_sales_price").alias("ext_sales_price"),
            ]
        )
    )
    catalog_component = (
        catalog_sales.filter(pl.col("cs_warehouse_sk").is_null())
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .select(
            [
                pl.lit("catalog").alias("channel"),
                pl.lit("cs_warehouse_sk").alias("col_name"),
                "d_year",
                "d_qoy",
                "i_category",
                pl.col("cs_ext_sales_price").alias("ext_sales_price"),
            ]
        )
    )
    return (
        pl.concat([store_component, web_component, catalog_component])
        .group_by(["channel", "col_name", "d_year", "d_qoy", "i_category"])
        .agg(
            [
                pl.len().cast(pl.Int64).alias("sales_cnt"),
                pl.when(pl.col("ext_sales_price").count() > 0)
                .then(pl.col("ext_sales_price").sum())
                .otherwise(None)
                .alias("sales_amt"),
            ]
        )
        .select(
            [
                "channel",
                "col_name",
                "d_year",
                "d_qoy",
                "i_category",
                "sales_cnt",
                "sales_amt",
            ]
        )
        .sort(["channel", "col_name", "d_year", "d_qoy", "i_category"], nulls_last=True)
        .limit(100)
    )
