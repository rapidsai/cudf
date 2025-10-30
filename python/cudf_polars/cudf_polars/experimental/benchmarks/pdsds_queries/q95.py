# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 95."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 95."""
    return """
    WITH ws_wh AS
    (
           SELECT ws1.ws_order_number,
                  ws1.ws_warehouse_sk wh1,
                  ws2.ws_warehouse_sk wh2
           FROM   web_sales ws1,
                  web_sales ws2
           WHERE  ws1.ws_order_number = ws2.ws_order_number
           AND    ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk)
    SELECT
             Count(DISTINCT ws_order_number) AS 'order count' ,
             Sum(ws_ext_ship_cost)           AS 'total shipping cost' ,
             Sum(ws_net_profit)              AS 'total net profit'
    FROM     web_sales ws1 ,
             date_dim ,
             customer_address ,
             web_site
    WHERE    d_date BETWEEN '2000-4-01' AND      (
                      Cast('2000-4-01' AS DATE) + INTERVAL '60' day)
    AND      ws1.ws_ship_date_sk = d_date_sk
    AND      ws1.ws_ship_addr_sk = ca_address_sk
    AND      ca_state = 'IN'
    AND      ws1.ws_web_site_sk = web_site_sk
    AND      web_company_name = 'pri'
    AND      ws1.ws_order_number IN
             (
                    SELECT ws_order_number
                    FROM   ws_wh)
    AND      ws1.ws_order_number IN
             (
                    SELECT wr_order_number
                    FROM   web_returns,
                           ws_wh
                    WHERE  wr_order_number = ws_wh.ws_order_number)
    ORDER BY count(DISTINCT ws_order_number)
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 95."""
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    start_date = (pl.date(2000, 4, 1)).cast(pl.Datetime("us"))
    end_date = (start_date + pl.duration(days=60)).cast(pl.Datetime("us"))
    multi_warehouse_orders = (
        web_sales.group_by("ws_order_number")
        .agg([pl.col("ws_warehouse_sk").n_unique().alias("warehouse_count")])
        .filter(pl.col("warehouse_count") > 1)
        .select("ws_order_number")
    )
    returned_multi_warehouse_orders = (
        web_returns.select("wr_order_number")
        .unique()
        .join(
            multi_warehouse_orders,
            left_on="wr_order_number",
            right_on="ws_order_number",
            how="inner",
        )
        .select("wr_order_number")
    )
    return (
        web_sales.join(
            date_dim, left_on="ws_ship_date_sk", right_on="d_date_sk", how="inner"
        )
        .join(
            customer_address,
            left_on="ws_ship_addr_sk",
            right_on="ca_address_sk",
            how="inner",
        )
        .join(web_site, left_on="ws_web_site_sk", right_on="web_site_sk", how="inner")
        .filter(
            (pl.col("d_date") >= start_date)
            & (pl.col("d_date") <= end_date)
            & (pl.col("ca_state") == "IN")
            & (pl.col("web_company_name") == "pri")
        )
        .join(
            multi_warehouse_orders,
            left_on="ws_order_number",
            right_on="ws_order_number",
            how="inner",
        )
        .join(
            returned_multi_warehouse_orders,
            left_on="ws_order_number",
            right_on="wr_order_number",
            how="inner",
        )
        .select(
            [
                pl.col("ws_order_number")
                .n_unique()
                .cast(pl.Int64)
                .alias("order count"),
                pl.col("ws_ext_ship_cost").sum().alias("total shipping cost"),
                pl.col("ws_net_profit").sum().alias("total net profit"),
            ]
        )
        .sort("order count", nulls_last=True)
        .limit(100)
    )
