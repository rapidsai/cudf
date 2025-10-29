# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 40."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 40."""
    return """
    SELECT
                    w_state ,
                    i_item_id ,
                    Sum(
                    CASE
                                    WHEN (
                                                                    Cast(d_date AS DATE) < Cast ('2002-06-01' AS DATE)) THEN cs_sales_price - COALESCE(cr_refunded_cash,0)
                                    ELSE 0
                    END) AS sales_before ,
                    Sum(
                    CASE
                                    WHEN (
                                                                    Cast(d_date AS DATE) >= Cast ('2002-06-01' AS DATE)) THEN cs_sales_price - COALESCE(cr_refunded_cash,0)
                                    ELSE 0
                    END) AS sales_after
    FROM            catalog_sales
    LEFT OUTER JOIN catalog_returns
    ON              (
                                    cs_order_number = cr_order_number
                    AND             cs_item_sk = cr_item_sk) ,
                    warehouse ,
                    item ,
                    date_dim
    WHERE           i_current_price BETWEEN 0.99 AND             1.49
    AND             i_item_sk = cs_item_sk
    AND             cs_warehouse_sk = w_warehouse_sk
    AND             cs_sold_date_sk = d_date_sk
    AND             d_date BETWEEN (Cast ('2002-06-01' AS DATE) - INTERVAL '30' day) AND             (
                                    cast ('2002-06-01' AS date) + INTERVAL '30' day)
    GROUP BY        w_state,
                    i_item_id
    ORDER BY        w_state,
                    i_item_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 40."""
    # Load tables
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    # Define the target date and date range
    target_date = pl.date(2002, 6, 1)
    start_date = pl.date(2002, 5, 2)  # 2002-06-01 - 30 days
    end_date = pl.date(2002, 7, 1)  # 2002-06-01 + 30 days
    return (
        catalog_sales.join(
            catalog_returns,
            left_on=["cs_order_number", "cs_item_sk"],
            right_on=["cr_order_number", "cr_item_sk"],
            how="left",
        )  # LEFT OUTER JOIN
        .join(warehouse, left_on="cs_warehouse_sk", right_on="w_warehouse_sk")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("i_current_price").is_between(0.99, 1.49))
            & (pl.col("d_date").is_between(start_date, end_date))
        )
        .with_columns(
            [
                pl.when(pl.col("d_date") < target_date)
                .then(
                    pl.col("cs_sales_price") - pl.col("cr_refunded_cash").fill_null(0)
                )
                .otherwise(0)
                .alias("sales_before_amount"),
                pl.when(pl.col("d_date") >= target_date)
                .then(
                    pl.col("cs_sales_price") - pl.col("cr_refunded_cash").fill_null(0)
                )
                .otherwise(0)
                .alias("sales_after_amount"),
            ]
        )
        .group_by(["w_state", "i_item_id"])
        .agg(
            [
                pl.col("sales_before_amount").sum().alias("sales_before"),
                pl.col("sales_after_amount").sum().alias("sales_after"),
            ]
        )
        .sort(["w_state", "i_item_id"])
        .limit(100)
    )
